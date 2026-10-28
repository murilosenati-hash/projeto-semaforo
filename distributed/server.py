import argparse
import json
import logging
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Garante importação do pacote core mesmo rodando direto da pasta distributed/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.fsm import ControladorSemaforoFSM, EstadoSemaforo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SERVIDOR-CENTRAL] %(message)s",
    datefmt="%H:%M:%S",
)


class ControladorDistribuido:
    """
    Gerencia o estado global do cruzamento agregando leituras de múltiplos
    nós sensores (câmeras/radares) via rede e executando o ciclo da FSM.
    """

    def __init__(self, fsm: ControladorSemaforoFSM | None = None, node_timeout: float = 6.0):
        self.fsm = fsm or ControladorSemaforoFSM()
        self.node_timeout = node_timeout
        self.lock = threading.Lock()
        self.nos_sensores: dict[str, dict] = {}
        self.rodando = False
        self._tick_thread: threading.Thread | None = None

    def registrar_telemetria(self, payload: dict) -> dict:
        """
        Recebe pacote JSON de um nó sensor:
        {
            "node_id": "cam_norte",
            "veiculos": 5,
            "pedestres": 1,
            "animais": 0,
            "timestamp": 1700000000.0
        }
        """
        node_id = str(payload.get("node_id", "sensor_anonimo"))
        veiculos = max(0, int(payload.get("veiculos", 0)))
        pedestres = max(0, int(payload.get("pedestres", 0)))
        animais = max(0, int(payload.get("animais", 0)))
        agora = time.time()

        with self.lock:
            self.nos_sensores[node_id] = {
                "veiculos": veiculos,
                "pedestres": pedestres,
                "animais": animais,
                "ultimo_contato": agora,
                "timestamp_origem": payload.get("timestamp", agora),
            }
            self._agregar_e_atualizar_fsm(agora)
            self.fsm.tick()
            return self.obter_status_Unsafe(agora)

    def _agregar_e_atualizar_fsm(self, agora: float) -> None:
        """Soma as detecções apenas dos nós ativos (dentro da janela de timeout)."""
        nos_ativos = {
            nid: dados
            for nid, dados in self.nos_sensores.items()
            if (agora - dados["ultimo_contato"]) <= self.node_timeout
        }
        if not nos_ativos:
            return

        total_veic = sum(d["veiculos"] for d in nos_ativos.values())
        total_ped = sum(d["pedestres"] for d in nos_ativos.values())
        total_anim = sum(d["animais"] for d in nos_ativos.values())
        self.fsm.atualizar_sensores(total_veic, total_ped, total_anim)

    def tick_periodico(self) -> None:
        with self.lock:
            agora = time.time()
            self._agregar_e_atualizar_fsm(agora)
            self.fsm.tick()

    def obter_status(self) -> dict:
        with self.lock:
            return self.obter_status_Unsafe(time.time())

    def obter_status_Unsafe(self, agora: float) -> dict:
        carro_r, carro_y, carro_g, ped_r, ped_g = self.fsm.obter_luzes()
        tempo_estado = round(agora - self.fsm.inicio_estado, 2)
        nos_info = {
            nid: {
                "veiculos": d["veiculos"],
                "pedestres": d["pedestres"],
                "animais": d["animais"],
                "idade_s": round(agora - d["ultimo_contato"], 2),
                "ativo": (agora - d["ultimo_contato"]) <= self.node_timeout,
            }
            for nid, d in self.nos_sensores.items()
        }
        return {
            "status": "ok",
            "estado": self.fsm.estado_atual.value,
            "modo_failsafe": self.fsm.modo_failsafe,
            "tempo_no_estado_s": tempo_estado,
            "tempo_verde_meta_s": self.fsm.tempo_verde_calculado,
            "totais": {
                "veiculos": self.fsm.veiculos_detectados,
                "pedestres": self.fsm.pedestres_detectados,
                "animais": self.fsm.animais_detectados,
            },
            "luzes_digitais": {
                "carro_vermelho": carro_r,
                "carro_amarelo": carro_y,
                "carro_verde": carro_g,
                "pedestre_vermelho": ped_r,
                "pedestre_verde": ped_g,
            },
            "nos_sensores": nos_info,
            "timestamp_servidor": agora,
        }


def iniciar_servidor_socket(controlador: ControladorDistribuido, host: str, port: int, stop_event: threading.Event):
    """Servidor TCP Socket (protocolo JSON delimitado por quebra de linha - NDJSON)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    srv.settimeout(0.5)
    logging.info(f"Servidor TCP Socket escutando em {host}:{port}")

    def tratar_cliente(conn: socket.socket, addr):
        logging.info(f"Conexão TCP recebida de {addr}")
        with conn:
            buffer = ""
            while not stop_event.is_set():
                try:
                    dados = conn.recv(4096)
                    if not dados:
                        break
                    buffer += dados.decode("utf-8")
                    while "\n" in buffer:
                        linha, buffer = buffer.split("\n", 1)
                        linha = linha.strip()
                        if not linha:
                            continue
                        pacote = json.loads(linha)
                        if pacote.get("acao") == "status":
                            resposta = controlador.obter_status()
                        else:
                            resposta = controlador.registrar_telemetria(pacote)
                        conn.sendall((json.dumps(resposta) + "\n").encode("utf-8"))
                except (ConnectionResetError, BrokenPipeError):
                    break
                except json.JSONDecodeError as e:
                    erro = {"status": "erro", "detalhe": f"JSON inválido: {e}"}
                    conn.sendall((json.dumps(erro) + "\n").encode("utf-8"))
                except Exception as e:
                    logging.error(f"Erro no cliente {addr}: {e}")
                    break

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
            t = threading.Thread(target=tratar_cliente, args=(conn, addr), daemon=True)
            t.start()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"Erro no accept TCP: {e}")
            break

    srv.close()


def criar_handler_http(controlador: ControladorDistribuido):
    class SemaforoHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Silencia log HTTP verboso a cada request para manter console limpo
            pass

        def _enviar_json(self, codigo: int, payload: dict):
            corpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            if self.path in ("/", "/api/status", "/health"):
                self._enviar_json(200, controlador.obter_status())
            else:
                self._enviar_json(404, {"status": "erro", "detalhe": "Rota não encontrada"})

        def do_POST(self):
            if self.path == "/api/telemetria":
                try:
                    tam = int(self.headers.get("Content-Length", 0))
                    corpo = self.rfile.read(tam).decode("utf-8")
                    payload = json.loads(corpo) if corpo else {}
                    resposta = controlador.registrar_telemetria(payload)
                    self._enviar_json(200, resposta)
                except Exception as e:
                    self._enviar_json(400, {"status": "erro", "detalhe": str(e)})
            else:
                self._enviar_json(404, {"status": "erro", "detalhe": "Rota não encontrada"})

    return SemaforoHTTPHandler


def main():
    parser = argparse.ArgumentParser(description="Servidor Central Distribuído do Semáforo Inteligente")
    parser.add_argument("--host", default="127.0.0.1", help="Endereço IP de bind (padrão: 127.0.0.1)")
    parser.add_argument("--tcp-port", type=int, default=5050, help="Porta do servidor TCP Socket (padrão: 5050)")
    parser.add_argument("--http-port", type=int, default=8080, help="Porta da API REST HTTP (padrão: 8080)")
    args = parser.parse_args()

    controlador = ControladorDistribuido()
    stop_event = threading.Event()

    # Thread para o servidor TCP Socket
    t_socket = threading.Thread(
        target=iniciar_servidor_socket,
        args=(controlador, args.host, args.tcp_port, stop_event),
        daemon=True,
    )
    t_socket.start()

    # Thread para a API REST HTTP
    httpd = HTTPServer((args.host, args.http_port), criar_handler_http(controlador))
    t_http = threading.Thread(target=httpd.serve_forever, daemon=True)
    t_http.start()
    logging.info(f"API REST HTTP escutando em http://{args.host}:{args.http_port}/api/status")

    ultimo_estado = None
    try:
        while True:
            controlador.tick_periodico()
            status = controlador.obter_status()
            if status["estado"] != ultimo_estado:
                logging.info(
                    f"TRANSIÇÃO DE ESTADO -> {status['estado']} | "
                    f"Veículos={status['totais']['veiculos']} | "
                    f"Pedestres={status['totais']['pedestres']} | "
                    f"Animais={status['totais']['animais']} | "
                    f"Luzes={status['luzes_digitais']}"
                )
                ultimo_estado = status["estado"]
            time.sleep(0.2)
    except KeyboardInterrupt:
        logging.info("Encerrando Servidor Central Distribuído...")
        stop_event.set()
        httpd.shutdown()


if __name__ == "__main__":
    main()
