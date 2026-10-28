import argparse
import json
import logging
import os
import random
import socket
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NÓ-SENSOR] %(message)s",
    datefmt="%H:%M:%S",
)


class ClienteTelemetria:
    """
    Cliente de rede do Nó Sensor Distribuído.
    Suporta envio via Socket TCP persistente (NDJSON) ou API REST HTTP POST.
    """

    def __init__(self, node_id: str, protocolo: str = "tcp", host: str = "127.0.0.1", port: int = 5050):
        self.node_id = node_id
        self.protocolo = protocolo.lower()
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._buffer = ""

    def enviar_leitura(self, veiculos: int, pedestres: int, animais: int = 0) -> dict | None:
        payload = {
            "node_id": self.node_id,
            "veiculos": int(veiculos),
            "pedestres": int(pedestres),
            "animais": int(animais),
            "timestamp": time.time(),
        }
        if self.protocolo == "http":
            return self._enviar_http(payload)
        return self._enviar_tcp(payload)

    def _enviar_tcp(self, payload: dict) -> dict | None:
        try:
            if self._sock is None:
                self._sock = socket.create_connection((self.host, self.port), timeout=2.5)
            msg = (json.dumps(payload) + "\n").encode("utf-8")
            self._sock.sendall(msg)
            while "\n" not in self._buffer:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Conexão fechada pelo servidor")
                self._buffer += chunk.decode("utf-8")
            linha, self._buffer = self._buffer.split("\n", 1)
            return json.loads(linha)
        except Exception as e:
            logging.warning(f"Falha na transmissão TCP ({e}). Reconectando no próximo ciclo...")
            self.fechar()
            return None

    def _enviar_http(self, payload: dict) -> dict | None:
        url = f"http://{self.host}:{self.port}/api/telemetria"
        dados = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=dados, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logging.warning(f"Falha na requisição HTTP POST ({e}).")
            return None

    def fechar(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._buffer = ""


def rodar_modo_simulacao(cliente: ClienteTelemetria, intervalo: float, ciclos: int = 0):
    """Simula cenários realistas de tráfego (fluxo intenso, pedestres, animal na via)."""
    cenarios = [
        (8, 0, 0, "Tráfego moderado de veículos"),
        (14, 0, 0, "Horário de pico - fila de carros"),
        (4, 3, 0, "Pedestres aguardando na faixa"),
        (2, 0, 1, "Alerta: Animal (cachorro) detectado na pista"),
        (0, 0, 0, "Via livre / madrugada"),
    ]
    i = 0
    while True:
        veic, ped, anim, desc = cenarios[i % len(cenarios)]
        # Adiciona leve variação aleatória nos veículos
        veic_atual = max(0, veic + random.randint(-1, 2)) if veic > 0 else 0
        resposta = cliente.enviar_leitura(veic_atual, ped, anim)
        if resposta:
            logging.info(
                f"[{cliente.node_id}] ({desc}) Enviado: V={veic_atual}, P={ped}, A={anim} -> "
                f" Controlador: Estado={resposta.get('estado')} | MetaVerde={resposta.get('tempo_verde_meta_s')}s"
            )
        i += 1
        if ciclos > 0 and i >= ciclos:
            break
        time.sleep(intervalo)


def rodar_modo_yolo(cliente: ClienteTelemetria, source: str, model_path: str, intervalo: float):
    """Executa inferência YOLOv8 real no nó de borda (Edge Node) e envia telemetria ao servidor."""
    import cv2
    from core.detector import DetectorTrafego

    fonte = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        logging.error(f"Não foi possível abrir fonte de vídeo {source}. Alternando para modo simulação.")
        rodar_modo_simulacao(cliente, intervalo)
        return

    detector = DetectorTrafego(model_path=model_path, confianca=0.35, pular_frames=3, imgsz=320)
    roi = [(0.15, 0.30), (0.85, 0.30), (0.98, 0.98), (0.02, 0.98)]
    ultimo_envio = 0.0

    logging.info(f"Nó de visão YOLOv8 iniciado na fonte '{source}'")
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(fonte, str) and os.path.exists(fonte):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        resultado = detector.processar_frame(frame, roi_normalizada=roi)
        agora = time.time()
        if agora - ultimo_envio >= intervalo:
            resp = cliente.enviar_leitura(
                veiculos=resultado["veiculos"],
                pedestres=resultado["pedestres"],
                animais=resultado["animais"],
            )
            if resp:
                logging.info(
                    f"[{cliente.node_id}] YOLOv8: V={resultado['veiculos']} P={resultado['pedestres']} A={resultado['animais']} -> "
                    f"Estado Central: {resp.get('estado')}"
                )
            ultimo_envio = agora

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Nó Sensor Distribuído (Edge Node YOLOv8 / Simulador)")
    parser.add_argument("--node-id", default="camera_norte_01", help="Identificador único deste nó sensor")
    parser.add_argument("--protocolo", choices=["tcp", "http"], default="tcp", help="Protocolo de transporte")
    parser.add_argument("--host", default="127.0.0.1", help="IP do Servidor Central")
    parser.add_argument("--port", type=int, default=5050, help="Porta do Servidor Central (5050 TCP ou 8080 HTTP)")
    parser.add_argument("--simulate", action="store_true", help="Roda gerador de tráfego sintético sem exigir vídeo/GPU")
    parser.add_argument("--source", default=str(ROOT_DIR / "transito.mp4"), help="Caminho do vídeo ou webcam")
    parser.add_argument("--yolo", default=str(ROOT_DIR / "yolov8n.pt"), help="Caminho do modelo YOLOv8")
    parser.add_argument("--intervalo", type=float, default=1.0, help="Intervalo em segundos entre pacotes de rede")
    args = parser.parse_args()

    cliente = ClienteTelemetria(node_id=args.node_id, protocolo=args.protocolo, host=args.host, port=args.port)
    try:
        if args.simulate:
            rodar_modo_simulacao(cliente, args.intervalo)
        else:
            rodar_modo_yolo(cliente, args.source, args.yolo, args.intervalo)
    except KeyboardInterrupt:
        logging.info("Encerrando Nó Sensor...")
    finally:
        cliente.fechar()


if __name__ == "__main__":
    main()
