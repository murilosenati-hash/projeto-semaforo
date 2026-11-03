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


HTML_DISPLAY = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Semáforo Inteligente — Display Web</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }
    body {
      background-color: #080a0f;
      color: #f1f5f9;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-height: 100vh;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    #main-container {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 5vw;
      width: 100vw;
      height: 100vh;
      padding: 2vw;
      transition: transform 0.4s ease, gap 0.4s ease;
    }

    .traffic-box {
      background: linear-gradient(145deg, #18191f, #0d0e12);
      border: 4px solid #2d313d;
      border-radius: 40px;
      padding: 30px 24px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.9), inset 0 2px 4px rgba(255,255,255,0.05);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 22px;
      position: relative;
      transition: all 0.3s ease;
    }

    .traffic-box.horizontal {
      flex-direction: row;
      padding: 24px 30px;
    }

    .box-title {
      font-size: 1.1rem;
      font-weight: 800;
      letter-spacing: 3px;
      color: #94a3b8;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    .lamp-housing {
      position: relative;
      width: 22vh;
      height: 22vh;
      max-width: 180px;
      max-height: 180px;
      min-width: 110px;
      min-height: 110px;
      background: radial-gradient(circle, #1a1a24 60%, #0a0a0f 100%);
      border-radius: 50%;
      border: 5px solid #282b35;
      box-shadow: inset 0 6px 14px rgba(0,0,0,0.95);
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .lamp-glow {
      width: 86%;
      height: 86%;
      border-radius: 50%;
      background: #1e1e24;
      transition: all 0.15s ease-out;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .lamp-red.active {
      background: radial-gradient(circle, #ff4d4d 20%, #cc0000 85%);
      box-shadow: 0 0 35px #ff2222, 0 0 70px rgba(255, 34, 34, 0.6), inset 0 0 15px #ffffff;
    }
    .lamp-red.inactive {
      background: #330808;
      box-shadow: none;
    }

    .lamp-yellow.active {
      background: radial-gradient(circle, #ffea6b 20%, #e5a000 85%);
      box-shadow: 0 0 35px #ffcc00, 0 0 70px rgba(255, 204, 0, 0.6), inset 0 0 15px #ffffff;
    }
    .lamp-yellow.inactive {
      background: #362905;
      box-shadow: none;
    }

    .lamp-green.active {
      background: radial-gradient(circle, #4df07a 20%, #00a83e 85%);
      box-shadow: 0 0 35px #00ff66, 0 0 70px rgba(0, 255, 102, 0.6), inset 0 0 15px #ffffff;
    }
    .lamp-green.inactive {
      background: #042911;
      box-shadow: none;
    }

    .ped-icon {
      width: 62%;
      height: 62%;
      fill: #222;
      transition: all 0.15s ease;
    }

    .ped-red.active .ped-icon {
      fill: #ff3333;
      filter: drop-shadow(0 0 16px #ff2222) drop-shadow(0 0 30px #ff2222);
    }
    .ped-red.active {
      background: #3d0c0c;
      box-shadow: inset 0 0 20px rgba(255,30,30,0.5);
    }

    .ped-green.active .ped-icon {
      fill: #10f065;
      filter: drop-shadow(0 0 16px #10f065) drop-shadow(0 0 30px #10f065);
    }
    .ped-green.active {
      background: #0a3316;
      box-shadow: inset 0 0 20px rgba(16,240,101,0.5);
    }

    @keyframes piscaAlerta {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.1; }
    }
    .piscando {
      animation: piscaAlerta 0.5s infinite;
    }

    .timer-badge {
      font-family: 'Courier New', Courier, monospace;
      font-size: 2.2rem;
      font-weight: 900;
      color: #38bdf8;
      background: #040810;
      border: 2px solid #1e293b;
      border-radius: 12px;
      padding: 6px 18px;
      letter-spacing: 2px;
      box-shadow: inset 0 2px 6px rgba(0,0,0,0.8);
      margin-top: 6px;
    }

    .sub-telemetria {
      font-size: 0.85rem;
      color: #64748b;
      margin-top: 4px;
      font-weight: 600;
    }

    #drawer-toggle-area {
      position: fixed;
      top: 0;
      right: 0;
      width: 140px;
      height: 90px;
      z-index: 100;
    }

    #control-drawer {
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid #334155;
      border-radius: 20px;
      padding: 18px 22px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.7);
      z-index: 101;
      display: flex;
      flex-direction: column;
      gap: 14px;
      transition: opacity 0.4s ease, transform 0.4s ease;
      opacity: 0;
      pointer-events: none;
      transform: translateY(-15px);
    }

    #control-drawer.visible {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }

    .drawer-header {
      font-size: 0.85rem;
      font-weight: 800;
      color: #38bdf8;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .btn-group {
      display: flex;
      gap: 8px;
    }

    .btn {
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 8px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: all 0.2s;
    }
    .btn:hover {
      background: #334155;
      border-color: #38bdf8;
      color: #ffffff;
    }
    .btn.active {
      background: #0284c7;
      border-color: #38bdf8;
      color: #ffffff;
    }

    .btn-full {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      background: #0f766e;
      border-color: #14b8a6;
    }
    .btn-full:hover {
      background: #115e59;
    }

    #hint-pill {
      position: fixed;
      top: 14px;
      right: 14px;
      background: rgba(30, 41, 59, 0.7);
      color: #94a3b8;
      border: 1px solid #334155;
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 0.75rem;
      pointer-events: none;
      transition: opacity 0.3s;
      z-index: 99;
    }
  </style>
</head>
<body>

  <div id="hint-pill">⚙ Passe o mouse aqui</div>
  <div id="drawer-toggle-area"></div>

  <div id="control-drawer">
    <div class="drawer-header">
      <span>Painel do Semáforo</span>
      <span id="conn-status" style="color: #10b981; font-size: 0.7rem;">● ONLINE</span>
    </div>

    <div>
      <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 5px;">MODO DE EXIBIÇÃO</div>
      <div class="btn-group">
        <button class="btn active" id="btn-mode-both" onclick="setMode('both')">Ambos</button>
        <button class="btn" id="btn-mode-car" onclick="setMode('car')">Veículos</button>
        <button class="btn" id="btn-mode-ped" onclick="setMode('ped')">Pedestres</button>
      </div>
    </div>

    <div>
      <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 5px;">ORIENTAÇÃO</div>
      <div class="btn-group">
        <button class="btn active" id="btn-ori-vert" onclick="setOrientation('vert')">Vertical</button>
        <button class="btn" id="btn-ori-horiz" onclick="setOrientation('horiz')">Horizontal</button>
      </div>
    </div>

    <button class="btn btn-full" onclick="toggleFullScreen()">⛶ Tela Cheia</button>
  </div>

  <div id="main-container">
    <!-- Semáforo Veicular -->
    <div class="traffic-box" id="box-veiculos">
      <div class="box-title">VEÍCULOS</div>
      
      <div class="lamp-housing">
        <div class="lamp-glow lamp-red inactive" id="v-red"></div>
      </div>
      <div class="lamp-housing">
        <div class="lamp-glow lamp-yellow inactive" id="v-yellow"></div>
      </div>
      <div class="lamp-housing">
        <div class="lamp-glow lamp-green active" id="v-green"></div>
      </div>

      <div class="timer-badge" id="timer-box">--</div>
      <div class="sub-telemetria" id="estado-sub">ESTADO: VERDE</div>
    </div>

    <!-- Semáforo de Pedestres -->
    <div class="traffic-box" id="box-pedestres">
      <div class="box-title">PEDESTRES</div>
      
      <div class="lamp-housing">
        <div class="lamp-glow ped-red active" id="p-red">
          <svg class="ped-icon" viewBox="0 0 24 24">
            <circle cx="12" cy="4.5" r="2.5" />
            <path d="M10 8h4c1.1 0 2 .9 2 2v5h-2v6h-1.5v-6h-1v6H10v-6H8v-5c0-1.1.9-2 2-2z" />
          </svg>
        </div>
      </div>

      <div class="lamp-housing">
        <div class="lamp-glow ped-green inactive" id="p-green">
          <svg class="ped-icon" viewBox="0 0 24 24">
            <circle cx="13.5" cy="4" r="2.2" />
            <path d="M13.5 7.5c-.3 0-.6.1-.9.3l-2.4 1.5c-.5.3-.8.9-.8 1.5v3.2h2v-2.3l1.3-.8v3.9l-2.6 4.3 1.7 1 2.9-4.8v-4.5l1.6 1.4c.4.3.9.5 1.4.5h1.3v-2h-1.1l-2.4-2.1c-.5-.4-1.2-.6-2-.6z" />
          </svg>
        </div>
      </div>

      <div class="timer-badge" id="ped-timer-box">--</div>
      <div class="sub-telemetria" id="ped-sub">TRAVESSIA</div>
    </div>
  </div>

  <script>
    let currentMode = 'both';
    let currentOrientation = 'vert';
    let hideDrawerTimeout = null;
    const drawer = document.getElementById('control-drawer');
    const hintPill = document.getElementById('hint-pill');

    function showDrawer() {
      drawer.classList.add('visible');
      hintPill.style.opacity = '0';
      clearTimeout(hideDrawerTimeout);
    }

    function scheduleHideDrawer() {
      clearTimeout(hideDrawerTimeout);
      hideDrawerTimeout = setTimeout(() => {
        drawer.classList.remove('visible');
        hintPill.style.opacity = '0.7';
      }, 2500);
    }

    document.getElementById('drawer-toggle-area').addEventListener('mouseenter', showDrawer);
    drawer.addEventListener('mouseenter', showDrawer);
    drawer.addEventListener('mouseleave', scheduleHideDrawer);
    document.addEventListener('mousemove', (e) => {
      if (e.clientX > window.innerWidth - 180 && e.clientY < 120) {
        showDrawer();
      }
    });

    function setMode(mode) {
      currentMode = mode;
      document.getElementById('btn-mode-both').classList.toggle('active', mode === 'both');
      document.getElementById('btn-mode-car').classList.toggle('active', mode === 'car');
      document.getElementById('btn-mode-ped').classList.toggle('active', mode === 'ped');

      const boxV = document.getElementById('box-veiculos');
      const boxP = document.getElementById('box-pedestres');

      if (mode === 'both') {
        boxV.style.display = 'flex';
        boxP.style.display = 'flex';
      } else if (mode === 'car') {
        boxV.style.display = 'flex';
        boxP.style.display = 'none';
      } else if (mode === 'ped') {
        boxV.style.display = 'none';
        boxP.style.display = 'flex';
      }
    }

    function setOrientation(ori) {
      currentOrientation = ori;
      document.getElementById('btn-ori-vert').classList.toggle('active', ori === 'vert');
      document.getElementById('btn-ori-horiz').classList.toggle('active', ori === 'horiz');

      const boxV = document.getElementById('box-veiculos');
      const boxP = document.getElementById('box-pedestres');

      if (ori === 'horiz') {
        boxV.classList.add('horizontal');
        boxP.classList.add('horizontal');
      } else {
        boxV.classList.remove('horizontal');
        boxP.classList.remove('horizontal');
      }
    }

    function toggleFullScreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => console.log(err));
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }
      }
    }

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('Falha no status');
        const data = await res.json();

        document.getElementById('conn-status').textContent = '● ONLINE';
        document.getElementById('conn-status').style.color = '#10b981';

        // 1. Luzes Veículo
        const vR = document.getElementById('v-red');
        const vY = document.getElementById('v-yellow');
        const vG = document.getElementById('v-green');

        const luzesCarro = (data.luzes && data.luzes.carro) ? data.luzes.carro : {
            vermelho: data.luzes_digitais ? data.luzes_digitais.carro_vermelho : false,
            amarelo: data.luzes_digitais ? data.luzes_digitais.carro_amarelo : false,
            verde: data.luzes_digitais ? data.luzes_digitais.carro_verde : false
        };

        vR.className = 'lamp-glow lamp-red ' + (luzesCarro.vermelho ? 'active' : 'inactive');
        vY.className = 'lamp-glow lamp-yellow ' + (luzesCarro.amarelo ? 'active' : 'inactive');
        vG.className = 'lamp-glow lamp-green ' + (luzesCarro.verde ? 'active' : 'inactive');

        // 2. Luzes Pedestre
        const pR = document.getElementById('p-red');
        const pG = document.getElementById('p-green');
        const luzesPed = (data.luzes && data.luzes.pedestre) ? data.luzes.pedestre : {
            vermelho: data.luzes_digitais ? data.luzes_digitais.pedestre_vermelho : false,
            verde: data.luzes_digitais ? data.luzes_digitais.pedestre_verde : false,
            piscando: data.estado === 'VERMELHO_PISCANDO_PEDESTRE'
        };
        const isPiscando = Boolean(luzesPed.piscando || data.estado === 'VERMELHO_PISCANDO_PEDESTRE');

        pR.className = 'lamp-glow ped-red ' + (luzesPed.vermelho ? 'active' : 'inactive') + (isPiscando ? ' piscando' : '');
        pG.className = 'lamp-glow ped-green ' + (luzesPed.verde ? 'active' : 'inactive');

        // 3. Temporizadores e Labels
        let tRest = data.tempo_restante !== undefined ? data.tempo_restante : (data.tempo_verde_meta_s ? Math.max(0, data.tempo_verde_meta_s - (data.tempo_no_estado_s || 0)) : 0);
        tRest = Math.max(0, Math.ceil(tRest));
        const tText = tRest < 10 ? '0' + tRest : '' + tRest;
        document.getElementById('timer-box').textContent = tText;
        document.getElementById('ped-timer-box').textContent = tText;

        document.getElementById('estado-sub').textContent = (data.estado || '').replace(/_/g, ' ');
        document.getElementById('ped-sub').textContent = luzesPed.verde ? 'ATRAVESSAR' : (isPiscando ? 'ATENÇÃO!' : 'AGUARDE');
      } catch (e) {
        document.getElementById('conn-status').textContent = '● CONECTANDO...';
        document.getElementById('conn-status').style.color = '#eab308';
      }
    }

    setInterval(updateStatus, 150);
    updateStatus();
    scheduleHideDrawer();
  </script>
</body>
</html>
"""


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
        tempo_meta = self.fsm.tempo_verde_calculado if self.fsm.estado_atual == EstadoSemaforo.VERDE_CARRO else (self.fsm.tempo_amarelo if self.fsm.estado_atual == EstadoSemaforo.AMARELO_CARRO else self.fsm.tempo_verde_pedestre)
        tempo_restante = max(0.0, round(tempo_meta - tempo_estado, 1))

        return {
            "status": "ok",
            "estado": self.fsm.estado_atual.value,
            "modo_failsafe": self.fsm.modo_failsafe,
            "tempo_no_estado_s": tempo_estado,
            "tempo_verde_meta_s": self.fsm.tempo_verde_calculado,
            "tempo_restante": tempo_restante,
            "totais": {
                "veiculos": self.fsm.veiculos_detectados,
                "pedestres": self.fsm.pedestres_detectados,
                "animais": self.fsm.animais_detectados,
            },
            "luzes": {
                "carro": {
                    "vermelho": carro_r,
                    "amarelo": carro_y,
                    "verde": carro_g,
                },
                "pedestre": {
                    "vermelho": ped_r,
                    "verde": ped_g,
                    "piscando": self.fsm.estado_atual == EstadoSemaforo.VERMELHO_PISCANDO_PEDESTRE,
                },
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

        def _enviar_html(self, status_code: int, html_str: str):
            conteudo = html_str.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(conteudo)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(conteudo)

        def do_GET(self):
            if self.path in ("/", "/display", "/index.html"):
                self._enviar_html(200, HTML_DISPLAY)
            elif self.path in ("/api/status", "/health"):
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
    parser.add_argument("--host", default="0.0.0.0", help="Endereço IP de bind (padrão: 0.0.0.0)")
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
