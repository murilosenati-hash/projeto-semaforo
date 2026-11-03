import time
import urllib.request
import json
from core.fsm import MaquinaEstadosSemaforo, EstadoSemaforo
from distributed.server import ServidorDistribuido


def test_comunicacao_distribuida_tcp_e_http():
    fsm = MaquinaEstadosSemaforo()
    servidor = ServidorDistribuido(fsm=fsm, host="127.0.0.1", porta=8888)
    servidor.iniciar()
    time.sleep(0.1)

    try:
        url = "http://127.0.0.1:8888/api/status"
        with urllib.request.urlopen(url, timeout=2.0) as resposta:
            assert resposta.status == 200
            dados = json.loads(resposta.read().decode("utf-8"))

            assert "estado" in dados
            assert "luzes" in dados
            assert "carro" in dados["luzes"]
            assert "pedestre" in dados["luzes"]
            assert dados["estado"] == EstadoSemaforo.VERDE_CARROS.value
            assert dados["luzes"]["carro"]["verde"] is True
            assert dados["luzes"]["pedestre"]["vermelho"] is True
    finally:
        servidor.parar()


def test_servidor_rotas_web_display_html():
    fsm = MaquinaEstadosSemaforo()
    servidor = ServidorDistribuido(fsm=fsm, host="127.0.0.1", porta=8889)
    servidor.iniciar()
    time.sleep(0.1)

    try:
        # Rota raiz '/'
        url_root = "http://127.0.0.1:8889/"
        with urllib.request.urlopen(url_root, timeout=2.0) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-type", "")
            conteudo = resp.read().decode("utf-8")
            assert "Semáforo Inteligente" in conteudo
            assert "control-drawer" in conteudo
            assert "btn-mode-car" in conteudo
            assert "btn-mode-ped" in conteudo
            assert "requestFullscreen" in conteudo

        # Rota '/display'
        url_display = "http://127.0.0.1:8889/display"
        with urllib.request.urlopen(url_display, timeout=2.0) as resp:
            assert resp.status == 200
            conteudo_disp = resp.read().decode("utf-8")
            assert "main-container" in conteudo_disp
    finally:
        servidor.parar()
