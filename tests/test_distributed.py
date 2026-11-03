import os
import sys
import threading
import time
import urllib.request
import json
from http.server import HTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fsm import ControladorSemaforoFSM, EstadoSemaforo
from distributed.sensor_node import ClienteTelemetria
from distributed.server import ControladorDistribuido, criar_handler_http, iniciar_servidor_socket


def test_comunicacao_distribuida_tcp_e_http():
    fsm = ControladorSemaforoFSM(
        min_verde_carro=0.2,
        max_verde_carro=2.0,
        tempo_amarelo=0.15,
        tempo_limpeza=0.1,
    )
    controlador = ControladorDistribuido(fsm=fsm, node_timeout=2.0)
    stop_event = threading.Event()

    # Inicia servidor TCP em porta de teste
    tcp_port = 15050
    http_port = 18080
    t_tcp = threading.Thread(
        target=iniciar_servidor_socket,
        args=(controlador, "127.0.0.1", tcp_port, stop_event),
        daemon=True,
    )
    t_tcp.start()

    # Inicia servidor HTTP em porta de teste
    httpd = HTTPServer(("127.0.0.1", http_port), criar_handler_http(controlador))
    t_http = threading.Thread(target=httpd.serve_forever, daemon=True)
    t_http.start()
    time.sleep(0.15)

    try:
        # 1. Nó Norte via TCP Socket
        sensor_norte = ClienteTelemetria("cam_norte", protocolo="tcp", host="127.0.0.1", port=tcp_port)
        resp_tcp = sensor_norte.enviar_leitura(veiculos=6, pedestres=0, animais=0)
        assert resp_tcp is not None and resp_tcp["status"] == "ok"
        assert resp_tcp["totais"]["veiculos"] == 6

        # 2. Nó Sul via HTTP REST API (agregação multi-sensor)
        sensor_sul = ClienteTelemetria("cam_sul", protocolo="http", host="127.0.0.1", port=http_port)
        resp_http = sensor_sul.enviar_leitura(veiculos=4, pedestres=2, animais=1)
        assert resp_http is not None and resp_http["status"] == "ok"
        assert resp_http["totais"]["veiculos"] == 10
        assert resp_http["totais"]["pedestres"] == 2
        assert resp_http["totais"]["animais"] == 1
        assert len(resp_http["nos_sensores"]) == 2

        # 3. Verifica transição para fase de atenção ou limpeza após min_verde_carro com pedestre/animal
        time.sleep(0.22)
        resp_transicao = sensor_norte.enviar_leitura(veiculos=5, pedestres=2, animais=1)
        assert resp_transicao is not None
        assert resp_transicao["estado"] in (EstadoSemaforo.AMARELO_CARRO.value, EstadoSemaforo.VERMELHO_LIMPEZA.value)

        sensor_norte.fechar()
        sensor_sul.fechar()
    finally:
        stop_event.set()
        httpd.shutdown()


def test_servidor_rotas_web_display_html():
    fsm = ControladorSemaforoFSM()
    controlador = ControladorDistribuido(fsm=fsm, node_timeout=3.0)
    http_port = 18089
    httpd = HTTPServer(("127.0.0.1", http_port), criar_handler_http(controlador))
    t_http = threading.Thread(target=httpd.serve_forever, daemon=True)
    t_http.start()
    time.sleep(0.15)

    try:
        # Rota raiz '/'
        url_root = f"http://127.0.0.1:{http_port}/"
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
        url_display = f"http://127.0.0.1:{http_port}/display"
        with urllib.request.urlopen(url_display, timeout=2.0) as resp:
            assert resp.status == 200
            conteudo_disp = resp.read().decode("utf-8")
            assert "main-container" in conteudo_disp

        # Rota '/api/status'
        url_status = f"http://127.0.0.1:{http_port}/api/status"
        with urllib.request.urlopen(url_status, timeout=2.0) as resp:
            assert resp.status == 200
            dados = json.loads(resp.read().decode("utf-8"))
            assert "estado" in dados
            assert "luzes" in dados
            assert "carro" in dados["luzes"]
            assert "pedestre" in dados["luzes"]
            assert dados["luzes"]["carro"]["verde"] is True
            assert dados["luzes"]["pedestre"]["vermelho"] is True
    finally:
        httpd.shutdown()


def test_main_integrated_servidor_web_embutido():
    """Garante que main_integrated inicia o servidor web na mesma FSM."""
    from main_integrated import iniciar_servidor_web_embutido
    fsm = ControladorSemaforoFSM()
    porta_teste = 18099
    controlador = iniciar_servidor_web_embutido(fsm, host="127.0.0.1", port=porta_teste)
    assert controlador is not None
    time.sleep(0.15)

    url = f"http://127.0.0.1:{porta_teste}/api/status"
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        assert resp.status == 200
        dados = json.loads(resp.read().decode("utf-8"))
        assert dados["estado"] == fsm.estado_atual.value
        assert dados["luzes"]["carro"]["verde"] is True
