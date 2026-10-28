import os
import sys
import threading
import time
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

        # 3. Verifica transição para AMARELO_CARRO após min_verde_carro com pedestre/animal
        time.sleep(0.25)
        resp_transicao = sensor_norte.enviar_leitura(veiculos=5, pedestres=2, animais=1)
        assert resp_transicao["estado"] == EstadoSemaforo.AMARELO_CARRO.value

        sensor_norte.fechar()
        sensor_sul.fechar()
        print("[TESTE PASSOU] Comunicação distribuída (TCP Socket + HTTP REST) e agregação multi-nó 100%!")
    finally:
        stop_event.set()
        httpd.shutdown()


if __name__ == "__main__":
    test_comunicacao_distribuida_tcp_e_http()
