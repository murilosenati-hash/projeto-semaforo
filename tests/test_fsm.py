import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fsm import ControladorSemaforoFSM, EstadoSemaforo

def test_transicao_basica():
    fsm = ControladorSemaforoFSM(
        min_verde_carro=0.2,
        max_verde_carro=1.0,
        tempo_amarelo=0.2,
        tempo_limpeza=0.1,
        tempo_verde_pedestre=0.2,
        tempo_pisca_pedestre=0.1
    )
    assert fsm.estado_atual == EstadoSemaforo.VERDE_CARRO
    fsm.atualizar_sensores(veiculos=2, pedestres=1, animais=0)

    # Aguarda o verde mínimo
    time.sleep(0.25)
    fsm.tick()
    assert fsm.estado_atual == EstadoSemaforo.AMARELO_CARRO, f"Esperado AMARELO_CARRO, obteve {fsm.estado_atual}"

    # Aguarda o amarelo
    time.sleep(0.25)
    fsm.tick()
    assert fsm.estado_atual == EstadoSemaforo.VERMELHO_LIMPEZA

    # Aguarda a limpeza
    time.sleep(0.15)
    fsm.tick()
    assert fsm.estado_atual == EstadoSemaforo.VERDE_PEDESTRE
    print("[TESTE PASSOU] Transição básica de estados funcionando 100%!")

def test_failsafe():
    fsm = ControladorSemaforoFSM()
    # Simula timeout de heartbeat
    fsm.ultimo_heartbeat = time.time() - 10.0
    fsm.tick()
    assert fsm.estado_atual == EstadoSemaforo.FAIL_SAFE
    print("[TESTE PASSOU] Modo Fail-Safe ativado com sucesso em caso de perda de rede!")

if __name__ == '__main__':
    test_transicao_basica()
    test_failsafe()
