import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.traffic_simulator import SimuladorTrafegoVirtual, VeiculoVirtual
from core.fsm import EstadoSemaforo


def test_spawn_e_despawn():
    """
    Testa ciclo de vida de veículos virtuais:
    - Spawn forçado via spawn_imediato()
    - Remoção automática ao cruzar posicao >= 1.0
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=3)
    assert len(sim.veiculos) == 0

    sucesso = sim.spawn_imediato()
    assert sucesso is True
    assert len(sim.veiculos) == 1
    carro = sim.veiculos[0]
    assert carro.posicao == 0.0
    assert carro.id == 1

    # Avança no sinal verde até cruzar 1.0
    for _ in range(100):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)
        if len(sim.veiculos) == 0:
            break

    assert len(sim.veiculos) == 0, "Veículo não foi despawnado após cruzar posicao >= 1.0"


def test_parada_no_sinal_fechado():
    """
    Testa se o veículo respeita a Stop Line (0.55) quando o sinal está fechado
    (AMARELO ou VERMELHO/PEDESTRE) e não invade a faixa.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=2)
    sim.spawn_imediato()

    # Atualiza por tempo prolongado com sinal fechado
    for _ in range(80):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERMELHO_LIMPEZA)

    assert len(sim.veiculos) == 1
    carro = sim.veiculos[0]
    assert carro.posicao <= 0.5501, f"Carro ultrapassou a Stop Line: {carro.posicao}"
    assert carro.parado is True
    assert carro.velocidade == 0.0


def test_arranque_no_sinal_verde():
    """
    Testa se veículos retidos na Stop Line arrancam e ganham velocidade
    assim que a FSM transita para VERDE_CARRO.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=2)
    sim.spawn_imediato()

    # Para na Stop Line no sinal vermelho
    for _ in range(60):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_PEDESTRE)

    carro = sim.veiculos[0]
    assert carro.parado is True
    pos_parada = carro.posicao

    # Abre o sinal para veículos
    sim.atualizar(dt=0.2, fsm_state=EstadoSemaforo.VERDE_CARRO)
    assert carro.velocidade > 0.0
    assert carro.parado is False

    # Avança mais alguns passos no verde
    for _ in range(20):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)

    assert carro.posicao > pos_parada


def test_anti_colisao_fila():
    """
    Testa se dois veículos em sequência formam fila sem colisão/acavalamento
    quando o primeiro está parado no sinal vermelho.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=3)
    sim.spawn_imediato()

    # Deixa o primeiro carro avançar um pouco
    for _ in range(15):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)

    # Spawna o segundo carro atrás
    sim.spawn_imediato()
    assert len(sim.veiculos) == 2

    # Fecha o sinal
    for _ in range(80):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.AMARELO_CARRO)

    carro_lider = max(sim.veiculos, key=lambda c: c.posicao)
    carro_traseiro = min(sim.veiculos, key=lambda c: c.posicao)

    # O carro de trás deve manter distância de segurança
    distancia = carro_lider.posicao - carro_traseiro.posicao
    assert distancia >= carro_lider.comprimento, "Veículo traseiro colidiu ou ultrapassou o dianteiro"
    assert carro_lider.parado is True
    assert carro_traseiro.parado is True


def test_renderizacao_buffer_preservado():
    """
    Testa se a função renderizar() não corrompe o buffer de imagem,
    mantém dimensões, tipo uint8 e executa sem lançar exceções.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=3)
    sim.spawn_imediato()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_copy = frame.copy()
    roi_pixels = np.array([
        [256, 216],
        [1024, 216],
        [1024, 648],
        [256, 648]
    ], dtype=np.int32)

    res = sim.renderizar(frame, roi_pixels, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    
    assert isinstance(res, np.ndarray)
    assert res.shape == (720, 1280, 3)
    assert res.dtype == np.uint8
    # Frame foi modificado (renderizou elementos AR)
    assert not np.array_equal(res, frame_copy)

    # Teste de robustez com ROI vazia ou None
    res_none = sim.renderizar(frame, None, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert res_none.shape == (720, 1280, 3)
