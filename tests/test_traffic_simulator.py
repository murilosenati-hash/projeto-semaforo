import os
import sys
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.traffic_simulator import SimuladorTrafegoVirtual, VeiculoVirtual, gerar_sprite_carro
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

    for _ in range(15):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)

    sim.spawn_imediato()
    assert len(sim.veiculos) == 2

    # Fecha o sinal
    for _ in range(80):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.AMARELO_CARRO)

    carro_lider = max(sim.veiculos, key=lambda c: c.posicao)
    carro_traseiro = min(sim.veiculos, key=lambda c: c.posicao)

    distancia = carro_lider.posicao - carro_traseiro.posicao
    assert distancia >= carro_lider.comprimento, "Veículo traseiro colidiu ou ultrapassou o dianteiro"
    assert carro_lider.parado is True
    assert carro_traseiro.parado is True


def test_geracao_sprite_e_asset():
    """
    Testa a geração procedural de sprites RGBA (4 canais) com rodas,
    vidros escuros, faróis e luzes de freio dinâmicas.
    """
    sp = gerar_sprite_carro(120, 240, cor_bgr=(200, 50, 40), parado=False)
    assert isinstance(sp, np.ndarray)
    assert sp.shape == (240, 120, 4)
    assert sp.dtype == np.uint8
    
    # Canal alfa contém partes opacas
    alpha = sp[:, :, 3]
    assert np.max(alpha) == 255
    assert np.min(alpha) == 0

    # Sprite com freio ativado
    sp_parado = gerar_sprite_carro(120, 240, cor_bgr=(200, 50, 40), parado=True)
    assert not np.array_equal(sp, sp_parado)

    # Verifica persistência do asset assets/car_top.png
    sim = SimuladorTrafegoVirtual()
    caminho_asset = os.path.join(os.path.dirname(__file__), '..', 'assets', 'car_top.png')
    assert os.path.exists(caminho_asset)


def test_renderizacao_buffer_preservado_e_poligonos_n_vertices():
    """
    Testa renderização com sprites, sombras projetadas e compatibilidade
    com polígonos de 4 vértices, 3 vértices e N vértices.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=3)
    sim.spawn_imediato()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_copy = frame.copy()
    
    # 1. Teste com 4 vértices
    roi_4 = np.array([[256, 216], [1024, 216], [1024, 648], [256, 648]], dtype=np.int32)
    res4 = sim.renderizar(frame.copy(), roi_4, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert res4.shape == (720, 1280, 3)
    assert res4.dtype == np.uint8
    assert not np.array_equal(res4, frame_copy)

    # 2. Teste com 5 vértices (N = 5)
    roi_5 = np.array([[300, 200], [900, 200], [1050, 600], [640, 680], [200, 600]], dtype=np.int32)
    res5 = sim.renderizar(frame.copy(), roi_5, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert res5.shape == (720, 1280, 3)
    assert not np.array_equal(res5, frame_copy)

    # 3. Teste com 3 vértices (Triângulo)
    roi_3 = np.array([[640, 150], [1000, 650], [280, 650]], dtype=np.int32)
    res3 = sim.renderizar(frame.copy(), roi_3, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert res3.shape == (720, 1280, 3)
    assert not np.array_equal(res3, frame_copy)

    # Robustez com None
    res_none = sim.renderizar(frame.copy(), None, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert np.array_equal(res_none, frame)
