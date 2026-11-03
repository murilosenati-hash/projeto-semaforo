import os
import sys
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.traffic_simulator import (
    SimuladorTrafegoVirtual,
    VeiculoVirtual,
    CarroVirtual,
    gerar_sprite_veiculo,
    gerar_sprite_carro
)
from core.fsm import EstadoSemaforo


def test_spawn_e_despawn_duas_faixas():
    """
    Testa ciclo de vida de veículos distribuídos nas duas faixas (faixa 0 e 1)
    e múltiplos tipos.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55, max_veiculos=4)
    
    s0 = sim.spawn_veiculo(faixa=0, tipo="carro")
    s1 = sim.spawn_veiculo(faixa=1, tipo="caminhao")
    
    assert s0 is True
    assert s1 is True
    assert len(sim.veiculos) == 2
    assert sim.veiculos[0].faixa == 0
    assert sim.veiculos[1].faixa == 1
    assert sim.veiculos[0].tipo == "carro"
    assert sim.veiculos[1].tipo == "caminhao"
    
    # Avança no sinal verde até despawn
    for _ in range(80):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)
        
    assert all(v.posicao < 1.0 for v in sim.veiculos)


def test_parada_no_sinal_fechado_e_arranque():
    """
    Testa parada e arranque respeitando a Stop Line.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55)
    sim.spawn_veiculo(faixa=0, tipo="moto")
    v = sim.veiculos[0]
    v.posicao = 0.45

    for _ in range(60):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERMELHO_LIMPEZA)

    assert v.posicao <= 0.5501
    assert v.parado is True
    assert v.velocidade == 0.0

    # Abre o sinal
    sim.atualizar(dt=0.2, fsm_state=EstadoSemaforo.VERDE_CARRO)
    assert v.velocidade > 0.0
    assert v.parado is False


def test_anti_colisao_independente_por_faixa():
    """
    Verifica se a lógica anti-colisão opera estritamente na mesma faixa,
    sem travar veículos que trafegam na faixa vizinha livre.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55)
    
    # Faixa 0: líder parado
    v_lider_f0 = VeiculoVirtual(id=1, posicao=0.50, velocidade=0.0, velocidade_max=0.20,
                                comprimento=0.09, cor=(200, 50, 50), parado=True, faixa=0, tipo="carro")
    sim.veiculos.append(v_lider_f0)

    # Faixa 0: seguidor atrás do líder
    v_seg_f0 = VeiculoVirtual(id=2, posicao=0.40, velocidade=0.15, velocidade_max=0.20,
                              comprimento=0.09, cor=(200, 50, 50), parado=False, faixa=0, tipo="carro")
    sim.veiculos.append(v_seg_f0)

    # Faixa 1: veículo livre na outra faixa
    v_livre_f1 = VeiculoVirtual(id=3, posicao=0.38, velocidade=0.18, velocidade_max=0.22,
                                comprimento=0.09, cor=(50, 200, 50), parado=False, faixa=1, tipo="carro")
    sim.veiculos.append(v_livre_f1)

    # Atualiza sob sinal verde
    for _ in range(15):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERDE_CARRO)

    # Faixa 0 deve manter distância segura
    assert (v_lider_f0.posicao - v_seg_f0.posicao) >= 0.06
    # Faixa 1 deve continuar se movendo livremente
    assert v_livre_f1.velocidade > 0.10
    assert v_livre_f1.posicao > 0.45


def test_geracao_sprites_multi_veiculos():
    """
    Verifica geração procedural de sprites para carro, caminhão e moto.
    """
    sprite_carro = gerar_sprite_veiculo(120, 240, (200, 50, 40), tipo="carro")
    assert sprite_carro.shape == (240, 120, 4)
    assert np.any(sprite_carro[:, :, 3] > 0)

    sprite_caminhao = gerar_sprite_veiculo(130, 310, (50, 150, 200), tipo="caminhao")
    assert sprite_caminhao.shape == (310, 130, 4)
    assert np.any(sprite_caminhao[:, :, 3] > 0)

    sprite_moto = gerar_sprite_veiculo(50, 120, (20, 20, 20), tipo="moto")
    assert sprite_moto.shape == (120, 50, 4)
    assert np.any(sprite_moto[:, :, 3] > 0)


def test_renderizacao_aspect_ratio_e_duas_faixas():
    """
    Testa renderização preservando proporção rígida em 2 faixas e N vértices.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55)
    sim.spawn_veiculo(faixa=0, tipo="carro")
    sim.spawn_veiculo(faixa=1, tipo="caminhao")
    
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    roi_pixels = np.array([[200, 100], [1080, 100], [1200, 700], [80, 700]], dtype=np.int32)

    res = sim.renderizar(frame.copy(), roi_pixels, estado_fsm=EstadoSemaforo.VERDE_CARRO)
    assert res.shape == (720, 1280, 3)
    assert not np.all(res == 40)


def test_stop_line_precisao_parada_sem_invasao():
    """
    Testa precisão milimétrica na parada da stop line com margem de segurança de 30cm,
    garantindo que veículos de diferentes portes (sedan 4.5m e caminhão 10m)
    nunca invadam a linha de retenção com o parachoque dianteiro.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55, comprimento_pista_m=50.0)
    v_carro = VeiculoVirtual(id=1, posicao=0.40, velocidade=0.15, velocidade_max=0.20,
                             comprimento=4.5, cor=(200, 50, 50), parado=False, faixa=0, tipo="carro")
    v_caminhao = VeiculoVirtual(id=2, posicao=0.35, velocidade=0.12, velocidade_max=0.16,
                                comprimento=10.0, cor=(50, 100, 200), parado=False, faixa=1, tipo="caminhao")
    sim.veiculos.extend([v_carro, v_caminhao])

    for _ in range(80):
        sim.atualizar(dt=0.1, fsm_state=EstadoSemaforo.VERMELHO_LIMPEZA)

    assert v_carro.parado is True
    assert v_carro.velocidade == 0.0
    parachoque_carro = v_carro.posicao + (v_carro.comprimento * 0.5) / sim.comprimento_pista_m
    assert parachoque_carro <= sim.stop_line - (0.3 / sim.comprimento_pista_m) + 0.001

    assert v_caminhao.parado is True
    assert v_caminhao.velocidade == 0.0
    parachoque_caminhao = v_caminhao.posicao + (v_caminhao.comprimento * 0.5) / sim.comprimento_pista_m
    assert parachoque_caminhao <= sim.stop_line - (0.3 / sim.comprimento_pista_m) + 0.001
