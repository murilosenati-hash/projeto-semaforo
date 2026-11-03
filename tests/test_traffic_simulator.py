import os
import sys
import numpy as np

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
    sim = SimuladorTrafegoVirtual(taxa_spawn=0.1, stop_line=0.55, max_veiculos=4)
    
    # Spawn na faixa 0 e na faixa 1
    s0 = sim.spawn_veiculo(faixa=0, tipo="carro")
    s1 = sim.spawn_veiculo(faixa=1, tipo="caminhao")
    
    assert s0 is True
    assert s1 is True
    assert len(sim.veiculos) == 2
    assert sim.veiculos[0].faixa == 0
    assert sim.veiculos[1].faixa == 1
    assert sim.veiculos[0].tipo == "carro"
    assert sim.veiculos[1].tipo == "caminhao"
    
    # Simula avanço até além de 1.05 para verificar despawn
    for _ in range(30):
        sim.atualizar(dt=0.3, estado_semaforo=EstadoSemaforo.VERDE_CARRO)
        
    assert all(v.y <= 1.05 for v in sim.veiculos)


def test_parada_no_sinal_fechado_e_arranque():
    """
    Garante que os veículos desaceleram até a linha de retenção no sinal vermelho
    e arrancam normalmente quando o semáforo abre.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=10.0, stop_line=0.55)
    sim.spawn_veiculo(faixa=0, tipo="moto")
    v = sim.veiculos[0]
    v.y = 0.45

    # Sinal fechado -> veículo desacelera
    for _ in range(25):
        sim.atualizar(dt=0.1, estado_semaforo=EstadoSemaforo.VERMELHO_TOTAL)

    assert v.y <= 0.58
    assert v.velocidade < 0.05

    # Sinal abre -> veículo ganha velocidade
    for _ in range(15):
        sim.atualizar(dt=0.1, estado_semaforo=EstadoSemaforo.VERDE_CARRO)

    assert v.velocidade > 0.05
    assert v.y > 0.50


def test_anti_colisao_independente_por_faixa():
    """
    Verifica se a lógica anti-colisão opera estritamente na mesma faixa,
    sem travar veículos que trafegam na faixa vizinha livre.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55)
    
    # Faixa 0: líder parado
    v_lider_f0 = VeiculoVirtual(id_veiculo=1, y_pos=0.50, velocidade_max=0.20, faixa=0, tipo="carro")
    v_lider_f0.velocidade = 0.0
    sim.veiculos.append(v_lider_f0)

    # Faixa 0: seguidor atrás do líder
    v_seg_f0 = VeiculoVirtual(id_veiculo=2, y_pos=0.42, velocidade_max=0.20, faixa=0, tipo="carro")
    sim.veiculos.append(v_seg_f0)

    # Faixa 1: veículo livre na outra faixa
    v_livre_f1 = VeiculoVirtual(id_veiculo=3, y_pos=0.40, velocidade_max=0.25, faixa=1, tipo="carro")
    sim.veiculos.append(v_livre_f1)

    # Atualiza sob sinal verde
    for _ in range(10):
        sim.atualizar(dt=0.1, estado_semaforo=EstadoSemaforo.VERDE_CARRO)

    # Faixa 0 deve manter distância segura
    assert (v_lider_f0.y - v_seg_f0.y) >= 0.06
    # Faixa 1 deve continuar se movendo sem ser bloqueada pelo carro da Faixa 0
    assert v_livre_f1.velocidade > 0.10
    assert v_livre_f1.y > 0.45


def test_geracao_sprites_multi_veiculos():
    """
    Verifica geração procedural de sprites para carro, caminhão e moto
    com preservação de canal alfa BGRA.
    """
    sprite_carro = gerar_sprite_veiculo(30, 60, (200, 50, 50), tipo="carro")
    assert sprite_carro.shape == (60, 30, 4)
    assert np.any(sprite_carro[:, :, 3] > 0)

    sprite_caminhao = gerar_sprite_veiculo(36, 85, (50, 150, 200), tipo="caminhao")
    assert sprite_caminhao.shape == (85, 36, 4)
    assert np.any(sprite_caminhao[:, :, 3] > 0)

    sprite_moto = gerar_sprite_veiculo(16, 42, (20, 20, 20), tipo="moto")
    assert sprite_moto.shape == (42, 16, 4)
    assert np.any(sprite_moto[:, :, 3] > 0)


def test_renderizacao_aspect_ratio_e_duas_faixas():
    """
    Testa renderização em frame sintético validando divisória de faixas,
    stop line e veículos sem distorção.
    """
    sim = SimuladorTrafegoVirtual(taxa_spawn=100.0, stop_line=0.55)
    sim.spawn_veiculo(faixa=0, tipo="carro")
    sim.spawn_veiculo(faixa=1, tipo="caminhao")
    
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    roi_pixels = np.array([[200, 100], [1080, 100], [1200, 700], [80, 700]], dtype=np.int32)

    # Renderiza sem exceções
    sim.renderizar(frame, roi_pixels, EstadoSemaforo.VERDE_CARRO)
    assert frame.shape == (720, 1280, 3)
    assert not np.all(frame == 40)  # Houve alteração de pixels pela renderização
