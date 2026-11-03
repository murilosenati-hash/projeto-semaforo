import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_integrated import (
    carregar_roi,
    salvar_roi,
    validar_poligono,
    validar_poligono_4_pontos,
    normalizar_pontos_roi,
    ROI_PADRAO
)
from core.detector import DetectorTrafego


def test_roi_carregar_padrao_fallback():
    """
    Garante que se o arquivo não existir ou estiver corrompido,
    carregar_roi() faz fallback gracioso para ROI_PADRAO.
    """
    caminho_inexistente = os.path.join(tempfile.gettempdir(), "roi_inexistente_fallback_test.json")
    if os.path.exists(caminho_inexistente):
        os.remove(caminho_inexistente)

    roi = carregar_roi(caminho=caminho_inexistente)
    assert len(roi) == 4
    assert roi == [list(p) for p in ROI_PADRAO]


def test_roi_salvar_e_carregar_persistencia():
    """
    Testa ciclo completo de persistência (salvar_roi e carregar_roi)
    em arquivo JSON dedicado com coordenadas normalizadas.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        caminho_tmp = tf.name

    try:
        roi_customizada = [
            [0.1234, 0.2345],
            [0.8765, 0.2345],
            [0.8765, 0.9876],
            [0.1234, 0.9876]
        ]
        sucesso = salvar_roi(roi_customizada, caminho=caminho_tmp)
        assert sucesso is True
        assert os.path.exists(caminho_tmp)

        roi_lida = carregar_roi(caminho=caminho_tmp)
        assert roi_lida == roi_customizada
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)


def test_validar_poligono_4_pontos_geometria():
    """
    Testa regras geométricas com threshold atualizado de 500px:
    1. Polígono válido convexo (trapezoidal / perspectiva)
    2. Polígono degenerado com área menor que 500px
    3. Polígono auto-intersectado (formato de 8 / cruzado / não-convexo)
    4. Quantidade incorreta de vértices (< 3)
    """
    # 1. Trapézio convexo típico de pista (válido)
    trapezio_valido = [[200, 150], [600, 150], [750, 450], [50, 450]]
    valido, msg = validar_poligono_4_pontos(trapezio_valido)
    assert valido is True
    assert msg == "OK"

    # 2. Área degenerada (< 500 px, ex: quadrado convexo de 5x5 = 25px)
    pontos_pequenos = [[100, 100], [105, 100], [105, 105], [100, 105]]
    valido, msg = validar_poligono_4_pontos(pontos_pequenos)
    assert valido is False
    assert "insuficiente" in msg.lower()

    # Pontos colineares (rejeitados por não-convexidade ou área)
    pontos_colineares = [[100, 100], [110, 100], [120, 100], [130, 100]]
    valido, msg = validar_poligono_4_pontos(pontos_colineares)
    assert valido is False
    assert "insuficiente" in msg.lower() or "não-convexo" in msg.lower() or "auto-intersectado" in msg.lower()

    # 3. Cruzamento em formato de 8 (ampulheta / auto-interseção)
    pontos_cruzados = [[100, 100], [400, 400], [400, 100], [100, 400]]
    valido, msg = validar_poligono_4_pontos(pontos_cruzados)
    assert valido is False
    assert "não-convexo" in msg.lower() or "auto-intersectado" in msg.lower()

    # 4. Quantidade incorreta de pontos (< 3)
    valido, msg = validar_poligono_4_pontos([[10, 10], [20, 20]])
    assert valido is False


def test_validar_poligono_n_vertices_livre():
    """
    Testa o suporte a polígonos livres com N vértices (N >= 3, ex: N=3, N=5, N=6)
    com threshold de área >= 500 px.
    """
    # Triângulo válido (N = 3, área > 500)
    triangulo = [[200, 100], [600, 100], [400, 500]]
    valido, msg = validar_poligono(triangulo)
    assert valido is True

    # Pentágono convexo (N = 5, área > 500)
    pentagono = [[300, 100], [500, 150], [550, 400], [250, 450], [150, 300]]
    valido, msg = validar_poligono(pentagono)
    assert valido is True

    # Hexágono convexo (N = 6)
    hexagono = [[300, 100], [500, 100], [600, 300], [500, 500], [300, 500], [200, 300]]
    valido, msg = validar_poligono(hexagono)
    assert valido is True

    # Rejeição de triângulo com área minúscula (< 500 px)
    tri_pequeno = [[100, 100], [120, 100], [110, 110]]
    valido, msg = validar_poligono(tri_pequeno)
    assert valido is False
    assert "insuficiente" in msg.lower()


def test_normalizar_pontos_roi_clamping():
    """
    Testa conversão com clamping em [0.0, 1.0] e arredondamento a 4 casas decimais.
    """
    pts = [
        [-50, -20],        # Fora dos limites (negativo) -> clamp para 0.0
        [1350, 50],        # Fora dos limites (w > 1280) -> clamp para 1.0
        [640, 360],        # Ponto central exato -> 0.5, 0.5
        [256, 720]         # 256/1280 = 0.2, 720/720 = 1.0
    ]
    fw, fh = 1280, 720
    norm = normalizar_pontos_roi(pts, fw, fh)

    assert norm[0] == [0.0, 0.0]
    assert norm[1] == [1.0, 0.0694]
    assert norm[2] == [0.5, 0.5]
    assert norm[3] == [0.2, 1.0]


def test_roi_compatibilidade_detector_pixels():
    """
    Valida se a lista de vértices normalizados (4 ou N vértices) é compatível
    com DetectorTrafego.roi_para_pixels e operações do OpenCV.
    """
    roi_normalizada = [
        [0.2, 0.3],
        [0.8, 0.3],
        [0.8, 0.9],
        [0.2, 0.9]
    ]
    shape_frame = (720, 1280, 3)
    roi_px = DetectorTrafego.roi_para_pixels(roi_normalizada, shape_frame)

    assert isinstance(roi_px, np.ndarray)
    assert roi_px.shape == (4, 2)
    assert roi_px.dtype == np.int32
    assert roi_px[0].tolist() == [256, 216]
    assert roi_px[1].tolist() == [1024, 216]
    assert roi_px[2].tolist() == [1024, 648]
    assert roi_px[3].tolist() == [256, 648]

    # Teste de ponto dentro da ROI
    ponto_dentro = (640, 400)
    ponto_fora = (100, 100)
    assert DetectorTrafego.ponto_dentro_roi(ponto_dentro, roi_px) is True
    assert DetectorTrafego.ponto_dentro_roi(ponto_fora, roi_px) is False

    # Teste com N vértices (Pentágono)
    pent_norm = [[0.3, 0.1], [0.7, 0.1], [0.9, 0.5], [0.7, 0.9], [0.3, 0.9]]
    pent_px = DetectorTrafego.roi_para_pixels(pent_norm, shape_frame)
    assert len(pent_px) == 5
    assert DetectorTrafego.ponto_dentro_roi((640, 360), pent_px) is True
