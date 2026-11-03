import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_integrated import (
    carregar_roi,
    salvar_roi,
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
    Testa regras geométricas dos 4 Goldfish:
    1. Polígono válido convexo (trapezoidal / perspectiva)
    2. Polígono degenerado com área menor que 1000px
    3. Polígono auto-intersectado (formato de 8 / cruzado / não-convexo)
    4. Quantidade incorreta de vértices
    """
    # 1. Trapézio convexo típico de pista (válido)
    trapezio_valido = [[200, 150], [600, 150], [750, 450], [50, 450]]
    valido, msg = validar_poligono_4_pontos(trapezio_valido)
    assert valido is True
    assert msg == "OK"

    # 2. Área degenerada (< 1000 px ou colinear)
    pontos_colineares = [[100, 100], [110, 100], [120, 100], [130, 100]]
    valido, msg = validar_poligono_4_pontos(pontos_colineares)
    assert valido is False
    assert "insuficiente" in msg.lower()

    # 3. Cruzamento em formato de 8 (ampulheta / auto-interseção com área > 1000px)
    # P1(100,100), P2(600,500), P3(600,100), P4(100,300)
    pontos_cruzados = [[100, 100], [600, 500], [600, 100], [100, 300]]
    valido, msg = validar_poligono_4_pontos(pontos_cruzados)
    assert valido is False
    assert "não-convexo" in msg.lower() or "auto-intersectado" in msg.lower()

    # 4. Quantidade incorreta de pontos
    valido, msg = validar_poligono_4_pontos([[10, 10], [20, 20]])
    assert valido is False


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
    Valida se a lista de 4 vértices normalizados é compatível
    com DetectorTrafego.roi_para_pixels e operações do OpenCV.
    """
    roi_normalizada = [
        [0.2, 0.3],
        [0.8, 0.3],
        [0.8, 0.9],
        [0.2, 0.9]
    ]
    shape = (720, 1280, 3)  # h=720, w=1280
    poly = DetectorTrafego.roi_para_pixels(roi_normalizada, shape)

    assert isinstance(poly, np.ndarray)
    assert poly.shape == (4, 2)
    assert poly.dtype == np.int32

    # p0: (0.2 * 1280, 0.3 * 720) = (256, 216)
    np.testing.assert_array_equal(poly[0], [256, 216])
    # p1: (0.8 * 1280, 0.3 * 720) = (1024, 216)
    np.testing.assert_array_equal(poly[1], [1024, 216])
    # p2: (0.8 * 1280, 0.9 * 720) = (1024, 648)
    np.testing.assert_array_equal(poly[2], [1024, 648])
    # p3: (0.2 * 1280, 0.9 * 720) = (256, 648)
    np.testing.assert_array_equal(poly[3], [256, 648])

    # Valida teste de ponto dentro da ROI
    assert DetectorTrafego.ponto_dentro_roi((640, 400), poly) is True
    assert DetectorTrafego.ponto_dentro_roi((100, 100), poly) is False
