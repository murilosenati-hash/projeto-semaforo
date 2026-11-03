import os
import sys
import tempfile
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_integrated import (
    carregar_rois,
    carregar_roi,
    salvar_rois,
    salvar_roi,
    validar_poligono,
    validar_poligono_4_pontos,
    normalizar_pontos_roi,
    ROI_PADRAO_VEICULOS,
    ROI_PADRAO_PEDESTRES,
    ROI_PADRAO
)
from core.detector import DetectorTrafego


def test_roi_carregar_padrao_fallback():
    """
    Garante que se o arquivo não existir ou estiver corrompido,
    carregar_roi() faz fallback gracioso para ROI_PADRAO_VEICULOS.
    """
    caminho_inexistente = os.path.join(tempfile.gettempdir(), "roi_inexistente_fallback_test.json")
    if os.path.exists(caminho_inexistente):
        os.remove(caminho_inexistente)

    roi = carregar_roi(caminho=caminho_inexistente)
    assert len(roi) == 4
    assert roi == [list(p) for p in ROI_PADRAO_VEICULOS]


def test_dual_roi_salvar_e_carregar_persistencia():
    """
    Testa ciclo completo de persistência dual-ROI (veículos e pedestres)
    com formato estruturado em dicionário.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        caminho_tmp = tf.name

    try:
        rois_dict = {
            "veiculos": [
                [0.10, 0.20],
                [0.90, 0.20],
                [0.90, 0.90],
                [0.10, 0.90]
            ],
            "pedestres": [
                [0.05, 0.60],
                [0.25, 0.60],
                [0.25, 0.95],
                [0.05, 0.95]
            ]
        }
        sucesso = salvar_rois(rois_dict, caminho=caminho_tmp)
        assert sucesso is True
        assert os.path.exists(caminho_tmp)

        rois_lidas = carregar_rois(caminho=caminho_tmp)
        assert rois_lidas["veiculos"] == rois_dict["veiculos"]
        assert rois_lidas["pedestres"] == rois_dict["pedestres"]

        # Teste de atualização individual via salvar_roi(tipo='pedestres')
        nova_ped = [[0.02, 0.40], [0.18, 0.40], [0.18, 0.85], [0.02, 0.85]]
        salvar_roi(nova_ped, caminho=caminho_tmp, tipo="pedestres")
        rois_atualizadas = carregar_rois(caminho=caminho_tmp)
        assert rois_atualizadas["pedestres"] == nova_ped
        assert rois_atualizadas["veiculos"] == rois_dict["veiculos"]
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)


def test_dual_roi_retrocompatibilidade_arquivo_legado():
    """
    Garante que se o arquivo JSON contiver apenas uma lista de pontos (formato legado),
    carregar_rois() carrega veículos da lista e mantém pedestres com fallback padrão.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tf:
        caminho_tmp = tf.name
        lista_legada = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]]
        json.dump(lista_legada, tf)

    try:
        rois = carregar_rois(caminho=caminho_tmp)
        assert rois["veiculos"] == lista_legada
        assert rois["pedestres"] == [list(p) for p in ROI_PADRAO_PEDESTRES]
        assert carregar_roi(caminho=caminho_tmp) == lista_legada
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
    triangulo = [[200, 100], [600, 100], [400, 500]]
    valido, msg = validar_poligono(triangulo)
    assert valido is True

    pentagono = [[300, 100], [500, 150], [550, 400], [250, 450], [150, 300]]
    valido, msg = validar_poligono(pentagono)
    assert valido is True

    hexagono = [[300, 100], [500, 100], [600, 300], [500, 500], [300, 500], [200, 300]]
    valido, msg = validar_poligono(hexagono)
    assert valido is True

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

    ponto_dentro = (640, 400)
    ponto_fora = (100, 100)
    assert DetectorTrafego.ponto_dentro_roi(ponto_dentro, roi_px) is True
    assert DetectorTrafego.ponto_dentro_roi(ponto_fora, roi_px) is False

    pent_norm = [[0.3, 0.1], [0.7, 0.1], [0.9, 0.5], [0.7, 0.9], [0.3, 0.9]]
    pent_px = DetectorTrafego.roi_para_pixels(pent_norm, shape_frame)
    assert len(pent_px) == 5
    assert DetectorTrafego.ponto_dentro_roi((640, 360), pent_px) is True
