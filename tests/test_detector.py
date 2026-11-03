import os
import sys
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.detector import (
    DetectorTrafego,
    DebouncePedestre,
    CLASSE_PESSOA,
    CLASSES_VEICULOS,
    CLASSES_ANIMAIS,
    CLASSES_ALVO
)


def test_debounce_pedestre_permanencia():
    """
    Testa lógica estrita de debounce de permanência (1.5 segundos):
    - Menos de 1.5s não confirma
    - Atingindo 1.5s confirma
    - Se o pedestre sair da zona, zera imediatamente o acumulador
    """
    debounce = DebouncePedestre(tempo_confirmacao=1.5)

    # 1. Zero inicial
    conf, t, prog = debounce.atualizar(pedestre_presente=False, dt=0.1)
    assert conf is False
    assert t == 0.0
    assert prog == 0.0

    # 2. Pedestre entra e fica por 0.5s
    conf, t, prog = debounce.atualizar(pedestre_presente=True, dt=0.5)
    assert conf is False
    assert pytest_approx(t, 0.5)
    assert pytest_approx(prog, 0.5 / 1.5)

    # 3. Pedestre continua por mais 0.6s (total 1.1s < 1.5s)
    conf, t, prog = debounce.atualizar(pedestre_presente=True, dt=0.6)
    assert conf is False
    assert pytest_approx(t, 1.1)

    # 4. Pedestre atinge 1.5s (mais 0.4s -> 1.5s)
    conf, t, prog = debounce.atualizar(pedestre_presente=True, dt=0.4)
    assert conf is True
    assert pytest_approx(t, 1.5)
    assert prog == 1.0

    # 5. Pedestre sai da zona (pedestre_presente=False) -> reseta
    conf, t, prog = debounce.atualizar(pedestre_presente=False, dt=0.1)
    assert conf is False
    assert t == 0.0
    assert prog == 0.0


def test_detector_roi_e_inferencia():
    # 1. Teste geométrico de ROI e conversão
    roi_norm = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    shape = (480, 640, 3)
    poly = DetectorTrafego.roi_para_pixels(roi_norm, shape)
    assert poly.shape == (4, 2)
    assert DetectorTrafego.ponto_dentro_roi((320, 240), poly) is True
    assert DetectorTrafego.ponto_dentro_roi((700, 500), poly) is False

    # 2. Teste de inicialização do detector YOLO e inferência com frame sintético
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'))
    assert os.path.exists(model_path), f"Modelo não encontrado em {model_path}"
    detector = DetectorTrafego(model_path=model_path, confianca=0.35, imgsz=320)
    assert detector.model is not None

    frame_dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    res = detector.processar_frame(frame_dummy, roi_normalizada=roi_norm)

    assert 'veiculos' in res
    assert 'pedestres' in res
    assert 'animais' in res
    assert 'detalhes' in res
    assert 'frame_anotado' in res
    assert isinstance(res['veiculos'], int)
    assert isinstance(res['pedestres'], int)
    assert isinstance(res['animais'], int)


def test_detector_ponto_contato_pes_pessoa():
    """
    Garante que para pessoas o ponto avaliado são os pés ((x1+x2)//2, y2)
    contra a ROI de pedestres, evitando falsos positivos por inclinação.
    """
    detector = DetectorTrafego.__new__(DetectorTrafego)
    detector.confianca = 0.35
    detector.imgsz = 320
    detector.pular_frames = 1
    detector.frame_count = 0
    detector.usar_ponto_contato = True
    detector.ultimo_resultado = None
    detector.classes_alvo = [0, 1, 2, 3, 5, 7]

    # Simula mock do resultado YOLO com uma pessoa em x1=100, y1=100, x2=200, y2=300
    # Centro: (150, 200) | Pés: (150, 300)
    mock_box = MagicMock()
    mock_box.cls = [CLASSE_PESSOA]
    mock_box.xyxy = [np.array([100, 100, 200, 300])]
    mock_box.id = [1]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    
    mock_model = MagicMock()
    mock_model.track.return_value = [mock_res]
    detector.model = mock_model

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Caso A: ROI de pedestres cobre o chão (pés y=300 estão dentro da ROI [250 a 350])
    # Normalizado para frame 400x400: x=[0.25, 0.75], y=[0.60, 0.90] -> x=[100, 300], y=[240, 360]
    roi_ped_cobre_pes = [[0.25, 0.60], [0.75, 0.60], [0.75, 0.90], [0.25, 0.90]]
    res_a = detector.processar_frame(frame.copy(), roi_pedestres_normalizada=roi_ped_cobre_pes)
    assert res_a['pedestres'] == 1
    assert res_a['detalhes'][0]['ponto_pes'] == (150, 300)

    # Caso B: ROI de pedestres cobre apenas a cabeça/tronco (y=[100, 220]), mas não os pés (y=300)
    roi_ped_cobre_tronco = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.55], [0.25, 0.55]]
    res_b = detector.processar_frame(frame.copy(), roi_pedestres_normalizada=roi_ped_cobre_tronco)
    assert res_b['pedestres'] == 0


def test_detector_exclusao_passaros():
    """
    Garante que a classe de pássaro (ID 14) foi completamente removida de CLASSES_ANIMAIS
    e de CLASSES_ALVO para evitar que pássaros em fios ou voando acionem o semáforo.
    Apenas animais terrestres (15: gato, 16: cachorro, 17: cavalo, 18: ovelha, 19: vaca) são considerados.
    """
    assert 14 not in CLASSES_ANIMAIS
    assert 14 not in CLASSES_ALVO
    assert 15 in CLASSES_ANIMAIS  # gato
    assert 16 in CLASSES_ANIMAIS  # cachorro
    assert 17 in CLASSES_ANIMAIS  # cavalo
    assert 18 in CLASSES_ANIMAIS  # ovelha
    assert 19 in CLASSES_ANIMAIS  # vaca


def test_detector_ponto_contato_rodas_veiculo_e_patas_animal():
    """
    Valida a detecção pelo ponto de contato com o solo: 
    - Rodas para veículos ((x1+x2)//2, y2)
    - Patas para animais ((x1+x2)//2, y2)
    - Alternância para centróide quando usar_ponto_contato=False.
    """
    detector = DetectorTrafego.__new__(DetectorTrafego)
    detector.confianca = 0.35
    detector.imgsz = 320
    detector.pular_frames = 1
    detector.frame_count = 0
    detector.usar_ponto_contato = True
    detector.ultimo_resultado = None
    detector.classes_alvo = CLASSES_ALVO

    # Carro (cls=2) de y1=100 a y2=300, centro y=200, rodas y=300
    box_carro = MagicMock()
    box_carro.cls = [2]
    box_carro.xyxy = [np.array([100, 100, 200, 300])]
    box_carro.id = [1]

    # Cachorro (cls=16) de y1=100 a y2=300, centro y=200, patas y=300
    box_animal = MagicMock()
    box_animal.cls = [16]
    box_animal.xyxy = [np.array([250, 100, 350, 300])]
    box_animal.id = [2]

    mock_res = MagicMock()
    mock_res.boxes = [box_carro, box_animal]

    mock_model = MagicMock()
    mock_model.track.return_value = [mock_res]
    detector.model = mock_model

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # ROI cobre a pista no solo: y=[250, 350] (y=300 está DENTRO da pista, mas centro y=200 está FORA)
    roi_solo = [[0.10, 0.60], [0.90, 0.60], [0.90, 0.90], [0.10, 0.90]]

    # 1. Modo Ponto de Contato (Solo): Rodas e Patas (y=300) dentro da ROI -> Ambos detectados
    detector.usar_ponto_contato = True
    res_solo = detector.processar_frame(frame.copy(), roi_veiculos_normalizada=roi_solo)
    assert res_solo['veiculos'] == 1
    assert res_solo['animais'] == 1
    assert res_solo['detalhes'][0]['ponto_teste'] == (150, 300)
    assert res_solo['detalhes'][1]['ponto_teste'] == (300, 300)

    # 2. Modo Centróide: Centro (y=200) fora da ROI -> Nenhum detectado
    detector.usar_ponto_contato = False
    res_centro = detector.processar_frame(frame.copy(), roi_veiculos_normalizada=roi_solo)
    assert res_centro['veiculos'] == 0
    assert res_centro['animais'] == 0


def pytest_approx(val, expected, tol=1e-3):
    return abs(val - expected) < tol
