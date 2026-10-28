import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.detector import DetectorTrafego

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
    print("[TESTE PASSOU] DetectorTrafego (ROI + inferência YOLO) validado 100%!")

if __name__ == '__main__':
    test_detector_roi_e_inferencia()
