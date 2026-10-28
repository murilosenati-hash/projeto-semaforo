import cv2
import numpy as np
from ultralytics import YOLO

# Mapeamento oficial de IDs de classes do COCO
CLASSE_PESSOA = 0
CLASSES_VEICULOS = {
    1: "Bicicleta",
    2: "Carro",
    3: "Moto",
    5: "Onibus",
    7: "Caminhao"
}
CLASSES_ANIMAIS = {
    14: "Passaro",
    15: "Gato",
    16: "Cachorro",
    17: "Cavalo",
    18: "Ovelha",
    19: "Vaca"
}

class DetectorTrafego:
    def __init__(self, model_path="yolov8n.pt", confianca=0.35, pular_frames=3, imgsz=320):
        self.model = YOLO(model_path)
        self.confianca = confianca
        self.pular_frames = pular_frames
        self.imgsz = imgsz
        self.frame_count = 0
        self.ultimo_resultado = None

    @staticmethod
    def roi_para_pixels(roi, shape):
        height, width = shape[:2]
        return np.array([(int(x * width), int(y * height)) for x, y in roi], dtype=np.int32)

    @staticmethod
    def ponto_dentro_roi(ponto, poligono_roi):
        return cv2.pointPolygonTest(poligono_roi, ponto, False) >= 0

    def processar_frame(self, frame, roi_normalizada=None):
        """
        Processa um frame e retorna contagens de veículos, pessoas e animais,
        além de desenhar os bounding boxes e ROI anotados.
        """
        self.frame_count += 1
        deve_inferir = (self.frame_count % self.pular_frames == 0) or (self.ultimo_resultado is None)

        if deve_inferir:
            self.ultimo_resultado = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confianca,
                imgsz=self.imgsz,
                verbose=False
            )[0]

        result = self.ultimo_resultado
        roi_poly = None
        if roi_normalizada:
            roi_poly = self.roi_para_pixels(roi_normalizada, frame.shape)

        contagem_veiculos = 0
        contagem_pessoas = 0
        contagem_animais = 0
        detalhes = []

        if result is not None and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Filtra pela ROI se fornecida
                dentro_roi = True
                if roi_poly is not None:
                    dentro_roi = self.ponto_dentro_roi(centro, roi_poly)

                if not dentro_roi:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None

                if cls_id == CLASSE_PESSOA:
                    contagem_pessoas += 1
                    tipo = "Pessoa"
                    cor = (255, 120, 0)
                elif cls_id in CLASSES_VEICULOS:
                    contagem_veiculos += 1
                    tipo = CLASSES_VEICULOS[cls_id]
                    cor = (0, 220, 0)
                elif cls_id in CLASSES_ANIMAIS:
                    contagem_animais += 1
                    tipo = f"Animal ({CLASSES_ANIMAIS[cls_id]})"
                    cor = (0, 165, 255) # Laranja
                else:
                    continue

                detalhes.append({
                    "tipo": tipo,
                    "classe_id": cls_id,
                    "bbox": (x1, y1, x2, y2),
                    "centro": centro,
                    "track_id": track_id
                })

                # Desenha BBox e rótulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                rotulo = f"{tipo} #{track_id}" if track_id else tipo
                cv2.putText(frame, rotulo, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                cv2.circle(frame, centro, 3, cor, -1)

        if roi_poly is not None:
            cv2.polylines(frame, [roi_poly], True, (0, 255, 255), 2)

        return {
            "veiculos": contagem_veiculos,
            "pedestres": contagem_pessoas,
            "animais": contagem_animais,
            "detalhes": detalhes,
            "frame_anotado": frame
        }
