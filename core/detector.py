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


class DebouncePedestre:
    """
    Controla o debounce de permanência para pedestres na zona de espera.
    Exige tempo contínuo de permanência (padrão 1.5s) para acionar a FSM.
    """
    def __init__(self, tempo_confirmacao: float = 1.5):
        self.tempo_confirmacao = float(tempo_confirmacao)
        self.tempo_acumulado = 0.0
        self.confirmado = False

    def atualizar(self, pedestre_presente: bool, dt: float) -> tuple[bool, float, float]:
        """
        Atualiza o acumulador com o delta de tempo dt.
        Retorna: (confirmado: bool, tempo_acumulado: float, progresso: float [0.0 a 1.0])
        """
        if dt < 0:
            dt = 0.0
        
        if pedestre_presente:
            self.tempo_acumulado += dt
            if self.tempo_acumulado >= self.tempo_confirmacao:
                self.confirmado = True
        else:
            self.tempo_acumulado = 0.0
            self.confirmado = False

        progresso = min(1.0, self.tempo_acumulado / self.tempo_confirmacao) if self.tempo_confirmacao > 0 else 1.0
        return self.confirmado, self.tempo_acumulado, progresso

    def resetar(self):
        self.tempo_acumulado = 0.0
        self.confirmado = False


class DetectorTrafego:
    def __init__(self, model_path="yolov8n.pt", confianca=0.35, pular_frames=3, imgsz=320):
        self.model = YOLO(model_path)
        self.confianca = confianca
        self.pular_frames = pular_frames
        self.imgsz = imgsz
        self.frame_count = 0
        self.ultimo_resultado = None
        # Classes de interesse para filtrar direto no NMS do YOLO
        self.classes_alvo = [CLASSE_PESSOA] + list(CLASSES_VEICULOS.keys()) + list(CLASSES_ANIMAIS.keys())

    @staticmethod
    def roi_para_pixels(roi, shape):
        height, width = shape[:2]
        return np.array([(int(x * width), int(y * height)) for x, y in roi], dtype=np.int32)

    @staticmethod
    def ponto_dentro_roi(ponto, poligono_roi):
        return cv2.pointPolygonTest(poligono_roi, (float(ponto[0]), float(ponto[1])), False) >= 0

    def processar_frame(self, frame, roi_normalizada=None, roi_veiculos_normalizada=None, roi_pedestres_normalizada=None):
        """
        Processa um frame e retorna contagens de veículos, pessoas e animais,
        além de desenhar os bounding boxes e ROIs anotadas.
        
        Filtragem Espacial Estrita:
        - Veículos e Animais são testados pelo centroide contra a ROI de veículos.
        - Pessoas são testadas pelo ponto inferior de contato com o solo ((x1+x2)//2, y2) contra a ROI de pedestres.
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
                classes=self.classes_alvo,
                verbose=False
            )[0]

        result = self.ultimo_resultado

        # Resolução de ROI de Veículos
        roi_v_norm = roi_veiculos_normalizada if roi_veiculos_normalizada is not None else roi_normalizada
        roi_poly_veic = self.roi_para_pixels(roi_v_norm, frame.shape) if roi_v_norm else None

        # Resolução de ROI de Pedestres
        roi_poly_ped = self.roi_para_pixels(roi_pedestres_normalizada, frame.shape) if roi_pedestres_normalizada else None

        contagem_veiculos = 0
        contagem_pessoas = 0
        contagem_animais = 0
        detalhes = []

        if result is not None and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                ponto_pes = ((x1 + x2) // 2, y2)

                # Filtragem espacial por classe e ponto de contato
                if cls_id == CLASSE_PESSOA:
                    # Ponto de contato com o solo (pés)
                    if roi_poly_ped is not None:
                        dentro_roi = self.ponto_dentro_roi(ponto_pes, roi_poly_ped)
                    elif roi_poly_veic is not None:
                        dentro_roi = self.ponto_dentro_roi(ponto_pes, roi_poly_veic)
                    else:
                        dentro_roi = True
                else:
                    # Centroide para veículos e animais na pista
                    if roi_poly_veic is not None:
                        dentro_roi = self.ponto_dentro_roi(centro, roi_poly_veic)
                    else:
                        dentro_roi = True

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
                    "ponto_pes": ponto_pes,
                    "track_id": track_id
                })

                # Desenha BBox e rótulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                rotulo = f"{tipo} #{track_id}" if track_id else tipo
                cv2.putText(frame, rotulo, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                
                # Marcador de ponto de contato com o solo para pedestres
                if cls_id == CLASSE_PESSOA:
                    cv2.circle(frame, ponto_pes, 5, (0, 255, 255), -1)
                    cv2.putText(frame, "[PES]", (ponto_pes[0] - 14, ponto_pes[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                else:
                    cv2.circle(frame, centro, 3, cor, -1)

        # Renderização das duas ROIs se presentes
        if roi_poly_veic is not None:
            cv2.polylines(frame, [roi_poly_veic], True, (0, 215, 255), 2)
        if roi_poly_ped is not None:
            cor_borda_ped = (0, 255, 0) if contagem_pessoas > 0 else (255, 255, 0)
            cv2.polylines(frame, [roi_poly_ped], True, cor_borda_ped, 2)

        return {
            "veiculos": contagem_veiculos,
            "pedestres": contagem_pessoas,
            "animais": contagem_animais,
            "detalhes": detalhes,
            "frame_anotado": frame
        }
