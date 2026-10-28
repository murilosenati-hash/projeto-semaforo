import cv2
import numpy as np
import time
import argparse
import os
from core.detector import DetectorTrafego
from core.fsm import ControladorSemaforoFSM, EstadoSemaforo

def desenhar_painel_semaforo(frame, fsm):
    """
    Desenha um semáforo virtual digital no canto do frame com luzes realistas.
    """
    h, w = frame.shape[:2]
    luz_carro_r, luz_carro_y, luz_carro_g, luz_ped_r, luz_ped_g = fsm.obter_luzes()

    # Painel de fundo semáforo veículos
    cv2.rectangle(frame, (w - 240, 20), (w - 140, 250), (25, 25, 25), -1)
    cv2.rectangle(frame, (w - 240, 20), (w - 140, 250), (180, 180, 180), 2)

    # Luzes Veículo
    cor_r = (0, 0, 255) if luz_carro_r else (0, 0, 50)
    cor_y = (0, 230, 255) if luz_carro_y else (0, 50, 50)
    cor_g = (0, 255, 0) if luz_carro_g else (0, 50, 0)

    cv2.circle(frame, (w - 190, 60), 25, cor_r, -1)
    cv2.circle(frame, (w - 190, 130), 25, cor_y, -1)
    cv2.circle(frame, (w - 190, 200), 25, cor_g, -1)
    cv2.putText(frame, "VEICULOS", (w - 230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Painel semáforo pedestres
    cv2.rectangle(frame, (w - 120, 20), (w - 30, 180), (25, 25, 25), -1)
    cv2.rectangle(frame, (w - 120, 20), (w - 30, 180), (180, 180, 180), 2)

    cor_ped_r = (0, 0, 255) if luz_ped_r else (0, 0, 50)
    cor_ped_g = (0, 255, 0) if luz_ped_g else (0, 50, 0)

    cv2.circle(frame, (w - 75, 65), 22, cor_ped_r, -1)
    cv2.circle(frame, (w - 75, 135), 22, cor_ped_g, -1)
    cv2.putText(frame, "PEDESTRE", (w - 115, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # HUD Superior com Estado e Telemetria
    tempo_no_estado = time.time() - fsm.inicio_estado
    cv2.rectangle(frame, (20, 20), (450, 140), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (450, 140), (0, 255, 255), 2)

    cv2.putText(frame, f"ESTADO: {fsm.estado_atual.value}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"TEMPO NO ESTADO: {tempo_no_estado:.1f}s / Meta: {fsm.tempo_verde_calculado:.0f}s", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"SENSORES: {fsm.veiculos_detectados} Veic | {fsm.pedestres_detectados} Ped | {fsm.animais_detectados} Animal", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "TECLAS: [P] Pedestre | [A] Animal | [Q] Sair", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="Semáforo Inteligente Distribuído")
    parser.add_argument("--source", default="transito.mp4", help="Fonte de vídeo: arquivo, webcam (0) ou RTSP/IP")
    parser.add_argument("--yolo", default="yolov8n.pt", help="Caminho do modelo YOLOv8")
    args = parser.parse_args()

    fonte = args.source
    if fonte.isdigit():
        fonte = int(fonte)
    elif not os.path.exists(fonte) and fonte != 0:
        # Se não achou o vídeo local, tenta webcam
        print(f"[AVISO] Arquivo {fonte} não encontrado. Tentando webcam padrão (0)...")
        fonte = 0

    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {fonte}")
        return

    detector = DetectorTrafego(model_path=args.yolo, confianca=0.35, pular_frames=2, imgsz=320)
    fsm = ControladorSemaforoFSM(
        min_verde_carro=6.0,
        max_verde_carro=25.0,
        tempo_amarelo=3.0,
        tempo_limpeza=2.0,
        tempo_verde_pedestre=8.0,
        tempo_pisca_pedestre=3.0
    )

    # ROI padrão normalizada cobrindo a pista central
    roi = [(0.15, 0.30), (0.85, 0.30), (0.98, 0.98), (0.02, 0.98)]

    pedestre_manual = False
    animal_manual = False
    inicio_manual = 0

    print("=== SEMÁFORO INTELIGENTE ATIVO ===")
    print("Pressione P para simular pedestre na faixa")
    print("Pressione A para simular animal na via")
    print("Pressione Q para encerrar")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Se for vídeo e acabou, reinicia o vídeo em loop para demonstração contínua
            if isinstance(fonte, str) and os.path.exists(fonte):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        resultado = detector.processar_frame(frame, roi_normalizada=roi)

        # Se foi acionado manualmente via teclado, mantém por 8s
        if (pedestre_manual or animal_manual) and (time.time() - inicio_manual > 8.0):
            pedestre_manual = False
            animal_manual = False

        veiculos = resultado["veiculos"]
        pedestres = resultado["pedestres"] or (1 if pedestre_manual else 0)
        animais = resultado["animais"] or (1 if animal_manual else 0)

        fsm.atualizar_sensores(veiculos=veiculos, pedestres=pedestres, animais=animais)
        fsm.tick()

        frame_anotado = resultado["frame_anotado"]
        desenhar_painel_semaforo(frame_anotado, fsm)

        cv2.imshow("Semaforo Inteligente - Sistemas Distribuidos & Digitais", frame_anotado)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pedestre_manual = True
            inicio_manual = time.time()
            print("[INTERAÇÃO] Botão virtual de pedestre acionado! Solicitando fase verde para travessia...")
        elif key == ord('a'):
            animal_manual = True
            inicio_manual = time.time()
            print("[INTERAÇÃO] Animal detectado na pista! Acionando freio e bloqueio de segurança...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
