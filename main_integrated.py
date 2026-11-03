import cv2
import numpy as np
import time
import argparse
import os
import json
from core.detector import DetectorTrafego
from core.fsm import ControladorSemaforoFSM, EstadoSemaforo
from core.video_stream import ThreadedVideoStream
from core.traffic_simulator import SimuladorTrafegoVirtual

CONFIG_ROI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_roi.json')
ROI_PADRAO = [(0.15, 0.30), (0.85, 0.30), (0.98, 0.98), (0.02, 0.98)]


def carregar_roi(caminho=None):
    """
    Carrega a ROI configurada do arquivo JSON. Se inexistente ou corrompido,
    retorna uma cópia da ROI_PADRAO.
    """
    if caminho is None:
        caminho = CONFIG_ROI_PATH
    try:
        if os.path.exists(caminho):
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                if isinstance(dados, list) and len(dados) == 4:
                    return [[float(p[0]), float(p[1])] for p in dados]
    except Exception as e:
        print(f"[AVISO] Erro ao carregar ROI de {caminho}: {e}. Usando ROI padrão.")
    return [list(p) for p in ROI_PADRAO]


def salvar_roi(roi, caminho=None):
    """
    Salva a ROI configurada em arquivo JSON seguro.
    """
    if caminho is None:
        caminho = CONFIG_ROI_PATH
    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(roi, f, indent=2)
        print(f"[CONFIG] ROI salva com sucesso em {caminho}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao salvar ROI em {caminho}: {e}")
        return False


def validar_poligono_4_pontos(pts):
    """
    Valida se os 4 pontos formam um polígono válido e não-degenerado:
    - Área mínima >= 1000 px
    - Contorno estritamente convexo (impede auto-interseção / formato de 8)
    Retorna: (valido: bool, mensagem: str)
    """
    if len(pts) != 4:
        return False, f"Necessário exatamente 4 pontos (fornecidos: {len(pts)})"
    
    cnt = np.array(pts, dtype=np.int32)
    if not cv2.isContourConvex(cnt):
        return False, "Polígono não-convexo ou auto-intersectado (formato de 8). Clique na ordem horária."

    area = cv2.contourArea(cnt)
    if area < 1000.0:
        return False, f"Área insuficiente ({area:.0f}px < 1000px). Evite pontos colineares."

    return True, "OK"


def normalizar_pontos_roi(pts, fw, fh):
    """
    Converte pontos de pixels para coordenadas normalizadas [0.0, 1.0] com clamp seguro.
    """
    fw = max(float(fw), 1.0)
    fh = max(float(fh), 1.0)
    return [
        [
            round(min(max(float(p[0]) / fw, 0.0), 1.0), 4),
            round(min(max(float(p[1]) / fh, 0.0), 1.0), 4)
        ]
        for p in pts
    ]


def selecionar_poligono_4_pontos(nome_janela, frame_base, roi_atual=None):
    """
    Seletor interativo de 4 vértices para calibração de perspectiva de ROI.
    Exibe Banner OSD no frame e validação de geometria convexa.
    """
    fh, fw = frame_base.shape[:2]
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    largura_janela = min(1280, fw) if fw > 0 else 1024
    altura_janela = int(largura_janela * (fh / fw)) if fw > 0 else 576
    cv2.resizeWindow(nome_janela, largura_janela, altura_janela)

    pontos_clicados = []
    msg_status = ""
    status_cor = (0, 255, 255)

    def mouse_callback(event, x, y, flags, param):
        nonlocal msg_status, status_cor
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pontos_clicados) < 4:
                pontos_clicados.append([x, y])
                msg_status = ""
                if len(pontos_clicados) == 4:
                    valido, motivo = validar_poligono_4_pontos(pontos_clicados)
                    if valido:
                        msg_status = "Poligono Completo! [ENTER/ESPACO] Salvar | [C] Limpar | [ESC] Cancelar"
                        status_cor = (0, 255, 0)
                    else:
                        msg_status = f"[INVALIDO] {motivo} | [C] Limpar"
                        status_cor = (0, 0, 255)

    cv2.setMouseCallback(nome_janela, mouse_callback)

    try:
        while True:
            frame_vis = frame_base.copy()

            # Banner OSD semitransparente superior
            overlay_osd = frame_vis.copy()
            cv2.rectangle(overlay_osd, (0, 0), (fw, 64), (20, 20, 20), -1)
            cv2.addWeighted(overlay_osd, 0.75, frame_vis, 0.25, 0, frame_vis)
            cv2.line(frame_vis, (0, 64), (fw, 64), (100, 100, 100), 1)

            # Texto do Banner OSD
            n_pts = len(pontos_clicados)
            if n_pts < 4:
                texto_banner = f"MODO CALIBRACAO ROI | Clique no Ponto {n_pts + 1}/4 | [C] Limpar | [ESC] Cancelar"
                cor_banner = (0, 255, 255)
            else:
                texto_banner = msg_status if msg_status else "Poligono Completo! [ENTER/ESPACO] Salvar | [C] Limpar"
                cor_banner = status_cor

            cv2.putText(frame_vis, texto_banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_banner, 2)

            # Desenho dos vértices e arestas
            if n_pts > 0:
                for idx, pt in enumerate(pontos_clicados):
                    # Círculo com borda contrastante
                    cv2.circle(frame_vis, tuple(pt), 7, (0, 0, 0), -1)
                    cv2.circle(frame_vis, tuple(pt), 5, (0, 255, 255), -1)
                    cv2.putText(frame_vis, f"P{idx + 1}", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                # Conecta pontos consecutivos
                for idx in range(n_pts - 1):
                    cv2.line(frame_vis, tuple(pontos_clicados[idx]), tuple(pontos_clicados[idx + 1]), (0, 255, 255), 2)

                # Se tiver os 4 pontos, fecha o polígono e aplica preenchimento semitransparente
                if n_pts == 4:
                    valido, _ = validar_poligono_4_pontos(pontos_clicados)
                    cor_linha = (0, 255, 0) if valido else (0, 0, 255)
                    cv2.line(frame_vis, tuple(pontos_clicados[3]), tuple(pontos_clicados[0]), cor_linha, 2)

                    if valido:
                        overlay_poly = frame_vis.copy()
                        poly_pts = np.array([pontos_clicados], dtype=np.int32)
                        cv2.fillPoly(overlay_poly, poly_pts, (0, 220, 100))
                        cv2.addWeighted(overlay_poly, 0.30, frame_vis, 0.70, 0, frame_vis)

            cv2.imshow(nome_janela, frame_vis)
            key = cv2.waitKey(20) & 0xFF

            # Tecla ESC (27): aborta e mantém ROI anterior
            if key == 27:
                print("[ROI] Calibração cancelada pelo usuário (ESC).")
                return None
            
            # Tecla 'C' / 'c': limpa pontos para reclicar
            elif key == ord('c') or key == ord('C'):
                pontos_clicados.clear()
                msg_status = ""
                status_cor = (0, 255, 255)
                print("[ROI] Pontos reiniciados. Clique novamente nos 4 cantos.")

            # SPACE (32) ou ENTER (13): confirma se 4 pontos válidos
            elif key in (13, 32):
                if len(pontos_clicados) == 4:
                    valido, motivo = validar_poligono_4_pontos(pontos_clicados)
                    if valido:
                        roi_norm = normalizar_pontos_roi(pontos_clicados, fw, fh)
                        print(f"[ROI] Calibração concluída com sucesso! Vértices: {roi_norm}")
                        return roi_norm
                    else:
                        msg_status = f"[INVALIDO] {motivo} | [C] Limpar"
                        status_cor = (0, 0, 255)
                else:
                    msg_status = f"Selecione os 4 pontos antes de salvar ({len(pontos_clicados)}/4)"
                    status_cor = (0, 0, 255)
    finally:
        # Deregistração obrigatória do callback de mouse
        cv2.setMouseCallback(nome_janela, lambda *args: None)


def desenhar_painel_semaforo(frame, fsm, ar_ativo=False):
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
    cv2.rectangle(frame, (20, 20), (490, 140), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (490, 140), (0, 255, 255), 2)

    tag_ar = " [AR ATIVO]" if ar_ativo else ""
    cv2.putText(frame, f"ESTADO: {fsm.estado_atual.value}{tag_ar}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"TEMPO NO ESTADO: {tempo_no_estado:.1f}s / Meta: {fsm.tempo_verde_calculado:.0f}s", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"SENSORES: {fsm.veiculos_detectados} Veic | {fsm.pedestres_detectados} Ped | {fsm.animais_detectados} Animal", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "TECLAS: [P] Ped | [A] Anim | [R] ROI | [V] AR | [C] Spawn | [X] Clean", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="Semáforo Inteligente Distribuído")
    parser.add_argument("--source", default="transito.mp4", help="Fonte de vídeo: arquivo, webcam (0) ou RTSP/IP")
    parser.add_argument("--yolo", default="yolov8n.pt", help="Caminho do modelo YOLOv8")
    parser.add_argument("--skip", type=int, default=3, help="Pulo de frames (padrão: 3)")
    parser.add_argument("--imgsz", type=int, default=320, help="Resolução da inferência (padrão: 320)")
    parser.add_argument("--simular-carros", action="store_true", default=False, help="Ativa simulação AR de carros virtuais na via")
    parser.add_argument("--carros-spawn", type=float, default=3.5, help="Intervalo em segundos entre spawn de carros virtuais")
    parser.add_argument("--carros-max", type=int, default=5, help="Capacidade máxima de carros virtuais na pista")
    args = parser.parse_args()

    fonte_str = str(args.source).strip()
    if fonte_str.isdigit():
        fonte = int(fonte_str)
    elif fonte_str.startswith(("http://", "https://", "rtsp://")):
        fonte = fonte_str
    elif not os.path.exists(fonte_str) and fonte_str != "0":
        print(f"[AVISO] Arquivo {fonte_str} não encontrado. Tentando webcam padrão (0)...")
        fonte = 0
    else:
        fonte = fonte_str

    nome_janela = "Semaforo Inteligente - Sistemas Distribuidos & Digitais"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)

    # Thread desacoplada de captura
    stream = ThreadedVideoStream(src=fonte).start()
    time.sleep(0.3)

    detector = DetectorTrafego(model_path=args.yolo, confianca=0.35, pular_frames=args.skip, imgsz=args.imgsz)
    fsm = ControladorSemaforoFSM(
        min_verde_carro=6.0,
        max_verde_carro=25.0,
        tempo_amarelo=3.0,
        tempo_limpeza=2.0,
        tempo_verde_pedestre=8.0,
        tempo_pisca_pedestre=3.0
    )

    # Inicializa simulador de tráfego AR (Nível 2 Híbrido)
    simulador_ar = SimuladorTrafegoVirtual(
        taxa_spawn=args.carros_spawn,
        stop_line=0.55,
        max_veiculos=args.carros_max
    )
    ar_ativo = args.simular_carros

    # Carrega ROI persistida ou fallback padrão
    roi = carregar_roi()

    pedestre_manual = False
    animal_manual = False
    inicio_manual = 0
    tempo_anterior = time.time()
    janela_configurada = False

    print("=== SEMÁFORO INTELIGENTE ATIVO (OTIMIZADO) ===")
    print(f"[CONFIG] Resolução: {args.imgsz}x{args.imgsz} | Salto: {args.skip} frames | I/O Thread: Ativa")
    print(f"[CONFIG] Simulação AR: {'ATIVADA' if ar_ativo else 'DESATIVADA'} | Tecla 'V' para alternar")
    print("Pressione P para simular pedestre na faixa")
    print("Pressione A para simular animal na via")
    print("Pressione R para calibrar a Região de Interesse (ROI) via 4 vértices")
    print("Pressione V para alternar Simulação AR de Carros")
    print("Pressione C para Spawn Imediato de Carro AR")
    print("Pressione X para Limpar Carros AR da Pista")
    print("Pressione Q para encerrar")

    while True:
        agora = time.time()
        dt = agora - tempo_anterior
        tempo_anterior = agora

        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        if not janela_configurada:
            h, w = frame.shape[:2]
            largura_janela = min(1280, w) if w > 0 else 1024
            altura_janela = int(largura_janela * (h / w)) if w > 0 else 576
            cv2.resizeWindow(nome_janela, largura_janela, altura_janela)
            janela_configurada = True

        # 1. Inferência YOLO para detectar pedestres/animais reais na cena física
        resultado = detector.processar_frame(frame, roi_normalizada=roi)

        # Se foi acionado manualmente via teclado, mantém por 8s
        if (pedestre_manual or animal_manual) and (time.time() - inicio_manual > 8.0):
            pedestre_manual = False
            animal_manual = False

        pedestres = resultado["pedestres"] or (1 if pedestre_manual else 0)
        animais = resultado["animais"] or (1 if animal_manual else 0)

        # 2. Fusão Sensorial Híbrida: Carros Virtuais AR vs YOLO real
        if ar_ativo:
            simulador_ar.atualizar(dt, fsm.estado_atual)
            veiculos = simulador_ar.obter_contagem_veiculos_roi()
        else:
            veiculos = resultado["veiculos"]

        fsm.atualizar_sensores(veiculos=veiculos, pedestres=pedestres, animais=animais)
        fsm.tick()

        frame_anotado = resultado["frame_anotado"]

        # 3. Renderização em camadas: frame -> carros AR e Stop Line -> HUD Semáforo
        if ar_ativo:
            roi_px = DetectorTrafego.roi_para_pixels(roi, frame.shape)
            simulador_ar.renderizar(frame_anotado, roi_px, fsm.estado_atual)

        desenhar_painel_semaforo(frame_anotado, fsm, ar_ativo=ar_ativo)

        cv2.imshow(nome_janela, frame_anotado)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            pedestre_manual = True
            inicio_manual = time.time()
            print("[INTERAÇÃO] Botão virtual de pedestre acionado! Solicitando fase verde para travessia...")
        elif key == ord('a') or key == ord('A'):
            animal_manual = True
            inicio_manual = time.time()
            print("[INTERAÇÃO] Animal detectado na pista! Acionando freio e bloqueio de segurança...")
        elif key == ord('v') or key == ord('V'):
            ar_ativo = not ar_ativo
            print(f"[MODO AR] Simulação de veículos AR: {'LIGADA' if ar_ativo else 'DESLIGADA'}")
        elif key == ord('c') or key == ord('C'):
            if ar_ativo:
                spawnow = simulador_ar.spawn_imediato()
                print(f"[MODO AR] Spawn manual de veículo virtual: {'Sucesso' if spawnow else 'Pista ocupada ou limite atingido'}")
            else:
                print("[AVISO] Ative a simulação AR pressionando 'V' para spawnar veículos.")
        elif key == ord('x') or key == ord('X'):
            simulador_ar.limpar_pista()
            print("[MODO AR] Todos os veículos virtuais foram removidos da pista.")
        elif key == ord('r') or key == ord('R'):
            frame_congelado = frame.copy()
            t0 = time.time()
            nova_roi = selecionar_poligono_4_pontos(nome_janela, frame_congelado, roi_atual=roi)
            fsm.inicio_estado += (time.time() - t0)
            tempo_anterior = time.time()
            if nova_roi is not None:
                roi = nova_roi
                salvar_roi(roi)

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
