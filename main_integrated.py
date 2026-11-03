import cv2
import numpy as np
import time
import argparse
import os
import json
from core.detector import DetectorTrafego, DebouncePedestre
from core.fsm import ControladorSemaforoFSM, EstadoSemaforo
from core.video_stream import ThreadedVideoStream
from core.traffic_simulator import SimuladorTrafegoVirtual

CONFIG_ROI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_roi.json')
ROI_PADRAO_VEICULOS = [(0.15, 0.30), (0.85, 0.30), (0.98, 0.98), (0.02, 0.98)]
ROI_PADRAO_PEDESTRES = [(0.02, 0.50), (0.22, 0.50), (0.22, 0.95), (0.02, 0.95)]
ROI_PADRAO = ROI_PADRAO_VEICULOS


def carregar_rois(caminho=None) -> dict:
    """
    Carrega as ROIs poligonais independentes de veículos e pedestres do arquivo JSON.
    Garante retrocompatibilidade total:
    - Se o arquivo contiver apenas uma lista, trata como ROI de veículos e preenche pedestres com padrão.
    - Suporta lista simples de pontos [ponto1, ponto2, ...] ou lista de polígonos [[p1, ...], [p2, ...]].
    """
    if caminho is None:
        caminho = CONFIG_ROI_PATH
    
    rois: dict = {
        "veiculos": [list(p) for p in ROI_PADRAO_VEICULOS],
        "pedestres": [list(p) for p in ROI_PADRAO_PEDESTRES]
    }
    
    try:
        if os.path.exists(caminho):
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    if "veiculos" in dados and isinstance(dados["veiculos"], list):
                        rois["veiculos"] = [[float(p[0]), float(p[1])] for p in dados["veiculos"]]
                    if "pedestres" in dados and isinstance(dados["pedestres"], list):
                        raw_ped = dados["pedestres"]
                        if len(raw_ped) > 0:
                            # Se for lista de polígonos: [[[x, y], ...], [[x, y], ...]]
                            if isinstance(raw_ped[0], (list, tuple)) and len(raw_ped[0]) > 0 and isinstance(raw_ped[0][0], (list, tuple, np.ndarray)):
                                rois["pedestres"] = [
                                    [[float(p[0]), float(p[1])] for p in poly]
                                    for poly in raw_ped if isinstance(poly, (list, tuple)) and len(poly) >= 3
                                ]
                            else:
                                # Lista simples de pontos [ponto1, ponto2, ...]
                                rois["pedestres"] = [[float(p[0]), float(p[1])] for p in raw_ped]
                elif isinstance(dados, list):
                    # Formato legado (apenas lista de veículos)
                    rois["veiculos"] = [[float(p[0]), float(p[1])] for p in dados]
    except Exception as e:
        print(f"[AVISO] Erro ao carregar ROIs de {caminho}: {e}. Usando ROIs padrão.")
        
    return rois


def carregar_roi(caminho=None):
    """Alias retrocompatível: retorna a lista de coordenadas da ROI de veículos."""
    return carregar_rois(caminho=caminho)["veiculos"]


def salvar_rois(rois: dict, caminho=None) -> bool:
    """Salva o dicionário completo com as ROIs de veículos e pedestres em arquivo JSON."""
    if caminho is None:
        caminho = CONFIG_ROI_PATH
    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(rois, f, indent=2)
        print(f"[CONFIG] ROIs salvas com sucesso em {caminho}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao salvar ROIs em {caminho}: {e}")
        return False


def salvar_roi(roi, caminho=None, tipo="veiculos", index=None) -> bool:
    """Atualiza e salva uma ROI específica ('veiculos' ou 'pedestres')."""
    if caminho is None:
        caminho = CONFIG_ROI_PATH
    rois = carregar_rois(caminho=caminho)
    if tipo == "pedestres":
        if index is not None:
            # Se for formato de lista de zonas
            if isinstance(rois["pedestres"], list) and len(rois["pedestres"]) > 0 and isinstance(rois["pedestres"][0], list) and len(rois["pedestres"][0]) > 0 and isinstance(rois["pedestres"][0][0], (list, tuple, np.ndarray)):
                if index < len(rois["pedestres"]):
                    rois["pedestres"][index] = roi
                else:
                    rois["pedestres"].append(roi)
            else:
                rois["pedestres"] = [roi]
        else:
            rois["pedestres"] = roi
    else:
        rois[tipo] = roi
    return salvar_rois(rois, caminho=caminho)


def validar_poligono(pts):
    """
    Valida se os pontos (N >= 3) formam um polígono válido e não-degenerado:
    - Área mínima >= 500 px
    - Contorno estritamente convexo (impede auto-interseção / formato de 8)
    Retorna: (valido: bool, mensagem: str)
    """
    if len(pts) < 3:
        return False, f"Necessário no mínimo 3 pontos (fornecidos: {len(pts)})"
    
    cnt = np.array(pts, dtype=np.int32)
    if not cv2.isContourConvex(cnt):
        return False, "Polígono não-convexo ou auto-intersectado (formato de 8). Clique na ordem horária."

    area = cv2.contourArea(cnt)
    if area < 500.0:
        return False, f"Área insuficiente ({area:.0f}px < 500px). Evite pontos colineares."

    return True, "OK"


def validar_poligono_4_pontos(pts):
    """Alias de compatibilidade."""
    return validar_poligono(pts)


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


def selecionar_poligono_livre(nome_janela, frame_base, roi_atual=None, tipo="veiculos", titulo_personalizado=None):
    """
    Seletor interativo de polígono livre de N vértices (N >= 3) para calibração de ROI.
    ENTER ou ESPAÇO finaliza e valida a geometria (área >= 500 px e convexidade).
    Permite calibrar tanto 'veiculos' quanto 'pedestres' com identidades visuais distintas.
    """
    fh, fw = frame_base.shape[:2]
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    largura_janela = min(1280, fw) if fw > 0 else 1024
    altura_janela = int(largura_janela * (fh / fw)) if fw > 0 else 576
    cv2.resizeWindow(nome_janela, largura_janela, altura_janela)

    if tipo.startswith("pedestres"):
        titulo_base = titulo_personalizado if titulo_personalizado else "CALIBRANDO ZONA DE PEDESTRES (ESPERA/FAIXA)"
        cor_tema = (255, 255, 0)      # Ciano
        cor_preenchimento = (255, 200, 0)
    else:
        titulo_base = titulo_personalizado if titulo_personalizado else "CALIBRANDO PISTA DE VEICULOS (ROI PRINCIPAL)"
        cor_tema = (0, 215, 255)      # Amarelo/Laranja
        cor_preenchimento = (0, 220, 100)

    pontos_clicados = []
    msg_status = ""
    status_cor = cor_tema

    def mouse_callback(event, x, y, flags, param):
        nonlocal msg_status, status_cor
        if event == cv2.EVENT_LBUTTONDOWN:
            pontos_clicados.append([x, y])
            msg_status = f"Ponto P{len(pontos_clicados)} marcado. [ENTER/ESPACO] para validar e salvar"
            status_cor = (0, 255, 255)

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
            if n_pts < 3:
                texto_banner = f"{titulo_base} | Vértices: {n_pts} (min 3) | [C] Limpar | [ESC] Cancelar"
                cor_banner = cor_tema
            else:
                texto_banner = msg_status if msg_status else f"{titulo_base} | {n_pts} vértices | [ENTER/ESPACO] Salvar | [C] Limpar"
                cor_banner = status_cor

            cv2.putText(frame_vis, texto_banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, cor_banner, 2)

            # Desenho dos vértices e arestas
            if n_pts > 0:
                for idx, pt in enumerate(pontos_clicados):
                    # Círculo com borda contrastante
                    cv2.circle(frame_vis, tuple(pt), 7, (0, 0, 0), -1)
                    cv2.circle(frame_vis, tuple(pt), 5, cor_tema, -1)
                    cv2.putText(frame_vis, f"P{idx + 1}", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_tema, 2)

                # Conecta pontos consecutivos
                for idx in range(n_pts - 1):
                    cv2.line(frame_vis, tuple(pontos_clicados[idx]), tuple(pontos_clicados[idx + 1]), cor_tema, 2)

                # Se tiver no mínimo 3 pontos, fecha o polígono e aplica preenchimento semitransparente
                if n_pts >= 3:
                    valido, _ = validar_poligono(pontos_clicados)
                    cor_linha = (0, 255, 0) if valido else (0, 0, 255)
                    cv2.line(frame_vis, tuple(pontos_clicados[-1]), tuple(pontos_clicados[0]), cor_linha, 2)

                    if valido:
                        overlay_poly = frame_vis.copy()
                        poly_pts = np.array([pontos_clicados], dtype=np.int32)
                        cv2.fillPoly(overlay_poly, poly_pts, cor_preenchimento)
                        cv2.addWeighted(overlay_poly, 0.30, frame_vis, 0.70, 0, frame_vis)

            cv2.imshow(nome_janela, frame_vis)
            key = cv2.waitKey(20) & 0xFF

            # Tecla ESC (27): aborta e mantém ROI anterior
            if key == 27:
                print(f"[ROI] Calibração de '{tipo}' cancelada pelo usuário (ESC).")
                return None
            
            # Tecla 'C' / 'c': limpa pontos para reclicar
            elif key == ord('c') or key == ord('C'):
                pontos_clicados.clear()
                msg_status = ""
                status_cor = cor_tema
                print(f"[ROI] Pontos de '{tipo}' reiniciados. Clique nos novos vértices.")

            # SPACE (32) ou ENTER (13): confirma se >= 3 pontos válidos
            elif key in (13, 32):
                if len(pontos_clicados) >= 3:
                    valido, motivo = validar_poligono(pontos_clicados)
                    if valido:
                        roi_norm = normalizar_pontos_roi(pontos_clicados, fw, fh)
                        print(f"[ROI] Calibração de '{tipo}' concluída com sucesso! Vértices: {roi_norm}")
                        return roi_norm
                    else:
                        msg_status = f"[INVALIDO] {motivo} | [C] Limpar"
                        status_cor = (0, 0, 255)
                else:
                    msg_status = f"Selecione no mínimo 3 pontos antes de salvar ({len(pontos_clicados)} marcados)"
                    status_cor = (0, 0, 255)
    finally:
        cv2.setMouseCallback(nome_janela, lambda *args: None)


# Alias de retrocompatibilidade
selecionar_poligono_4_pontos = selecionar_poligono_livre


def desenhar_painel_semaforo(frame, fsm, ar_ativo=False, modo_3d=False, debounce_info=None):
    """
    Desenha um semáforo virtual digital no canto do frame com luzes realistas
    e telemetria da FSM com feedback de debounce de pedestres.
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
    cv2.rectangle(frame, (20, 20), (520, 170), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (520, 170), (0, 255, 255), 2)

    tag_ar = " [AR 3D]" if (ar_ativo and modo_3d) else (" [AR 2D]" if ar_ativo else "")
    cv2.putText(frame, f"ESTADO: {fsm.estado_atual.value}{tag_ar}", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"TEMPO NO ESTADO: {tempo_no_estado:.1f}s / Meta: {fsm.tempo_verde_calculado:.0f}s", (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, f"SENSORES: {fsm.veiculos_detectados} Veic | {fsm.pedestres_detectados} Ped | {fsm.animais_detectados} Animal", (35, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
    cv2.putText(frame, "TECLAS: [P] Ped | [A] Anim | [R] Pista | [E] Zonas Ped | [V] AR | [M] 3D", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    # Linha de Feedback de Debounce de Pedestre
    if debounce_info is not None:
        confirmado, t_acumulado, progresso = debounce_info
        if confirmado:
            cv2.putText(frame, "[PEDESTRE CONFIRMADO] -> FSM ACIONADA", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
        elif t_acumulado > 0.0:
            num_barras = int(progresso * 8)
            barra_txt = "[" + "|" * num_barras + " " * (8 - num_barras) + "]"
            cv2.putText(frame, f"PEDESTRE AGUARDANDO: {barra_txt} {t_acumulado:.1f}s / 1.5s", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "ZONA PEDESTRES: Livre (Aguardando na calçada)", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)


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

    # Carrega ROIs persistidas (Veículos e Pedestres)
    rois = carregar_rois()
    debounce_pedestre = DebouncePedestre(tempo_confirmacao=1.5)

    pedestre_manual = False
    animal_manual = False
    inicio_manual = 0
    tempo_anterior = time.time()
    janela_configurada = False

    print("=== SEMÁFORO INTELIGENTE ATIVO (DUAL-ROI & DEBOUNCE EGM) ===")
    print(f"[CONFIG] Resolução: {args.imgsz}x{args.imgsz} | Salto: {args.skip} frames | I/O Thread: Ativa")
    print(f"[CONFIG] Simulação AR: {'ATIVADA' if ar_ativo else 'DESATIVADA'} | Tecla 'V' para alternar")
    print("Pressione P para simular pedestre na faixa")
    print("Pressione A para simular animal na via")
    print("Pressione R para calibrar a ROI de VEÍCULOS (pista poligonal livre)")
    print("Pressione E para calibrar a ROI de PEDESTRES (zona de espera/calçada)")
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

        # 1. Inferência YOLO com filtragem espacial dual (Centroide pista vs Pés calçada)
        resultado = detector.processar_frame(
            frame,
            roi_veiculos_normalizada=rois["veiculos"],
            roi_pedestres_normalizada=rois["pedestres"]
        )

        # Debounce temporal de permanência (1.5 segundos)
        pedestre_zona = (resultado["pedestres"] > 0)
        ped_confirmado, t_acumulado, progresso_aguard = debounce_pedestre.atualizar(pedestre_zona, dt)

        # Botão manual (P / A)
        if (pedestre_manual or animal_manual) and (time.time() - inicio_manual > 8.0):
            pedestre_manual = False
            animal_manual = False

        pedestres_fsm = 1 if (pedestre_manual or ped_confirmado) else 0
        animais = resultado["animais"] or (1 if animal_manual else 0)

        # 2. Fusão Sensorial Híbrida: Carros Virtuais AR vs YOLO real
        if ar_ativo:
            simulador_ar.atualizar(dt, fsm.estado_atual)
            veiculos = simulador_ar.obter_contagem_veiculos_roi()
        else:
            veiculos = resultado["veiculos"]

        fsm.atualizar_sensores(veiculos=veiculos, pedestres=pedestres_fsm, animais=animais)
        fsm.tick()

        frame_anotado = resultado["frame_anotado"]

        # 3. Renderização em camadas:
        # 3.1 Pista e Carros AR
        roi_px_veic = DetectorTrafego.roi_para_pixels(rois["veiculos"], frame.shape)
        if len(roi_px_veic) >= 3:
            cv2.polylines(frame_anotado, [roi_px_veic], True, (0, 215, 255), 2)
            p_topo_v = roi_px_veic[0]
            cv2.putText(frame_anotado, "[PISTA VEICULOS]", (p_topo_v[0] + 5, max(20, p_topo_v[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)

        if ar_ativo:
            simulador_ar.renderizar(frame_anotado, roi_px_veic, fsm.estado_atual)

        # 3.2 Zonas de Espera de Pedestres (Ciano quando livre, Verde vivo se detectado/confirmado)
        rois_ped = rois.get("pedestres", [])
        if rois_ped and isinstance(rois_ped[0], list) and len(rois_ped[0]) > 0 and isinstance(rois_ped[0][0], (int, float)):
            rois_ped = [rois_ped]

        for idx_z, ped_norm in enumerate(rois_ped):
            roi_px_ped = DetectorTrafego.roi_para_pixels(ped_norm, frame.shape)
            if roi_px_ped is not None and len(roi_px_ped) >= 3:
                cor_ped = (0, 255, 0) if (ped_confirmado or pedestre_zona) else (255, 255, 0)
                overlay_ped = frame_anotado.copy()
                cv2.fillPoly(overlay_ped, [roi_px_ped], cor_ped)
                alfa_ped = 0.40 if (ped_confirmado or pedestre_zona) else 0.20
                cv2.addWeighted(overlay_ped, alfa_ped, frame_anotado, 1.0 - alfa_ped, 0, frame_anotado)
                cv2.polylines(frame_anotado, [roi_px_ped], True, cor_ped, 2)
                
                p_topo_p = roi_px_ped[0]
                label_base = f"ZONA {idx_z + 1}"
                label_ped = f"[{label_base}: CONFIRMADO]" if ped_confirmado else (f"[{label_base}: AGUARDANDO]" if pedestre_zona else f"[{label_base}: PEDESTRES]")
                cv2.putText(frame_anotado, label_ped, (p_topo_p[0] + 5, max(20, p_topo_p[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor_ped, 1, cv2.LINE_AA)

        # 3.3 HUD Semáforo com Debounce Info e Modo 3D
        desenhar_painel_semaforo(
            frame_anotado,
            fsm,
            ar_ativo=ar_ativo,
            modo_3d=simulador_ar.modo_3d,
            debounce_info=(ped_confirmado, t_acumulado, progresso_aguard)
        )

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
        elif key == ord('m') or key == ord('M'):
            modo_3d_ativo = simulador_ar.alternar_modo_3d()
            print(f"[MODO AR] Modo 3D Wireframe / Solid AR: {'LIGADO' if modo_3d_ativo else 'DESLIGADO (2D Sprites)'}")
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
            nova_roi_v = selecionar_poligono_livre(nome_janela, frame_congelado, roi_atual=rois["veiculos"], tipo="veiculos")
            fsm.inicio_estado += (time.time() - t0)
            tempo_anterior = time.time()
            if nova_roi_v is not None:
                rois["veiculos"] = nova_roi_v
                salvar_rois(rois)
        elif key == ord('e') or key == ord('E'):
            frame_congelado = frame.copy()
            t0 = time.time()
            zonas_ped = rois.get("pedestres", [])
            if zonas_ped and isinstance(zonas_ped[0], list) and len(zonas_ped[0]) > 0 and isinstance(zonas_ped[0][0], (int, float)):
                zonas_ped = [zonas_ped]
            
            overlay_sel = frame_congelado.copy()
            cv2.rectangle(overlay_sel, (0, 0), (frame_congelado.shape[1], 75), (20, 20, 20), -1)
            cv2.addWeighted(overlay_sel, 0.85, frame_congelado, 0.15, 0, frame_congelado)
            texto_menu = f"CALIBRAR PEDESTRES: [1] Zona 1 (Esq) | [2] Zona 2 (Dir) | [+] Nova Zona ({len(zonas_ped)+1}) | [ESC] Cancelar"
            cv2.putText(frame_congelado, texto_menu, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow(nome_janela, frame_congelado)
            
            sub_key = cv2.waitKey(0) & 0xFF
            idx_calibrar = None
            if sub_key == ord('1'):
                idx_calibrar = 0
            elif sub_key == ord('2'):
                idx_calibrar = 1
            elif sub_key in (ord('+'), ord('='), ord('a'), ord('A'), ord('n'), ord('N')):
                idx_calibrar = len(zonas_ped)
            elif sub_key == 27:
                idx_calibrar = None
            else:
                idx_calibrar = 0
            
            if idx_calibrar is not None:
                roi_atual = zonas_ped[idx_calibrar] if idx_calibrar < len(zonas_ped) else None
                label_nome = "ESQUERDA" if idx_calibrar == 0 else ("DIREITA" if idx_calibrar == 1 else f"ZONA {idx_calibrar+1}")
                nova_roi_p = selecionar_poligono_livre(
                    nome_janela,
                    frame.copy(),
                    roi_atual=roi_atual,
                    tipo="pedestres",
                    titulo_personalizado=f"CALIBRANDO ZONA DE PEDESTRES {idx_calibrar + 1} ({label_nome})"
                )
                if nova_roi_p is not None:
                    if idx_calibrar < len(zonas_ped):
                        zonas_ped[idx_calibrar] = nova_roi_p
                    else:
                        zonas_ped.append(nova_roi_p)
                    rois["pedestres"] = zonas_ped
                    salvar_rois(rois)
            fsm.inicio_estado += (time.time() - t0)
            tempo_anterior = time.time()

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
