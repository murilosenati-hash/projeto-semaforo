import os
import random
import math
from dataclasses import dataclass
import numpy as np
import cv2
from core.fsm import EstadoSemaforo


@dataclass
class VeiculoVirtual:
    id: int
    posicao: float           # 0.0 (topo da ROI) a 1.0 (fim da ROI)
    velocidade: float        # unidades relativas por segundo
    velocidade_max: float    # velocidade limite desejada
    comprimento: float       # tamanho longitudinal relativo na pista
    cor: tuple               # BGR (ex: (230, 80, 50))
    parado: bool = False


def gerar_sprite_carro(largura: int = 120, altura: int = 240, cor_bgr: tuple = (230, 80, 50), parado: bool = False) -> np.ndarray:
    """
    Gera proceduralmente um sprite RGBA realista de alta fidelidade visual (visão superior)
    com carroceria estilizada, vidros escuros com reflexo, teto, 4 rodas e faróis/luzes de freio.
    """
    sprite = np.zeros((altura, largura, 4), dtype=np.uint8)
    
    # 1. Rodas / Pneus pretos nas 4 pontas
    w_pneu = max(4, int(largura * 0.15))
    h_pneu = max(8, int(altura * 0.17))
    cor_pneu = (25, 25, 25, 255)
    
    # Dianteiras
    cv2.rectangle(sprite, (4, int(altura * 0.14)), (4 + w_pneu, int(altura * 0.14) + h_pneu), cor_pneu, -1)
    cv2.rectangle(sprite, (largura - 4 - w_pneu, int(altura * 0.14)), (largura - 4, int(altura * 0.14) + h_pneu), cor_pneu, -1)
    # Traseiras
    cv2.rectangle(sprite, (4, int(altura * 0.69)), (4 + w_pneu, int(altura * 0.69) + h_pneu), cor_pneu, -1)
    cv2.rectangle(sprite, (largura - 4 - w_pneu, int(altura * 0.69)), (largura - 4, int(altura * 0.69) + h_pneu), cor_pneu, -1)

    # 2. Carroceria aerodinâmica (chassi principal)
    margem_x = int(largura * 0.10)
    margem_y = int(altura * 0.05)
    cor_corpo = (int(cor_bgr[0]), int(cor_bgr[1]), int(cor_bgr[2]), 255)
    
    pts_corpo = np.array([
        [margem_x + 8, margem_y],
        [largura - margem_x - 8, margem_y],
        [largura - margem_x, margem_y + 14],
        [largura - margem_x, altura - margem_y - 14],
        [largura - margem_x - 8, altura - margem_y],
        [margem_x + 8, altura - margem_y],
        [margem_x, altura - margem_y - 14],
        [margem_x, margem_y + 14]
    ], dtype=np.int32)
    cv2.fillPoly(sprite, [pts_corpo], cor_corpo)
    cv2.polylines(sprite, [pts_corpo], isClosed=True, color=(255, 255, 255, 160), thickness=2)

    # 3. Teto da cabine (painel central com tom coordenado)
    teto_mx = int(largura * 0.20)
    teto_y1 = int(altura * 0.27)
    teto_y2 = int(altura * 0.72)
    cor_teto = (
        max(0, cor_bgr[0] - 30),
        max(0, cor_bgr[1] - 30),
        max(0, cor_bgr[2] - 30),
        255
    )
    cv2.rectangle(sprite, (teto_mx, teto_y1), (largura - teto_mx, teto_y2), cor_teto, -1)

    # 4. Vidros escuros fumê (para-brisa dianteiro, traseiro e laterais)
    cor_vidro = (35, 25, 20, 240)
    
    # Para-brisa Dianteiro inclinado
    pts_para_brisa = np.array([
        [teto_mx + 2, teto_y1 - 2],
        [largura - teto_mx - 2, teto_y1 - 2],
        [largura - teto_mx - 6, teto_y1 - int(altura * 0.10)],
        [teto_mx + 6, teto_y1 - int(altura * 0.10)]
    ], dtype=np.int32)
    cv2.fillPoly(sprite, [pts_para_brisa], cor_vidro)
    # Reflexo suave de luz no vidro
    cv2.line(sprite, (teto_mx + 10, teto_y1 - 4), (largura - teto_mx - 12, teto_y1 - int(altura * 0.08)), (220, 220, 220, 130), 2)

    # Vidro Traseiro
    pts_vidro_tras = np.array([
        [teto_mx + 2, teto_y2 + 2],
        [largura - teto_mx - 2, teto_y2 + 2],
        [largura - teto_mx - 5, teto_y2 + int(altura * 0.07)],
        [teto_mx + 5, teto_y2 + int(altura * 0.07)]
    ], dtype=np.int32)
    cv2.fillPoly(sprite, [pts_para_brisa := pts_vidro_tras], cor_vidro)

    # Vidros Laterais
    cv2.rectangle(sprite, (teto_mx - 2, teto_y1 + 4), (teto_mx + 3, teto_y2 - 4), cor_vidro, -1)
    cv2.rectangle(sprite, (largura - teto_mx - 3, teto_y1 + 4), (largura - teto_mx + 2, teto_y2 - 4), cor_vidro, -1)

    # 5. Faróis Dianteiros e Lanternas Traseiras
    # Faróis LED dianteiros
    cv2.circle(sprite, (margem_x + 6, margem_y + 4), 5, (120, 255, 255, 255), -1)
    cv2.circle(sprite, (largura - margem_x - 6, margem_y + 4), 5, (120, 255, 255, 255), -1)
    
    # Lanternas Traseiras (Freio vermelho vivo se parado)
    cor_lanterna = (0, 0, 255, 255) if parado else (15, 15, 180, 255)
    cv2.rectangle(sprite, (margem_x + 4, altura - margem_y - 6), (margem_x + 14, altura - margem_y), cor_lanterna, -1)
    cv2.rectangle(sprite, (largura - margem_x - 14, altura - margem_y - 6), (largura - margem_x - 4, altura - margem_y), cor_lanterna, -1)

    return sprite


class SimuladorTrafegoVirtual:
    """
    Simulador de tráfego AR determinístico com realismo visual:
    - Cinemática 1D com anti-colisão e parada suave na Stop Line.
    - Sprites com canal alfa, textura e luzes.
    - Escala proporcional à profundidade ao longo da perspectiva da via.
    - Sombra projetada em elipse semi-transparente.
    - Suporte a polígonos de calibração de N vértices (N >= 3).
    """
    def __init__(self, taxa_spawn=3.5, stop_line=0.55, max_veiculos=5):
        self.taxa_spawn = float(taxa_spawn)
        self.stop_line = float(stop_line)
        self.max_veiculos = int(max_veiculos)
        
        self.veiculos: list[VeiculoVirtual] = []
        self._proximo_id = 1
        self._tempo_desde_spawn = 0.0
        
        # Parâmetros de cinemática
        self.d_seguranca = 0.07
        self.aceleracao = 0.20
        self.desaceleracao = 0.40
        self.cores_paleta = [
            (230, 80, 50),   # Azul esportivo
            (50, 200, 240),  # Amarelo urbano
            (60, 60, 230),   # Vermelho sedã
            (180, 180, 180), # Prata executivo
            (80, 220, 120),  # Verde elétrico
        ]
        self._sprite_cache = {}

        # Garante a existência do asset básico assets/car_top.png
        self._garantir_asset_padrao()

    def _garantir_asset_padrao(self):
        caminho_asset = os.path.join(os.path.dirname(__file__), '..', 'assets', 'car_top.png')
        if not os.path.exists(caminho_asset):
            os.makedirs(os.path.dirname(caminho_asset), exist_ok=True)
            sp = gerar_sprite_carro(120, 240, cor_bgr=(230, 80, 50), parado=False)
            cv2.imwrite(caminho_asset, sp)

    def _obter_sprite(self, cor: tuple, parado: bool) -> np.ndarray:
        chave = (cor, parado)
        if chave not in self._sprite_cache:
            self._sprite_cache[chave] = gerar_sprite_carro(120, 240, cor_bgr=cor, parado=parado)
        return self._sprite_cache[chave]

    def spawn_imediato(self) -> bool:
        """Força o spawn de um novo veículo na posição 0.0 se houver espaço livre."""
        if len(self.veiculos) >= self.max_veiculos:
            return False
        
        for v in self.veiculos:
            if v.posicao < (self.d_seguranca + 0.06):
                return False
        
        novo_veiculo = VeiculoVirtual(
            id=self._proximo_id,
            posicao=0.0,
            velocidade=0.10 + random.uniform(0.0, 0.05),
            velocidade_max=0.18 + random.uniform(0.0, 0.06),
            comprimento=0.09,
            cor=self.cores_paleta[(self._proximo_id - 1) % len(self.cores_paleta)],
            parado=False
        )
        self._proximo_id += 1
        self.veiculos.append(novo_veiculo)
        self._tempo_desde_spawn = 0.0
        return True

    def limpar_pista(self):
        """Remove todos os veículos ativos da pista."""
        self.veiculos.clear()

    def obter_contagem_veiculos_roi(self) -> int:
        """Retorna a contagem de veículos antes da stop line."""
        return len([v for v in self.veiculos if v.posicao <= self.stop_line])

    def atualizar(self, dt: float, fsm_state):
        """Atualiza cinemática 1D determinística e anti-colisão."""
        if dt <= 0.0:
            return

        estado_str = fsm_state.value if isinstance(fsm_state, EstadoSemaforo) else str(fsm_state)
        sinal_aberto = (estado_str == EstadoSemaforo.VERDE_CARRO.value)

        self._tempo_desde_spawn += dt
        if self._tempo_desde_spawn >= self.taxa_spawn:
            if len(self.veiculos) < self.max_veiculos:
                self.spawn_imediato()
            self._tempo_desde_spawn = 0.0

        self.veiculos.sort(key=lambda x: x.posicao, reverse=True)

        for i, v in enumerate(self.veiculos):
            limite_parada = 1.05

            # Anti-colisão na fila
            if i > 0:
                carro_frente = self.veiculos[i - 1]
                limite_carro = carro_frente.posicao - carro_frente.comprimento - self.d_seguranca
                limite_parada = min(limite_parada, limite_carro)

            # Parada antes da Stop Line no sinal fechado/amarelo
            if not sinal_aberto and v.posicao <= self.stop_line:
                limite_parada = min(limite_parada, self.stop_line)

            distancia_alvo = limite_parada - v.posicao

            if distancia_alvo <= 0.005:
                v.posicao = min(v.posicao, limite_parada)
                v.velocidade = 0.0
                v.parado = True
            elif distancia_alvo < 0.18 and (not sinal_aberto or (i > 0 and self.veiculos[i - 1].parado)):
                v.velocidade = max(0.0, v.velocidade - self.desaceleracao * dt)
                v.parado = (v.velocidade <= 0.002)
                if v.parado:
                    v.velocidade = 0.0
            else:
                v.velocidade = min(v.velocidade_max, v.velocidade + self.aceleracao * dt)
                v.parado = False

            v.posicao += v.velocidade * dt
            if v.posicao > limite_parada and limite_parada <= 1.0:
                v.posicao = limite_parada
                v.velocidade = 0.0
                v.parado = True

        self.veiculos = [v for v in self.veiculos if v.posicao < 1.0]

    def _obter_4_vertices_referencia(self, roi_pixels: np.ndarray):
        """
        Extrai 4 vértices de referência [P0(top-esq), P1(top-dir), P2(base-dir), P3(base-esq)]
        a partir de qualquer polígono de N vértices (N >= 3) para orientar a perspectiva da pista.
        """
        if len(roi_pixels) == 4:
            return [np.array(pt, dtype=np.float32) for pt in roi_pixels]
        
        pts = np.array(roi_pixels, dtype=np.float32)
        pts_y = sorted(pts, key=lambda p: p[1])
        metade = max(1, len(pts) // 2)
        
        # Topo: vértices superiores (menor Y)
        top_pts = sorted(pts_y[:metade], key=lambda p: p[0])
        p0 = top_pts[0]
        p1 = top_pts[-1] if len(top_pts) > 1 else top_pts[0]
        
        # Base: vértices inferiores (maior Y)
        bot_pts = sorted(pts_y[metade:], key=lambda p: p[0])
        p3 = bot_pts[0]
        p2 = bot_pts[-1] if len(bot_pts) > 1 else bot_pts[0]
        
        return [p0, p1, p2, p3]

    def _interpolar_ponto_roi(self, t: float, p_esq0, p_dir0, p_dir1, p_esq1):
        t = min(max(float(t), 0.0), 1.0)
        pt_esq = (1.0 - t) * np.array(p_esq0, dtype=np.float32) + t * np.array(p_esq1, dtype=np.float32)
        pt_dir = (1.0 - t) * np.array(p_dir0, dtype=np.float32) + t * np.array(p_dir1, dtype=np.float32)
        return pt_esq, pt_dir

    def renderizar(self, frame: np.ndarray, roi_pixels: np.ndarray, estado_fsm):
        """
        Renderiza a camada AR com realismo visual:
        - Stop Line AR projetada no chão.
        - Sombra elíptica translúcida projetada sob cada veículo.
        - Sprite de carro em perspectiva com escala de profundidade e canal alfa.
        - Faróis e luzes de freio dinâmicas.
        - Suporta polígonos livres de N vértices (N >= 3).
        """
        if roi_pixels is None or len(roi_pixels) < 3 or frame is None:
            return frame

        p0, p1, p2, p3 = self._obter_4_vertices_referencia(roi_pixels)
        
        estado_str = estado_fsm.value if isinstance(estado_fsm, EstadoSemaforo) else str(estado_fsm)
        sinal_aberto = (estado_str == EstadoSemaforo.VERDE_CARRO.value)

        # 1. Renderiza Linha de Retenção (Stop Line AR)
        pt_stop_esq, pt_stop_dir = self._interpolar_ponto_roi(self.stop_line, p0, p1, p2, p3)
        cor_stop = (0, 230, 0) if sinal_aberto else (0, 0, 255)
        pt_stop_esq_i = (int(pt_stop_esq[0]), int(pt_stop_esq[1]))
        pt_stop_dir_i = (int(pt_stop_dir[0]), int(pt_stop_dir[1]))
        
        cv2.line(frame, pt_stop_esq_i, pt_stop_dir_i, cor_stop, 3)
        cv2.putText(frame, "[STOP LINE AR]", (pt_stop_esq_i[0] + 5, pt_stop_esq_i[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_stop, 1, cv2.LINE_AA)

        # 2. Renderiza os Carros Virtuais com Perspectiva e Sprites
        h_frame, w_frame = frame.shape[:2]

        for v in self.veiculos:
            t_frente = min(max(v.posicao, 0.0), 1.0)
            
            # Fator de escala proporcional à profundidade (menor no topo, maior na base)
            escala_profundidade = 0.45 + 0.55 * t_frente
            comprimento_escalado = v.comprimento * escala_profundidade
            t_traseira = min(max(v.posicao - comprimento_escalado, 0.0), 1.0)

            pt_frente_esq, pt_frente_dir = self._interpolar_ponto_roi(t_frente, p0, p1, p2, p3)
            pt_tras_esq, pt_tras_dir = self._interpolar_ponto_roi(t_traseira, p0, p1, p2, p3)

            # Largura proporcional da via (margem de 10% nas bordas da pista)
            margem = 0.10
            c_fe = (1 - margem) * pt_frente_esq + margem * pt_frente_dir
            c_fd = margem * pt_frente_esq + (1 - margem) * pt_frente_dir
            c_td = margem * pt_tras_esq + (1 - margem) * pt_tras_dir
            c_te = (1 - margem) * pt_tras_esq + margem * pt_tras_dir

            dst_pts = np.array([c_fe, c_fd, c_td, c_te], dtype=np.float32)
            
            # Centro e dimensões do veículo
            centro_x = int(np.mean(dst_pts[:, 0]))
            centro_y = int(np.mean(dst_pts[:, 1]))
            largura_carro = int(np.linalg.norm(c_fd - c_fe))
            comprimento_carro = int(np.linalg.norm(c_fe - c_te))

            if largura_carro <= 2 or comprimento_carro <= 2:
                continue

            # 2.1 Sombra projetada em elipse semi-transparente (cv2.addWeighted)
            overlay_sombra = frame.copy()
            eixo_x = max(2, int(largura_carro * 0.60))
            eixo_y = max(2, int(comprimento_carro * 0.58))
            cv2.ellipse(overlay_sombra, (centro_x, centro_y + int(4 * escala_profundidade)),
                        (eixo_x, eixo_y), 0, 0, 360, (15, 15, 15), -1)
            cv2.addWeighted(overlay_sombra, 0.40, frame, 0.60, 0, frame)

            # 2.2 Projeção do Sprite com Textura e Canal Alfa
            sprite = self._obter_sprite(v.cor, v.parado)
            H_s, W_s = sprite.shape[:2]
            src_pts = np.array([[0, 0], [W_s, 0], [W_s, H_s], [0, H_s]], dtype=np.float32)

            min_x = max(0, int(np.min(dst_pts[:, 0])) - 1)
            max_x = min(w_frame, int(np.max(dst_pts[:, 0])) + 1)
            min_y = max(0, int(np.min(dst_pts[:, 1])) - 1)
            max_y = min(h_frame, int(np.max(dst_pts[:, 1])) + 1)

            w_roi = max_x - min_x
            h_roi = max_y - min_y

            if w_roi > 2 and h_roi > 2:
                local_dst = dst_pts.copy()
                local_dst[:, 0] -= min_x
                local_dst[:, 1] -= min_y

                M = cv2.getPerspectiveTransform(src_pts, local_dst)
                warped = cv2.warpPerspective(
                    sprite, M, (w_roi, h_roi),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

                # Blending pelo canal alfa (transparência suave)
                alpha = (warped[:, :, 3] / 255.0)[:, :, np.newaxis]
                bgr_warped = warped[:, :, :3]
                roi_frame = frame[min_y:max_y, min_x:max_x]
                
                frame[min_y:max_y, min_x:max_x] = np.clip(
                    alpha * bgr_warped + (1.0 - alpha) * roi_frame, 0, 255
                ).astype(np.uint8)

            # 2.3 Tag identificadora do Carro AR
            vel_kmh = int(v.velocidade * 250)
            status_txt = f"[AR] #{v.id} ({vel_kmh} km/h)" if not v.parado else f"[AR] #{v.id} (PARADO)"
            pos_tag = (int(c_fe[0]), max(15, int(c_fe[1]) - 6))
            
            cv2.putText(frame, status_txt, pos_tag,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40 * max(0.8, escala_profundidade), (0, 255, 255), 1, cv2.LINE_AA)

        return frame
