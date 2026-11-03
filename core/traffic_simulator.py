import os
import random
import math
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import cv2
from core.fsm import EstadoSemaforo

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')


@dataclass
class VeiculoVirtual:
    id: int
    posicao: float           # 0.0 (topo da ROI) a 1.0 (fim da ROI)
    velocidade: float        # unidades relativas por segundo
    velocidade_max: float    # velocidade limite desejada
    comprimento: float       # tamanho longitudinal relativo na pista
    cor: tuple               # BGR (ex: (230, 80, 50))
    parado: bool = False
    faixa: int = 0           # 0: faixa esquerda, 1: faixa direita
    tipo: str = "carro"     # 'carro', 'caminhao', 'moto'


# Alias de retrocompatibilidade
CarroVirtual = VeiculoVirtual


def gerar_sprite_veiculo(largura: int, altura: int, cor_bgr: tuple = (230, 80, 50), parado: bool = False, tipo: str = "carro") -> np.ndarray:
    """
    Gera proceduralmente um sprite BGRA realista (top-down) para múltiplos tipos:
    - 'carro': sedã típico com vidros, rodas e lanternas.
    - 'caminhao': veículo longo com cabine frontal e caçamba/carreta.
    - 'moto': silhueta estreita de moto com rodas em linha e piloto.
    """
    sprite = np.zeros((altura, largura, 4), dtype=np.uint8)
    b, g, r = cor_bgr

    if tipo == "moto":
        cor_pneu = (25, 25, 25, 255)
        # Pneus em linha
        cv2.rectangle(sprite, (largura // 2 - 2, 2), (largura // 2 + 2, 10), cor_pneu, -1)
        cv2.rectangle(sprite, (largura // 2 - 2, altura - 10), (largura // 2 + 2, altura - 2), cor_pneu, -1)
        # Corpo / chassi da moto
        pts_corpo = np.array([
            [largura // 2, 8],
            [largura // 2 + 6, altura // 2],
            [largura // 2, altura - 10],
            [largura // 2 - 6, altura // 2]
        ], dtype=np.int32)
        cv2.fillPoly(sprite, [pts_corpo], (b, g, r, 255))
        # Guidão e farol
        cv2.line(sprite, (largura // 2 - 8, 12), (largura // 2 + 8, 12), (200, 200, 200, 255), 2)
        cv2.circle(sprite, (largura // 2, 5), 3, (120, 255, 255, 255), -1)
        # Piloto com capacete
        cv2.circle(sprite, (largura // 2, altura // 2), 5, (230, 230, 230, 255), -1)
        # Lanterna de freio
        cor_lanterna = (0, 0, 255, 255) if parado else (15, 15, 180, 255)
        cv2.rectangle(sprite, (largura // 2 - 3, altura - 4), (largura // 2 + 3, altura - 1), cor_lanterna, -1)
        return sprite

    # Pneus nas 4 pontas
    w_pneu = max(3, int(largura * 0.14))
    h_pneu = max(8, int(altura * 0.16))
    cor_pneu = (25, 25, 25, 255)

    cv2.rectangle(sprite, (2, int(altura * 0.14)), (2 + w_pneu, int(altura * 0.14) + h_pneu), cor_pneu, -1)
    cv2.rectangle(sprite, (largura - 2 - w_pneu, int(altura * 0.14)), (largura - 2, int(altura * 0.14) + h_pneu), cor_pneu, -1)
    cv2.rectangle(sprite, (2, int(altura * 0.70)), (2 + w_pneu, int(altura * 0.70) + h_pneu), cor_pneu, -1)
    cv2.rectangle(sprite, (largura - 2 - w_pneu, int(altura * 0.70)), (largura - 2, int(altura * 0.70) + h_pneu), cor_pneu, -1)

    margem_x = int(largura * 0.08)
    margem_y = int(altura * 0.04)
    cor_corpo = (int(b), int(g), int(r), 255)

    if tipo == "caminhao":
        # Cabine Dianteira
        cabine_h = int(altura * 0.30)
        cv2.rectangle(sprite, (margem_x, margem_y), (largura - margem_x, margem_y + cabine_h), cor_corpo, -1)
        # Para-brisa amplo reto
        cv2.rectangle(sprite, (margem_x + 3, margem_y + int(cabine_h * 0.25)), (largura - margem_x - 3, margem_y + int(cabine_h * 0.60)), (35, 25, 20, 240), -1)
        # Junção e caçamba / baú
        cv2.line(sprite, (margem_x, margem_y + cabine_h + 3), (largura - margem_x, margem_y + cabine_h + 3), (40, 40, 40, 255), 2)
        cor_bau = (max(0, b - 20), max(0, g - 20), max(0, r - 20), 255)
        cv2.rectangle(sprite, (margem_x + 1, margem_y + cabine_h + 6), (largura - margem_x - 1, altura - margem_y), cor_bau, -1)
        cv2.rectangle(sprite, (margem_x + 1, margem_y + cabine_h + 6), (largura - margem_x - 1, altura - margem_y), (80, 80, 80, 255), 1)
    else:
        # Carroceria aerodinâmica com teto e vidros
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

        # Teto da cabine
        teto_mx = int(largura * 0.20)
        teto_y1 = int(altura * 0.27)
        teto_y2 = int(altura * 0.72)
        cor_teto = (max(0, b - 30), max(0, g - 30), max(0, r - 30), 255)
        cv2.rectangle(sprite, (teto_mx, teto_y1), (largura - teto_mx, teto_y2), cor_teto, -1)

        # Para-brisa e vidros
        cor_vidro = (35, 25, 20, 240)
        pts_para_brisa = np.array([
            [teto_mx + 2, teto_y1 - 2],
            [largura - teto_mx - 2, teto_y1 - 2],
            [largura - teto_mx - 6, teto_y1 - int(altura * 0.10)],
            [teto_mx + 6, teto_y1 - int(altura * 0.10)]
        ], dtype=np.int32)
        cv2.fillPoly(sprite, [pts_para_brisa], cor_vidro)
        cv2.rectangle(sprite, (teto_mx - 2, teto_y1 + 4), (teto_mx + 3, teto_y2 - 4), cor_vidro, -1)
        cv2.rectangle(sprite, (largura - teto_mx - 3, teto_y1 + 4), (largura - teto_mx + 2, teto_y2 - 4), cor_vidro, -1)

    # Faróis e Lanternas
    cv2.circle(sprite, (margem_x + 5, margem_y + 3), 4, (120, 255, 255, 255), -1)
    cv2.circle(sprite, (largura - margem_x - 5, margem_y + 3), 4, (120, 255, 255, 255), -1)

    cor_lanterna = (0, 0, 255, 255) if parado else (15, 15, 180, 255)
    cv2.rectangle(sprite, (margem_x + 3, altura - margem_y - 6), (margem_x + 13, altura - margem_y), cor_lanterna, -1)
    cv2.rectangle(sprite, (largura - margem_x - 13, altura - margem_y - 6), (largura - margem_x - 3, altura - margem_y), cor_lanterna, -1)

    return sprite


def gerar_sprite_carro(largura: int = 120, altura: int = 240, cor_bgr: tuple = (230, 80, 50), parado: bool = False) -> np.ndarray:
    """Alias de compatibilidade."""
    return gerar_sprite_veiculo(largura, altura, cor_bgr, parado=parado, tipo="carro")


class SimuladorTrafegoVirtual:
    """
    Simulador de tráfego AR determinístico com pista dupla (2 faixas paralelas),
    anti-colisão independente por faixa e suporte a múltiplos tipos de veículos.
    """
    def __init__(self, taxa_spawn=3.5, stop_line=0.55, max_veiculos=6):
        self.taxa_spawn = float(taxa_spawn)
        self.stop_line = float(stop_line)
        self.max_veiculos = int(max_veiculos)
        
        self.veiculos: list[VeiculoVirtual] = []
        self._proximo_id = 1
        self._tempo_desde_spawn = 0.0
        self._proxima_faixa = 0
        
        # Parâmetros de cinemática
        self.d_seguranca = 0.06
        self.aceleracao = 0.22
        self.desaceleracao = 0.42
        self.cores_paleta = [
            (230, 80, 50),   # Azul esportivo
            (50, 200, 240),  # Amarelo urbano
            (60, 60, 230),   # Vermelho sedã
            (180, 180, 180), # Prata executivo
            (80, 220, 120),  # Verde elétrico
        ]
        self._sprite_cache = {}
        self._garantir_asset_padrao()

    def _garantir_asset_padrao(self):
        caminho_asset = os.path.join(ASSETS_DIR, 'car_top.png')
        if not os.path.exists(caminho_asset):
            os.makedirs(os.path.dirname(caminho_asset), exist_ok=True)
            sp = gerar_sprite_carro(120, 240, cor_bgr=(230, 80, 50), parado=False)
            cv2.imwrite(caminho_asset, sp)

    def _carregar_asset_se_existir(self, nome_arquivo: str) -> Optional[np.ndarray]:
        caminho = os.path.join(ASSETS_DIR, nome_arquivo)
        if os.path.exists(caminho):
            img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                return img
        return None

    def _obter_sprite(self, cor: tuple, parado: bool, tipo: str = "carro") -> np.ndarray:
        chave = (cor, parado, tipo)
        if chave not in self._sprite_cache:
            # Mapeia tipos para os novos assets PNG entregues em assets/
            mapa_assets = {
                "carro": ["car_sedan.png", "car_hatch.png", "car_sports.png", "taxi.png"],
                "caminhao": ["truck.png", "bus.png", "truck_long.png"],
                "moto": ["motorcycle.png"],
                "van": ["van.png"],
                "servico": ["police.png", "ambulance.png"]
            }
            
            sprite_carregado = None
            arquivos = mapa_assets.get(tipo, mapa_assets["carro"])
            # Escolhe deterministicamente pelo hash da cor
            arquivo_alvo = arquivos[abs(hash(cor)) % len(arquivos)]
            sprite_carregado = self._carregar_asset_se_existir(arquivo_alvo)

            if sprite_carregado is not None:
                self._sprite_cache[chave] = sprite_carregado
            else:
                # Fallback procedural se o arquivo não existir
                if tipo == "caminhao":
                    self._sprite_cache[chave] = gerar_sprite_veiculo(130, 310, cor_bgr=cor, parado=parado, tipo="caminhao")
                elif tipo == "moto":
                    self._sprite_cache[chave] = gerar_sprite_veiculo(50, 120, cor_bgr=cor, parado=parado, tipo="moto")
                else:
                    self._sprite_cache[chave] = gerar_sprite_veiculo(120, 240, cor_bgr=cor, parado=parado, tipo="carro")
        return self._sprite_cache[chave]

    def spawn_veiculo(self, faixa: int = None, tipo: str = None) -> bool:
        """Gera um veículo especificando ou alternando faixa e tipo."""
        if len(self.veiculos) >= self.max_veiculos:
            return False

        if faixa is None:
            faixa = self._proxima_faixa
            self._proxima_faixa = 1 - self._proxima_faixa

        if tipo is None:
            tipo = random.choices(["carro", "caminhao", "moto"], weights=[0.65, 0.20, 0.15])[0]

        # Checa espaço livre apenas na mesma faixa
        veiculos_faixa = [v for v in self.veiculos if v.faixa == faixa]
        for v in veiculos_faixa:
            if v.posicao < (self.d_seguranca + 0.08):
                return False

        if tipo == "caminhao":
            comp = 0.14
            v_max = 0.16 + random.uniform(0.0, 0.03)
        elif tipo == "moto":
            comp = 0.06
            v_max = 0.24 + random.uniform(0.0, 0.05)
        else:
            comp = 0.09
            v_max = 0.19 + random.uniform(0.0, 0.04)

        novo = VeiculoVirtual(
            id=self._proximo_id,
            posicao=0.0,
            velocidade=0.10,
            velocidade_max=v_max,
            comprimento=comp,
            cor=self.cores_paleta[(self._proximo_id - 1) % len(self.cores_paleta)],
            parado=False,
            faixa=faixa,
            tipo=tipo
        )
        self._proximo_id += 1
        self.veiculos.append(novo)
        self._tempo_desde_spawn = 0.0
        return True

    def spawn_imediato(self, faixa: int = None) -> bool:
        """Força spawn imediato."""
        return self.spawn_veiculo(faixa=faixa)

    def limpar_pista(self):
        """Remove todos os veículos da via."""
        self.veiculos.clear()

    def obter_contagem_veiculos_roi(self) -> int:
        """Retorna veículos ativos antes da stop line."""
        return len([v for v in self.veiculos if v.posicao <= self.stop_line])

    def atualizar(self, dt: float, fsm_state=None, estado_semaforo=None):
        """
        Atualiza cinemática 1D com anti-colisão estritamente independente por faixa.
        Aceita fsm_state ou estado_semaforo para retrocompatibilidade total.
        """
        if dt <= 0.0:
            return

        estado_usado = fsm_state if fsm_state is not None else estado_semaforo
        estado_str = estado_usado.value if isinstance(estado_usado, EstadoSemaforo) else str(estado_usado)
        sinal_aberto = (estado_str == EstadoSemaforo.VERDE_CARRO.value)

        self._tempo_desde_spawn += dt
        if self._tempo_desde_spawn >= self.taxa_spawn:
            if len(self.veiculos) < self.max_veiculos:
                self.spawn_veiculo()
            self._tempo_desde_spawn = 0.0

        # Processa anti-colisão e movimento de forma independente por faixa
        for faixa_id in (0, 1):
            veiculos_faixa = [v for v in self.veiculos if v.faixa == faixa_id]
            veiculos_faixa.sort(key=lambda x: x.posicao, reverse=True)

            for i, v in enumerate(veiculos_faixa):
                limite_parada = 1.05

                # Anti-colisão na fila apenas da mesma faixa
                if i > 0:
                    lider = veiculos_faixa[i - 1]
                    limite_carro = lider.posicao - lider.comprimento - self.d_seguranca
                    limite_parada = min(limite_parada, limite_carro)

                # Parada na Stop Line
                if not sinal_aberto and v.posicao <= self.stop_line:
                    limite_parada = min(limite_parada, self.stop_line)

                distancia_alvo = limite_parada - v.posicao

                if distancia_alvo <= 0.005:
                    v.posicao = min(v.posicao, limite_parada)
                    v.velocidade = 0.0
                    v.parado = True
                elif distancia_alvo < 0.18 and (not sinal_aberto or (i > 0 and veiculos_faixa[i - 1].parado)):
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
        if len(roi_pixels) == 4:
            return [np.array(pt, dtype=np.float32) for pt in roi_pixels]
        
        pts = np.array(roi_pixels, dtype=np.float32)
        pts_y = sorted(pts, key=lambda p: p[1])
        metade = max(1, len(pts) // 2)
        
        top_pts = sorted(pts_y[:metade], key=lambda p: p[0])
        p0 = top_pts[0]
        p1 = top_pts[-1] if len(top_pts) > 1 else top_pts[0]
        
        bot_pts = sorted(pts_y[metade:], key=lambda p: p[0])
        p3 = bot_pts[0]
        p2 = bot_pts[-1] if len(bot_pts) > 1 else bot_pts[0]
        
        return [p0, p1, p2, p3]

    def _interpolar_ponto_roi(self, t: float, p_esq0, p_dir0, p_dir1, p_esq1):
        t = min(max(float(t), 0.0), 1.0)
        pt_esq = (1.0 - t) * np.array(p_esq0, dtype=np.float32) + t * np.array(p_esq1, dtype=np.float32)
        pt_dir = (1.0 - t) * np.array(p_dir0, dtype=np.float32) + t * np.array(p_dir1, dtype=np.float32)
        return pt_esq, pt_dir

    def renderizar(self, frame: np.ndarray, roi_pixels: np.ndarray, fsm_state=None, estado_fsm=None, estado_semaforo=None):
        """
        Renderiza a camada AR com suporte a pista dupla (2 faixas), linha divisória central,
        aspect ratio rígido (sem achatamento) e sombras projetadas.
        """
        if roi_pixels is None or len(roi_pixels) < 3 or frame is None:
            return frame

        estado_usado = fsm_state if fsm_state is not None else (estado_fsm if estado_fsm is not None else estado_semaforo)
        estado_str = estado_usado.value if isinstance(estado_usado, EstadoSemaforo) else str(estado_usado)
        sinal_aberto = (estado_str == EstadoSemaforo.VERDE_CARRO.value)

        p0, p1, p2, p3 = self._obter_4_vertices_referencia(roi_pixels)
        h_frame, w_frame = frame.shape[:2]

        # 1. Desenha Linha Divisória Central Tracejada entre as duas faixas
        for t in np.linspace(0.05, 0.95, 10):
            pt_e, pt_d = self._interpolar_ponto_roi(t, p0, p1, p2, p3)
            p_c_ini = 0.5 * pt_e + 0.5 * pt_d
            
            t_fim = min(1.0, t + 0.04)
            pt_e2, pt_d2 = self._interpolar_ponto_roi(t_fim, p0, p1, p2, p3)
            p_c_fim = 0.5 * pt_e2 + 0.5 * pt_d2
            cv2.line(frame, (int(p_c_ini[0]), int(p_c_ini[1])), (int(p_c_fim[0]), int(p_c_fim[1])), (220, 220, 220), 1, cv2.LINE_AA)

        # 2. Stop Line AR
        pt_stop_esq, pt_stop_dir = self._interpolar_ponto_roi(self.stop_line, p0, p1, p2, p3)
        cor_stop = (0, 230, 0) if sinal_aberto else (0, 0, 255)
        pt_stop_esq_i = (int(pt_stop_esq[0]), int(pt_stop_esq[1]))
        pt_stop_dir_i = (int(pt_stop_dir[0]), int(pt_stop_dir[1]))
        cv2.line(frame, pt_stop_esq_i, pt_stop_dir_i, cor_stop, 3)

        # 3. Renderiza Veículos ordenados por profundidade
        for v in sorted(self.veiculos, key=lambda x: x.posicao):
            t = min(max(float(v.posicao), 0.0), 1.0)
            pt_esq, pt_dir = self._interpolar_ponto_roi(t, p0, p1, p2, p3)
            
            # Posição lateral por faixa (Faixa 0: 25% da via, Faixa 1: 75% da via)
            fator_faixa = 0.28 if v.faixa == 0 else 0.72
            p_carro = (1.0 - fator_faixa) * pt_esq + fator_faixa * pt_dir
            cx, cy = int(p_carro[0]), int(p_carro[1])
            
            largura_pista_secao = np.linalg.norm(pt_dir - pt_esq)
            escala_profundidade = 0.40 + 0.60 * t
            
            # Obtém o sprite e seu Aspect Ratio original (sem achatamento)
            sprite_base = self._obter_sprite(v.cor, v.parado, v.tipo)
            orig_h, orig_w = sprite_base.shape[:2]
            ar = orig_w / float(orig_h)
            
            # Define dimensões mantendo estritamente a proporção
            larg_fator = 0.32 if v.tipo == "caminhao" else (0.16 if v.tipo == "moto" else 0.24)
            w_carro = max(14, int(largura_pista_secao * larg_fator * escala_profundidade))
            h_carro = int(w_carro / ar)
            
            # Sombra Projetada elíptica proporcional
            sombra_centro = (cx, cy + int(3 * escala_profundidade))
            sombra_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
            cv2.ellipse(sombra_mask, sombra_centro, (int(w_carro * 0.55), int(h_carro * 0.50)), 0, 0, 360, 255, -1)
            indices_sombra = sombra_mask > 0
            frame[indices_sombra] = (frame[indices_sombra].astype(np.float32) * 0.55).astype(np.uint8)

            # Redimensionamento do sprite
            sprite_res = cv2.resize(sprite_base, (w_carro, h_carro), interpolation=cv2.INTER_AREA)
            
            x1 = cx - w_carro // 2
            y1 = cy - h_carro // 2
            x2 = x1 + w_carro
            y2 = y1 + h_carro
            
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w_frame, x2), min(h_frame, y2)
            w_vis = x2_c - x1_c
            h_vis = y2_c - y1_c
            
            if w_vis > 0 and h_vis > 0:
                sx1 = x1_c - x1
                sy1 = y1_c - y1
                sx2 = sx1 + w_vis
                sy2 = sy1 + h_vis
                
                sprite_recorte = sprite_res[sy1:sy2, sx1:sx2]
                alfa = (sprite_recorte[:, :, 3] / 255.0)[:, :, np.newaxis]
                cor_rgb = sprite_recorte[:, :, :3]
                
                frame[y1_c:y2_c, x1_c:x2_c] = (
                    cor_rgb * alfa + frame[y1_c:y2_c, x1_c:x2_c] * (1.0 - alfa)
                ).astype(np.uint8)

            # Tag identificadora discreta
            tag_txt = f"{v.tipo[:3].upper()} F{v.faixa}"
            cv2.putText(frame, tag_txt, (cx - 14, max(14, cy - h_carro // 2 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
