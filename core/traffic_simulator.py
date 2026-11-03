import time
import random
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
    cor: tuple               # BGR (ex: (0, 165, 255))
    parado: bool = False


class SimuladorTrafegoVirtual:
    """
    Simulador de tráfego AR determinístico para pistas com perspectiva 2D.
    Gera veículos virtuais, gerencia filas anti-colisão e sincroniza com a FSM.
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

    def spawn_imediato(self) -> bool:
        """
        Força o spawn de um novo veículo na posição 0.0 se houver espaço livre.
        """
        if len(self.veiculos) >= self.max_veiculos:
            return False
        
        # Verifica se o ponto de entrada está desobstruído
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
        """
        Retorna a contagem de veículos que estão antes da linha de retenção,
        representando a fila de carros aguardando ou cruzando.
        """
        return len([v for v in self.veiculos if v.posicao <= self.stop_line])

    def atualizar(self, dt: float, fsm_state):
        """
        Atualiza a cinemática 1D determinística dos veículos.
        fsm_state pode ser uma instância de EstadoSemaforo ou string.
        """
        if dt <= 0.0:
            return

        # Identifica se a passagem para veículos está liberada
        estado_str = fsm_state.value if isinstance(fsm_state, EstadoSemaforo) else str(fsm_state)
        sinal_aberto = (estado_str == EstadoSemaforo.VERDE_CARRO.value)

        # Gerenciamento de spawn periódico
        self._tempo_desde_spawn += dt
        if self._tempo_desde_spawn >= self.taxa_spawn:
            if len(self.veiculos) < self.max_veiculos:
                self.spawn_imediato()
            self._tempo_desde_spawn = 0.0

        # Ordena veículos por posição decrescente (líder na frente)
        self.veiculos.sort(key=lambda x: x.posicao, reverse=True)

        for i, v in enumerate(self.veiculos):
            # Determina o limite máximo de avanço para este veículo
            limite_parada = 1.05  # Sem obstáculo por padrão

            # 1. Obstáculo da frente (anti-colisão na fila)
            if i > 0:
                carro_frente = self.veiculos[i - 1]
                limite_carro = carro_frente.posicao - carro_frente.comprimento - self.d_seguranca
                limite_parada = min(limite_parada, limite_carro)

            # 2. Obstáculo da Stop Line do Semáforo
            if not sinal_aberto and v.posicao <= self.stop_line:
                limite_parada = min(limite_parada, self.stop_line)

            distancia_alvo = limite_parada - v.posicao

            # Dinâmica de frenagem suave ou aceleração
            if distancia_alvo <= 0.005:
                # Já atingiu o ponto de parada
                v.posicao = min(v.posicao, limite_parada)
                v.velocidade = 0.0
                v.parado = True
            elif distancia_alvo < 0.18 and (not sinal_aberto or (i > 0 and self.veiculos[i - 1].parado)):
                # Desaceleração suave proporcional
                v.velocidade = max(0.0, v.velocidade - self.desaceleracao * dt)
                v.parado = (v.velocidade <= 0.002)
                if v.parado:
                    v.velocidade = 0.0
            else:
                # Aceleração em direção à velocidade máxima
                v.velocidade = min(v.velocidade_max, v.velocidade + self.aceleracao * dt)
                v.parado = False

            # Integração da posição
            v.posicao += v.velocidade * dt
            if v.posicao > limite_parada and limite_parada <= 1.0:
                v.posicao = limite_parada
                v.velocidade = 0.0
                v.parado = True

        # Despawn de veículos que já cruzaram todo o campo de visão
        self.veiculos = [v for v in self.veiculos if v.posicao < 1.0]

    def _interpolar_ponto_roi(self, t: float, p_esq0, p_dir0, p_dir1, p_esq1):
        """
        Interpola pontos ao longo do trapézio normalizado/em pixels.
        p_esq0: topo esquerdo, p_dir0: topo direito
        p_dir1: base direita, p_esq1: base esquerda
        """
        t = min(max(float(t), 0.0), 1.0)
        pt_esq = (1.0 - t) * np.array(p_esq0, dtype=np.float32) + t * np.array(p_esq1, dtype=np.float32)
        pt_dir = (1.0 - t) * np.array(p_dir0, dtype=np.float32) + t * np.array(p_dir1, dtype=np.float32)
        return pt_esq, pt_dir

    def renderizar(self, frame: np.ndarray, roi_pixels: np.ndarray, estado_fsm):
        """
        Renderiza a camada AR dos carros virtuais e a Stop Line projetada na ROI.
        frame: imagem BGR (modificada in-place com segurança).
        roi_pixels: 4 pontos ordenados [P0(top-left), P1(top-right), P2(bottom-right), P3(bottom-left)].
        """
        if roi_pixels is None or len(roi_pixels) != 4 or frame is None:
            return frame

        p0, p1, p2, p3 = [np.array(pt, dtype=np.float32) for pt in roi_pixels]
        
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

        # 2. Renderiza os Carros Virtuais projetados
        for v in self.veiculos:
            # Posições dianteira e traseira no trapézio
            t_frente = min(max(v.posicao, 0.0), 1.0)
            t_traseira = min(max(v.posicao - v.comprimento, 0.0), 1.0)

            pt_frente_esq, pt_frente_dir = self._interpolar_ponto_roi(t_frente, p0, p1, p2, p3)
            pt_tras_esq, pt_tras_dir = self._interpolar_ponto_roi(t_traseira, p0, p1, p2, p3)

            # Largura ajustada centralizada na pista (80% da largura da via na cota)
            margem = 0.10
            c_fe = (1 - margem) * pt_frente_esq + margem * pt_frente_dir
            c_fd = margem * pt_frente_esq + (1 - margem) * pt_frente_dir
            c_td = margem * pt_tras_esq + (1 - margem) * pt_tras_dir
            c_te = (1 - margem) * pt_tras_esq + margem * pt_tras_dir

            box_carro = np.array([c_fe, c_fd, c_td, c_te], dtype=np.int32)

            # Preenchimento semitransparente da carroceria
            overlay = frame.copy()
            cv2.fillPoly(overlay, [box_carro], v.cor)
            cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
            cv2.polylines(frame, [box_carro], isClosed=True, color=(255, 255, 255), thickness=2)

            # Faróis dianteiros (quando em movimento) ou luz de freio traseira (quando parado)
            if v.parado:
                cv2.circle(frame, (int(c_te[0]), int(c_te[1])), 4, (0, 0, 255), -1)
                cv2.circle(frame, (int(c_td[0]), int(c_td[1])), 4, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (int(c_fe[0]), int(c_fe[1])), 4, (0, 255, 255), -1)
                cv2.circle(frame, (int(c_fd[0]), int(c_fd[1])), 4, (0, 255, 255), -1)

            # Tag identificadora [AR] CARRO #id
            vel_kmh = int(v.velocidade * 250)
            status_txt = f"[AR] #{v.id} ({vel_kmh} km/h)" if not v.parado else f"[AR] #{v.id} (PARADO)"
            pos_tag = (int(c_fe[0]), max(15, int(c_fe[1]) - 6))
            
            cv2.putText(frame, status_txt, pos_tag,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return frame
