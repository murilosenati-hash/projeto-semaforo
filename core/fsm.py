import time
from enum import Enum

class EstadoSemaforo(Enum):
    VERDE_CARRO = "VERDE_CARRO"
    AMARELO_CARRO = "AMARELO_CARRO"
    VERMELHO_LIMPEZA = "VERMELHO_LIMPEZA"
    VERDE_PEDESTRE = "VERDE_PEDESTRE"
    VERMELHO_PISCANDO_PEDESTRE = "VERMELHO_PISCANDO_PEDESTRE"
    STANDBY_AMARELO = "STANDBY_AMARELO"
    FAIL_SAFE = "FAIL_SAFE"

class ControladorSemaforoFSM:
    """
    Máquina de Estados Finitos (FSM) para controle semafórico digital.
    Atende aos requisitos de Sistemas Digitais (temporizadores, estados seguros)
    e Sistemas Distribuídos (adaptação a eventos de sensores remotos).
    """
    def __init__(self,
                 min_verde_carro=5.0,
                 max_verde_carro=30.0,
                 tempo_amarelo=3.0,
                 tempo_limpeza=2.0,
                 tempo_verde_pedestre=10.0,
                 tempo_pisca_pedestre=4.0):
        self.min_verde_carro = min_verde_carro
        self.max_verde_carro = max_verde_carro
        self.tempo_amarelo = tempo_amarelo
        self.tempo_limpeza = tempo_limpeza
        self.tempo_verde_pedestre = tempo_verde_pedestre
        self.tempo_pisca_pedestre = tempo_pisca_pedestre

        self.estado_atual = EstadoSemaforo.VERDE_CARRO
        self.inicio_estado = time.time()
        self.tempo_verde_calculado = min_verde_carro

        # Telemetria acumulada dos sensores
        self.veiculos_detectados = 0
        self.pedestres_detectados = 0
        self.animais_detectados = 0
        self.ultimo_heartbeat = time.time()
        self.modo_failsafe = False

    def atualizar_sensores(self, veiculos, pedestres, animais=0):
        """Atualiza a leitura dos nós de visão distribuídos."""
        self.veiculos_detectados = veiculos
        self.pedestres_detectados = pedestres
        self.animais_detectados = animais
        self.ultimo_heartbeat = time.time()
        self.modo_failsafe = False

    def calcular_verde_adaptativo(self):
        """Ajusta a duração do sinal verde proporcionalmente ao fluxo de veículos."""
        if self.veiculos_detectados == 0:
            return self.min_verde_carro
        elif self.veiculos_detectados <= 3:
            return 8.0
        elif self.veiculos_detectados <= 8:
            return 15.0
        elif self.veiculos_detectados <= 15:
            return 22.0
        else:
            return self.max_verde_carro

    def tick(self):
        """Executa um passo da máquina de estados."""
        agora = time.time()
        tempo_no_estado = agora - self.inicio_estado

        # Verificador de Timeout Distribuído (Fail-Safe)
        if agora - self.ultimo_heartbeat > 8.0 and not self.modo_failsafe:
            self.modo_failsafe = True
            self._mudar_estado(EstadoSemaforo.FAIL_SAFE)
            return

        # Lógica por estado:
        if self.estado_atual == EstadoSemaforo.FAIL_SAFE:
            # Em modo de emergência, roda temporizador fixo seguro
            if not self.modo_failsafe:
                self._mudar_estado(EstadoSemaforo.VERMELHO_LIMPEZA)
            elif tempo_no_estado > 15.0:
                self.inicio_estado = agora

        elif self.estado_atual == EstadoSemaforo.STANDBY_AMARELO:
            # Se surgir veículo ou pedestre, acorda imediatamente
            if self.veiculos_detectados > 0 or self.pedestres_detectados > 0 or self.animais_detectados > 0:
                self._mudar_estado(EstadoSemaforo.VERMELHO_LIMPEZA)

        elif self.estado_atual == EstadoSemaforo.VERDE_CARRO:
            self.tempo_verde_calculado = self.calcular_verde_adaptativo()

            # Prioridade de Segurança: Animal na pista fecha para veículos
            if self.animais_detectados > 0 and tempo_no_estado >= self.min_verde_carro:
                self._mudar_estado(EstadoSemaforo.AMARELO_CARRO)
                return

            # Rua totalmente deserta por mais de 10s: modo economia / standby
            if self.veiculos_detectados == 0 and self.pedestres_detectados == 0 and tempo_no_estado > 12.0:
                self._mudar_estado(EstadoSemaforo.STANDBY_AMARELO)
                return

            # Se houver pedestre aguardando e cumprido o verde mínimo:
            if self.pedestres_detectados > 0 and tempo_no_estado >= self.min_verde_carro:
                self._mudar_estado(EstadoSemaforo.AMARELO_CARRO)
                return

            # Tempo verde máximo atingido
            if tempo_no_estado >= self.tempo_verde_calculado:
                self._mudar_estado(EstadoSemaforo.AMARELO_CARRO)

        elif self.estado_atual == EstadoSemaforo.AMARELO_CARRO:
            if tempo_no_estado >= self.tempo_amarelo:
                self._mudar_estado(EstadoSemaforo.VERMELHO_LIMPEZA)

        elif self.estado_atual == EstadoSemaforo.VERMELHO_LIMPEZA:
            if tempo_no_estado >= self.tempo_limpeza:
                # Se havia pedestre ou animal, abre para pedestres
                if self.pedestres_detectados > 0 or self.animais_detectados > 0:
                    self._mudar_estado(EstadoSemaforo.VERDE_PEDESTRE)
                else:
                    self._mudar_estado(EstadoSemaforo.VERDE_CARRO)

        elif self.estado_atual == EstadoSemaforo.VERDE_PEDESTRE:
            if tempo_no_estado >= self.tempo_verde_pedestre:
                self._mudar_estado(EstadoSemaforo.VERMELHO_PISCANDO_PEDESTRE)

        elif self.estado_atual == EstadoSemaforo.VERMELHO_PISCANDO_PEDESTRE:
            if tempo_no_estado >= self.tempo_pisca_pedestre:
                self._mudar_estado(EstadoSemaforo.VERMELHO_LIMPEZA)

    def _mudar_estado(self, novo_estado):
        self.estado_atual = novo_estado
        self.inicio_estado = time.time()

    def obter_luzes(self):
        """
        Retorna o estado das lâmpadas para os atuadores digitais.
        Formato: (Carro_R, Carro_Y, Carro_G, Ped_R, Ped_G)
        """
        agora = time.time()
        tempo_no_estado = agora - self.inicio_estado
        pisca = int(tempo_no_estado * 2) % 2 == 0

        if self.estado_atual == EstadoSemaforo.VERDE_CARRO:
            return (False, False, True, True, False)
        elif self.estado_atual == EstadoSemaforo.AMARELO_CARRO:
            return (False, True, False, True, False)
        elif self.estado_atual == EstadoSemaforo.VERMELHO_LIMPEZA:
            return (True, False, False, True, False)
        elif self.estado_atual == EstadoSemaforo.VERDE_PEDESTRE:
            return (True, False, False, False, True)
        elif self.estado_atual == EstadoSemaforo.VERMELHO_PISCANDO_PEDESTRE:
            return (True, False, False, pisca, False)
        elif self.estado_atual == EstadoSemaforo.STANDBY_AMARELO:
            return (False, pisca, False, False, False)
        elif self.estado_atual == EstadoSemaforo.FAIL_SAFE:
            return (False, pisca, False, True, False)
        return (True, False, False, True, False)
