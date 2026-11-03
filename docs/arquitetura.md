# Arquitetura Técnica — Semáforo Inteligente (Sistemas Digitais & Sistemas Distribuídos)

## 1. Visão Geral da Arquitetura
O projeto unifica três pilares de engenharia:
1. **Visão Computacional Multiclasse (`core/detector.py`)**: Inferência com **YOLOv8 Nano (`yolov8n.pt`)** + rastreamento **ByteTrack** filtrando objetos dentro de um polígono de interesse (**ROI**) e classificando **Veículos** (carro, moto, ônibus, caminhão, bicicleta), **Pedestres** (pessoas) e **Animais** (cães, gatos, cavalos, etc.).
2. **Sistemas Digitais (`core/fsm.py`)**: Máquina de Estados Finitos (**FSM** tipo Moore/Mealy híbrida) determinística com temporizadores adaptativos, fase de limpeza (*all-red*), proteção prioritária para pedestres/animais, modo *Standby* de madrugada e modo *Fail-Safe* (amarelo intermitente).
3. **Sistemas Distribuídos (`distributed/server.py` e `distributed/sensor_node.py`)**: Arquitetura **Edge-to-Central** onde múltiplos nós sensores de borda enviam telemetria em tempo real via **Sockets TCP (NDJSON)** ou **API REST HTTP** para o Controlador Central, com detecção de falhas por *heartbeat timeout*.

---

## 2. Especificação de Sistemas Digitais (Máquina de Estados Finitos)

### 2.1 Estados Codificados (`EstadoSemaforo`)
| Estado | Descrição | Condição de Transição (Próximo Estado) |
|---|---|---|
| `VERDE_CARRO` | Fluxo aberto para veículos ($T_{min}=5s$, $T_{max}=30s$) | Se animal/pedestre e $t \ge T_{min} \rightarrow$ `AMARELO_CARRO`<br>Se $t \ge T_{verde\_adaptativo} \rightarrow$ `AMARELO_CARRO`<br>Se $V=0, P=0$ por $>12s \rightarrow$ `STANDBY_AMARELO` |
| `AMARELO_CARRO` | Atenção / desaceleração ($3s$) | $t \ge 3s \rightarrow$ `VERMELHO_LIMPEZA` |
| `VERMELHO_LIMPEZA` | Fase *All-Red* de segurança ($2s$) | Se $P > 0$ ou $A > 0 \rightarrow$ `VERDE_PEDESTRE`<br>Senão $\rightarrow$ `VERDE_CARRO` |
| `VERDE_PEDESTRE` | Travessia aberta para pedestres ($10s$) | $t \ge 10s \rightarrow$ `VERMELHO_PISCANDO_PEDESTRE` |
| `VERMELHO_PISCANDO_PEDESTRE` | Alerta de fim de travessia ($4s$) | $t \ge 4s \rightarrow$ `VERMELHO_LIMPEZA` |
| `STANDBY_AMARELO` | Economia / madrugada (amarelo piscante) | Se $V>0, P>0$ ou $A>0 \rightarrow$ `VERMELHO_LIMPEZA` |
| `FAIL_SAFE` | Falha de comunicação / *heartbeat* $> 8s$ | Amarelo piscante p/ carros e vermelho p/ pedestres até recuperar rede |

### 2.2 Tabela Verdade dos Atuadores Digitais (`obter_luzes()`)
Saídas binárias: `(Carro_R, Carro_Y, Carro_G, Ped_R, Ped_G)` onde `CLK_PISCA` é um clock de $1\text{ Hz}$:

| Estado Atual | `Carro_R` | `Carro_Y` | `Carro_G` | `Ped_R` | `Ped_G` |
|---|:---:|:---:|:---:|:---:|:---:|
| `VERDE_CARRO` | 0 | 0 | 1 | 1 | 0 |
| `AMARELO_CARRO` | 0 | 1 | 0 | 1 | 0 |
| `VERMELHO_LIMPEZA` | 1 | 0 | 0 | 1 | 0 |
| `VERDE_PEDESTRE` | 1 | 0 | 0 | 0 | 1 |
| `VERMELHO_PISCANDO_PEDESTRE` | 1 | 0 | 0 | `CLK` | 0 |
| `STANDBY_AMARELO` | 0 | `CLK` | 0 | 0 | 0 |
| `FAIL_SAFE` | 0 | `CLK` | 0 | 1 | 0 |

---

## 3. Especificação de Sistemas Distribuídos

### 3.1 Topologia e Protocolos
- **Nó Sensor (`distributed/sensor_node.py`)**: Executa na borda (câmera da via Norte/Sul/Leste/Oeste), extrai apenas os metadados leves (`veiculos`, `pedestres`, `animais`, `timestamp`) reduzindo o consumo de banda em $>99\%$ em relação ao streaming de vídeo bruto.
- **Controlador Central (`distributed/server.py`)**:
  - **Canal 1 — TCP Socket (`porta 5050`)**: Conexão persistente de baixa latência usando mensagens NDJSON (JSON separado por `\n`).
  - **Canal 2 — API REST HTTP (`porta 8080`)**:
    - `POST /api/telemetria`: Recebe leituras de nós sensores HTTP/IoT.
    - `GET /api/status`: Retorna o estado completo da FSM, lâmpadas digitais e saúde dos nós.
- **Tolerância a Falhas**: Cada nó possui janela de vida (`node_timeout = 6.0s`). Se todos os sensores perderem conexão por mais de `8.0s`, o controlador comuta automaticamente para `FAIL_SAFE`.

---

## 4. Calibração Dinâmica e Persistência de ROI (Region of Interest)

### 4.1 Conceito e Motivação de Engenharia
Em cenários reais de tráfego, câmeras sofrem vibrações estruturais ou necessitam de ajustes finos de enquadramento para ignorar calçadas, acostamentos, vagas de estacionamento ou vias secundárias. A calibração dinâmica resolve isso sem necessidade de recompilar código ou reiniciar processos.

### 4.2 Fluxo Interativo de Calibração (Tecla `R`)
1. **Acionamento**: Ao pressionar a tecla `R` durante a execução, o pipeline entra em modo de calibração interativa.
2. **Demarcação Poligonal**: O usuário marca com o mouse os 4 vértices que delimitam a área útil da pista na imagem.
3. **Normalização de Coordenadas**: As coordenadas em pixels são convertidas para o espaço normalizado $[0.0, 1.0]$:
   $$x_{norm} = \frac{x_{pixel}}{W}, \quad y_{norm} = \frac{y_{pixel}}{H}$$
   Isso garante invariância completa de resolução (Full HD, 720p ou resolução reduzida de inferência de 320x320).
4. **Persistência Atômica (`roi_config.json`)**: A configuração é salva em disco via `salvar_roi()` em formato JSON:
   ```json
   [
     [0.15, 0.30],
     [0.85, 0.30],
     [0.98, 0.98],
     [0.02, 0.98]
   ]
   ```
5. **Recarregamento Automático e Fallback**: Na inicialização do sistema ou nós distribuídos, `carregar_roi()` carrega o JSON configurado; se inexistente ou corrompido, aplica fallback gracioso para a `ROI_PADRAO`.
6. **Filtragem Espacial Contínua**: O `DetectorTrafego.ponto_dentro_roi()` valida se o centro geométrico inferior do veículo/pedestre reside no polígono (`cv2.pointPolygonTest`), descartando falsos positivos fora da pista ativa.
