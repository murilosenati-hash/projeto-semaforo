# Relatório Técnico e Roteiro de Apresentação
## Semáforo Inteligente Adaptativo com Visão Computacional

**Matérias:** Sistemas Digitais & Sistemas Distribuídos  
**Aluno:** Murilo (Tomy)  
**Repositório:** `/home/murilods/projetos/semaforo`  

---

### 1. Resumo do Projeto

O projeto consiste em um sistema de controle de semáforos autônomo e adaptativo que otimiza o fluxo de trânsito urbano em tempo real. O sistema utiliza redes neurais convolucionais (**YOLOv8 Nano**) e rastreamento **ByteTrack** para identificar veículos, pedestres e animais, integrando os conceitos de **Sistemas Digitais** (Máquinas de Estados Finitos - FSM determinística com tempos adaptativos) e **Sistemas Distribuídos** (nós sensores de borda desacoplados, protocolo TCP Socket/REST e tolerância a falhas com fail-safe).

---

### 2. Fundamentação em Sistemas Digitais

* **Máquina de Estados Finitos (FSM - Modelo de Moore/Mealy Híbrido):**
  * `VERDE_CARRO`: Tempo adaptativo proporcional à fila de veículos ($T_{min}=5s$ a $T_{max}=25s$).
  * `AMARELO_CARRO`: Tempo fixo de segurança ($3s$) para desaceleração.
  * `VERMELHO_LIMPEZA`: Todos os semáforos em vermelho (*All-Red*, $2s$) para esvaziamento seguro do cruzamento.
  * `VERDE_PEDESTRE`: Liberação da travessia de pedestres ($8s$ a $10s$).
  * `VERMELHO_PISCANDO_PEDESTRE`: Sinalizador intermitente ($1\text{ Hz}$) alertando o fim iminente do tempo de travessia.
  * `STANDBY_AMARELO`: Amarelo piscante em vias desertas na madrugada ($V=0, P=0$ por mais de $12s$).
  * `FAIL_SAFE`: Estado de contingência máxima com amarelo intermitente caso ocorra perda de comunicação com nós sensores ($heartbeat > 8s$).

---

### 3. Fundamentação em Sistemas Distribuídos

* **Desacoplamento em Nós (Edge Computing):** O processamento pesado de visão computacional ocorre na borda (câmera IP, Raspberry Pi, workstation), transmitindo apenas telemetria leve (JSON: contagem e timestamp), economizando $>99\%$ de largura de banda em comparação ao streaming de vídeo bruto.
* **Comunicação Multicanal:** Socket TCP persistente com NDJSON (baixa latência) + API REST HTTP (`/api/telemetria` e `/api/status`).
* **Tolerância a Falhas e Heartbeat:** Monitoramento constante por timeout. Na ausência de dados de todos os nós por mais de 8 segundos, o controlador central assume `FAIL_SAFE` automaticamente sem congelar o tráfego.

---

### 4. Calibração Interativa de ROI (Region of Interest) em Tempo Real

* **Problema em Produção:** Câmeras sofrem trepidações, ventos ou variações angulares, além de enquadrarem calçadas e acostamentos que poluem a contagem de veículos.
* **Solução Implementada (Tecla `R`):**
  1. Ao pressionar a tecla `R` durante o funcionamento, o operador calibra interativamente os 4 pontos da pista diretamente sobre a janela de vídeo.
  2. As coordenadas são automaticamente normalizadas $[0.0, 1.0]$, garantindo portabilidade entre diferentes resoluções de câmera (Full HD, 720p ou inferência rápida 320x320).
  3. As coordenadas são persistidas em `roi_config.json`, sendo recarregadas no boot com fallback automático para a ROI padrão.
  4. Veículos fora da ROI demarcada são ignorados pelo algoritmo geométrico `cv2.pointPolygonTest`.

---

### 5. Roteiro Passo a Passo para Demonstrar ao Professor

1. **Abertura do Sistema:**
   ```bash
   cd ~/projetos/semaforo
   python3 main_integrated.py --source transito.mp4
   ```
2. **Demonstrar a IA:** Aponte para as caixas verdes delimitando veículos e a contagem no painel superior.
3. **Demonstrar a FSM Digital:** Mostre as luzes virtuais mudando na tela de acordo com os tempos e contagens.
4. **Demonstrar Calibração de ROI em Tempo Real:**
   * Pressione **`R`**: Explique que o sistema entra em modo de ajuste de câmera em tempo real para isolar apenas a pista útil.
5. **Demonstrar Interatividade:**
   * Pressione **`P`**: Explique que simula o pedestre apertando o botão ou entrando na faixa, forçando o ciclo de pedestre.
   * Pressione **`A`**: Explique a proteção contra atropelamento de animais na pista.
6. **Demonstrar os Testes Automatizados:**
   ```bash
   python3 tests/test_fsm.py
   python3 tests/test_detector.py
   ```
