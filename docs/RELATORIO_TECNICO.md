# Relatório Técnico e Roteiro de Apresentação
## Semáforo Inteligente Adaptativo com Visão Computacional

**Matérias:** Sistemas Digitais & Sistemas Distribuídos  
**Aluno:** Murilo (Tomy)  

---

### 1. Resumo do Projeto

O projeto consiste em um sistema de controle de semáforos autônomo e adaptativo que otimiza o fluxo de trânsito urbano em tempo real. O sistema utiliza redes neurais convolucionais (YOLOv8) para identificar veículos, pedestres e animais, combinando técnicas de **Sistemas Digitais** (Máquinas de Estados Finitos - FSM) e **Sistemas Distribuídos** (nós de sensores independentes, comunicação em rede e tolerância a falhas).

---

### 2. Fundamentação em Sistemas Digitais

* **Máquina de Estados Finitos (FSM - Modelo de Moore):**
  * `VERDE_CARRO`: Tempo adaptativo proporcional à fila de veículos (5s a 25s).
  * `AMARELO_CARRO`: Tempo fixo de segurança (3s) para frenagem.
  * `VERMELHO_LIMPEZA`: Todos os semáforos em vermelho (2s) para esvaziamento seguro do cruzamento.
  * `VERDE_PEDESTRE`: Liberação da travessia de pedestres (8s a 10s).
  * `VERMELHO_PISCANDO_PEDESTRE`: Sinalizador intermitente alertando o fim do tempo de travessia.
  * `STANDBY_AMARELO`: Economia de energia em vias desertas de madrugada.
  * `FAIL_SAFE`: Estado de segurança rígido caso ocorra perda de comunicação.

---

### 3. Fundamentação em Sistemas Distribuídos

* **Desacoplamento em Nós:** O processamento pesado de IA (câmera IP / celular / Raspberry Pi) roda em um dispositivo de borda, enquanto o atuador digital e a FSM rodam no microcontrolador/computador de controle.
* **Tolerância a Falhas:** Monitoramento constante de heartbeat. Se um nó cair, a rede continua viva e funcional em modo autônomo.

---

### 4. Roteiro Passo a Passo para Demonstrar ao Professor

1. **Abertura do Sistema:**
   ```bash
   cd ~/projetos/semaforo
   python3 main_integrated.py --source transito.mp4
   ```
2. **Demonstrar a IA:** Aponte para as caixas verdes delimitando veículos e a contagem no painel superior.
3. **Demonstrar a FSM Digital:** Mostre as luzes virtuais mudando na tela de acordo com os tempos e contagens.
4. **Demonstrar Interatividade:**
   * Pressione **`P`**: Explique que simula o pedestre apertando o botão ou entrando na faixa, forçando o ciclo de pedestre.
   * Pressione **`A`**: Explique a proteção contra atropelamento de animais na pista.
5. **Demonstrar os Testes de Unidade:**
   ```bash
   python3 tests/test_fsm.py
   ```
