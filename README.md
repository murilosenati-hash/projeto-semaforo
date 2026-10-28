# 🚦 Semáforo Inteligente Distribuído com Visão Computacional (YOLOv8)

Projeto interdisciplinar desenvolvido para as disciplinas de **Sistemas Distribuídos** e **Sistemas Digitais**.

## 📂 Estrutura do Projeto

```text
~/projetos/semaforo/
├── core/
│   ├── detector.py          # Detector YOLOv8 + ByteTrack (Veículos, Pedestres e Animais com ROI)
│   └── fsm.py               # Máquina de Estados Finitos (FSM) com tempos adaptativos e Fail-Safe
├── distributed/
│   ├── server.py            # Controlador Central Distribuído (Servidor TCP Socket + API REST HTTP)
│   └── sensor_node.py       # Nó Sensor de Borda (Cliente TCP/HTTP com YOLOv8 ou Simulador)
├── ml/
│   ├── dataset_farol.csv    # Dataset tabular de decisões semafóricas
│   └── pipeline_decisao.py  # Pré-processamento (SimpleImputer + MinMaxScaler) e RandomForest
├── tests/
│   ├── test_fsm.py          # Testes unitários da Máquina de Estados e Fail-Safe
│   └── test_distributed.py  # Testes integrados Cliente/Servidor (TCP + HTTP + Multi-nó)
├── docs/
│   └── arquitetura.md       # Tabela verdade digital, diagrama de estados e protocolo de rede
└── main_integrated.py       # Aplicação visual integrada com HUD OpenCV em tempo real
```

---

## 🚀 Como Executar

### 1. Modo Visual Integrado (OpenCV HUD + YOLOv8 + FSM)
Roda o vídeo `transito.mp4` (ou webcam) desenhando as detecções, a área demarcada (ROI) e o painel digital de lâmpadas do semáforo:
```bash
python3 main_integrated.py --source transito.mp4
```
- Pressione **`P`** na janela de vídeo para simular um pedestre pedindo travessia.
- Pressione **`A`** para simular um animal na pista (aciona fechamento de segurança).
- Pressione **`Q`** para sair.

### 2. Modo Sistemas Distribuídos (Cliente / Servidor)
Abra dois terminais:

**Terminal 1 — Iniciar o Controlador Central (TCP 5050 + HTTP 8080):**
```bash
python3 distributed/server.py --tcp-port 5050 --http-port 8080
```

**Terminal 2 — Iniciar Nó(s) Sensor(es):**
```bash
# Modo simulação (gera cenários de tráfego via Socket TCP):
python3 distributed/sensor_node.py --node-id cam_norte --protocolo tcp --port 5050 --simulate

# Ou usando API REST HTTP:
python3 distributed/sensor_node.py --node-id cam_sul --protocolo http --port 8080 --simulate

# Ou processando vídeo real com YOLOv8 no nó sensor:
python3 distributed/sensor_node.py --node-id cam_norte --protocolo tcp --port 5050 --source transito.mp4
```

**Consultar Status da FSM via HTTP REST:**
```bash
curl -s http://127.0.0.1:8080/api/status | python3 -m json.tool
```

### 3. Executar a Suíte de Testes e Pipeline ML
```bash
python3 tests/test_fsm.py
python3 tests/test_distributed.py
python3 ml/pipeline_decisao.py
```
