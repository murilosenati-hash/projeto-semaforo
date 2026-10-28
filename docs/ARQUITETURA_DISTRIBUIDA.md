# Arquitetura Distribuída: Semáforo Inteligente com Visão Computacional

**Disciplina:** Sistemas Distribuídos e Sistemas Digitais  
**Aluno:** Murilo (Tomy)  

---

## 1. Visão Geral da Arquitetura

O sistema adota uma **Arquitetura Distribuída em Camadas (Edge-to-Controller)**, separando as responsabilidades de captura e inteligência visual (nós de borda) da lógica crítica de segurança e chaveamento de estados semafóricos (controlador central):

```mermaid
flowchart TD
    subgraph Edge_Nodes [Nós Sensores de Borda - Câmeras]
        CAM1["Câmera 1 (Faixa/IP Webcam)\nYOLOv8 + ByteTrack"]
        CAM2["Câmera 2 (Acesso Local)\nYOLOv8 + ByteTrack"]
    end

    subgraph Network [Camada de Rede / Comunicação Distribuída]
        JSON_MSG["Telemetria JSON + Heartbeat\n(TCP/UDP Socket ou REST)"]
    end

    subgraph Central_Controller [Controlador Central do Semáforo]
        FSM["Máquina de Estados Finitos (FSM)\nTemporizadores Digitais & Segurança"]
        FAILSAFE["Monitor de Watchdog / Fail-Safe\n(Timeout de 8s)"]
        ACTUATOR["Atuador Digital de Semáforos\n(Carros + Pedestres)"]
    end

    CAM1 --> JSON_MSG
    CAM2 --> JSON_MSG
    JSON_MSG --> FSM
    JSON_MSG --> FAILSAFE
    FSM --> ACTUATOR
```

---

## 2. Protocolo de Comunicação (JSON Payload)

Cada nó de visão transmite telemetria periódica a cada frame processado (ou em intervalos de 500ms):

```json
{
  "node_id": "cam_01",
  "timestamp": 1730145600.123,
  "sequence_number": 1042,
  "payload": {
    "vehicles": 4,
    "pedestrians": 1,
    "animals": 0,
    "confidence_avg": 0.72
  },
  "heartbeat": true
}
```

---

## 3. Tolerância a Falhas e Fail-Safe (Conceito de Sistemas Distribuídos)

1. **Detecção de Falha por Ausência de Heartbeat:**
   * Se a câmera ou o link de rede cair por mais de **8 segundos**, o controlador assume o estado `FAIL_SAFE`.
2. **Operação em Modo Seguro:**
   * O semáforo chaveia para ciclo de temporização fixa padrão ou amarelo piscante de atenção.
   * Impede que uma pane de software ou desligamento de câmera congele o semáforo no vermelho perpetuamente.
3. **Recuperação Automática (Self-Healing):**
   * Assim que um novo pacote válido do nó de borda chega, o sistema transita com segurança pelo estado `VERMELHO_LIMPEZA` antes de retomar a adaptabilidade dinâmica.
