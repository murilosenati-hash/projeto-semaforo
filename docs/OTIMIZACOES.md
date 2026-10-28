# Documento de Otimizações de Desempenho e Engenharia de Borda

**Projeto:** Semáforo Inteligente com Visão Computacional  
**Disciplina:** Sistemas Distribuídos e Sistemas Digitais  
**Autor:** Murilo (Tomy)  

---

## 1. Otimizações Implementadas e Validadas

### Otimização 1: Conversão de Codec para H.264 (Compatibilidade de Hardware)
* **Problema:** O vídeo de demonstração original estava em formato AV1, gerando avisos contínuos de falta de aceleração de hardware na GPU/CPU.
* **Solução:** Conversão direta via FFmpeg para H.264 standard (`yuv420p`).
* **Impacto:** Eliminação completa dos erros de decodificação no OpenCV e taxa de decodificação fluida a 22.5 FPS contínuos.

### Otimização 2: Calibração de Resolução de Inferência (`imgsz=320`)
* **Benchmark em CPU Real (Intel/AMD):**
  * `imgsz=640` (Padrão): 236.5 ms por frame (~4.2 FPS) — Sensivelmente lento.
  * `imgsz=320` (Otimizado): 76.7 ms por frame (~13.0 FPS).
* **Ganho:** **+208.3% mais rápido** mantendo acurácia total nas faixas delimitadas por ROI.

### Otimização 3: I/O Decoupling com Thread Dedicada de Vídeo (`core/video_stream.py`)
* **Arquitetura:** O OpenCV executa a leitura (`cap.read()`) em uma *daemon thread* concorrente protegida por mutex (`threading.Lock`), desacoplando o I/O de disco da inferência de rede neural.
* **Salto Dinâmico de Frames (`skip_frames=3`):**
  * A rede neural processa 1 a cada 3 frames, enquanto a renderização e o rastreador ByteTrack mantêm suavidade visual de 30 FPS na tela.

---

## 2. Roadmap de Otimizações Futuras (Próximos Passos)

1. **Exportação para OpenVINO / ONNX Runtime:**
   * Rodando `model.export(format='openvino')`, o grafo computacional é quantizado para INT8/FP16 e compilado com instruções vetoriais AVX2/AVX-512 da CPU, reduzindo pela metade a latência restante.
2. **Pipeline Zero-Copy no Shm (Shared Memory):**
   * Comunicação IPC entre processo de captura e processo de inferência via memória compartilhada POSIX.
