import cv2
import threading
import time

class ThreadedVideoStream:
    """
    Leitor de vídeo/webcam desacoplado em Thread dedicada (I/O Decoupling).
    Elimina o gargalo onde a CPU fica esperando o disco ou driver da câmera
    decodificar o frame antes de rodar a inferência.
    """
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.is_file = isinstance(src, str) and not src.isdigit()

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            grabbed, frame = self.cap.read()
            if not grabbed:
                if self.is_file:
                    # Loop contínuo para apresentação acadêmica
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    self.stopped = True
                    break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005) # Yield leve para não saturar núcleo ocioso

    def read(self):
        with self.lock:
            return self.grabbed, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
