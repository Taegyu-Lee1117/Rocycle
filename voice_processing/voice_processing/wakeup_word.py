"""Detect Hello Rokey with a bounded microphone wait."""

from pathlib import Path
import time

import sounddevice as sd
from openwakeword.model import Model
from ament_index_python.packages import get_package_share_directory

from voice_processing.audio_device import resolve_input_device

MODEL_NAME = "hello_rokey_8332_32.tflite"
SAMPLE_RATE = 16000
FRAME = 1280


class WakeupWord:
    def __init__(self):
        self.model = None
        self.model_name = Path(MODEL_NAME).stem
        self.stream = None

    def is_wakeup(self):
        if self.stream.read_available < FRAME:
            return False
        audio_chunk, overflowed = self.stream.read(FRAME)
        if overflowed:
            raise RuntimeError("호출어 마이크 입력 버퍼 초과")
        confidence = self.model.predict(audio_chunk.flatten())[self.model_name]
        return bool(confidence > 0.3)

    def open(self):
        model_path = Path(get_package_share_directory("voice_processing")) / "resource" / MODEL_NAME
        self.model = Model(wakeword_models=[str(model_path)], inference_framework="tflite")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
            device=resolve_input_device(),
        )
        self.stream.start()

    def wait(self, timeout=30.0, keep_running=lambda: True):
        try:
            self.open()
            deadline = time.monotonic() + timeout
            while keep_running() and time.monotonic() < deadline:
                if self.is_wakeup():
                    return True
                time.sleep(0.01)
            return False
        finally:
            self.close()

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
