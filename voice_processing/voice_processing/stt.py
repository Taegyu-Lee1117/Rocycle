"""Record five seconds and transcribe with Whisper."""

import io
import time

from openai import OpenAI
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

from voice_processing.audio_device import resolve_input_device
from voice_processing.keyword_extraction import ConfigurationError


class STT:
    def __init__(self, openai_api_key):
        if not openai_api_key or not openai_api_key.strip():
            raise ConfigurationError("OPENAI_API_KEY가 없습니다.")
        self.client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=0)
        self.duration = 5
        self.samplerate = 16000

    def speech2text(self):
        stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="int16",
            device=resolve_input_device(),
        )
        chunks = []
        remaining = self.duration * self.samplerate
        try:
            stream.start()
            deadline = time.monotonic() + self.duration + 2.0
            while remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError("마이크 녹음 시간 초과")
                count = min(stream.read_available, remaining, 1280)
                if count == 0:
                    time.sleep(0.01)
                    continue
                chunk, overflowed = stream.read(count)
                if overflowed:
                    raise RuntimeError("마이크 입력 버퍼 초과")
                chunks.append(chunk.copy())
                remaining -= count
        finally:
            try:
                stream.stop()
            finally:
                stream.close()
        with io.BytesIO() as audio_file:
            audio_file.name = "command.wav"
            wav.write(audio_file, self.samplerate, np.concatenate(chunks))
            audio_file.seek(0)
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language="ko", temperature=0,
            )
        return transcript.text.strip()

    def close(self):
        self.client.close()
