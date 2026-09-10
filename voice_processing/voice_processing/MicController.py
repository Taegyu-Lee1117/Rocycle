from dataclasses import dataclass
import wave
import io
import pyaudio


@dataclass
class MicConfig:
    chunk: int = 12000
    rate: int = 16000
    channels: int = 1
    record_seconds: int = 5
    fmt: int = pyaudio.paInt16
    device_index: int = 10
    buffer_size: int = 24000


class MicController:
    def __init__(self, config: MicConfig = MicConfig()):
        self.config = config
        self.frames = []
        self.audio = None
        self.stream = None
        self.sample_width = None

    def open_stream(self):
        self.audio = pyaudio.PyAudio()
        self.sample_width = self.audio.get_sample_size(self.config.fmt)
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
        )

    def record_audio(self):
        print("start recording for 5 seconds")
        self.frames = []
        num_chunks = int(self.config.rate / self.config.chunk * self.config.record_seconds)

        for _ in range(num_chunks):
            data = self.stream.read(self.config.chunk, exception_on_overflow=False)
            self.frames.append(data)

    def close_stream(self):
        print("stop recording")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def save_wav(self, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.config.rate)
            wf.writeframes(b''.join(self.frames))
        print("파일 저장 완료!")

    def get_wav_data(self):
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.fmt))
            wf.setframerate(self.config.rate)
            wf.writeframes(b''.join(self.frames))
        return wav_buffer.getvalue()
