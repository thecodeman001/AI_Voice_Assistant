import io
import os
import time
import wave
from typing import Tuple

import webrtcvad
import sounddevice as sd
from groq import Groq


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    bio.seek(0)
    return bio.read()


class ASRClient:
    def __init__(self, model: str | None = None, sample_rate: int = 16000):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
        self.sample_rate = sample_rate

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> Tuple[str, float]:
        t0 = time.perf_counter()
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        resp = self.client.audio.transcriptions.create(model=self.model, file=bio)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = getattr(resp, "text", "")
        return text, latency_ms

    def streaming_listen(self, vad_stream: "VADStream", on_partial=lambda t: None,
                          partial_interval_ms: int = 800,
                          min_speech_ms: int = 200,
                          max_silence_ms: int = 600) -> Tuple[str, float, float]:
        buf = bytearray()
        started = False
        voiced_ms = 0
        unvoiced_ms = 0
        last_partial_time = time.perf_counter()
        t_listen_start = time.perf_counter()

        while True:
            data = vad_stream.stream.read(int(vad_stream.frame_bytes / 2))[0].tobytes()
            if len(data) < vad_stream.frame_bytes:
                continue
            is_speech = vad_stream.vad.is_speech(data, vad_stream.sample_rate)
            if is_speech:
                buf.extend(data)
                voiced_ms += 30
                unvoiced_ms = 0
                if not started and voiced_ms >= min_speech_ms:
                    started = True
            else:
                if started:
                    unvoiced_ms += 30
                    if unvoiced_ms >= max_silence_ms:
                        break
                else:
                    voiced_ms = 0

            # Emit partials at intervals once started and enough audio accumulated
            if started:
                now = time.perf_counter()
                if (now - last_partial_time) * 1000 >= partial_interval_ms and len(buf) > int(self.sample_rate * 0.5) * 2:
                    try:
                        wav = pcm16_to_wav_bytes(bytes(buf), self.sample_rate)
                        text, _ = self.transcribe_wav_bytes(wav)
                        if text:
                            on_partial(text)
                    except Exception:
                        pass
                    last_partial_time = now

        # Final transcription
        wav = pcm16_to_wav_bytes(bytes(buf), self.sample_rate)
        final_text, final_ms = self.transcribe_wav_bytes(wav)
        asr_secs = len(buf) / 2 / self.sample_rate
        return final_text, final_ms, asr_secs


class VADStream:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.stream = None

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16")
        self.stream.start()

    def read_frames(self, duration_ms: int) -> bytes:
        frames = []
        total_bytes = int(self.sample_rate * (duration_ms / 1000.0) * 2)
        remaining = total_bytes
        while remaining > 0:
            block = self.stream.read(int(self.frame_bytes / 2))[0].tobytes()
            frames.append(block)
            remaining -= len(block)
        return b"".join(frames)

    def detect_speech_segment(self, max_silence_ms: int = 600, min_speech_ms: int = 200) -> bytes:
        speech = bytearray()
        voiced = 0
        unvoiced = 0
        started = False
        while True:
            data = self.stream.read(int(self.frame_bytes / 2))[0].tobytes()
            if len(data) < self.frame_bytes:
                continue
            is_speech = self.vad.is_speech(data, self.sample_rate)
            if is_speech:
                voiced += 1
                unvoiced = 0
                speech.extend(data)
                if not started and voiced * 30 >= min_speech_ms:
                    started = True
            else:
                if started:
                    unvoiced += 1
                    if unvoiced * 30 >= max_silence_ms:
                        break
                else:
                    voiced = 0
        return bytes(speech)
