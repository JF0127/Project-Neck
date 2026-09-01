"""In-memory faster-whisper transcription for 16 kHz mono PCM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start_time: float
    end_time: float

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass(frozen=True)
class Transcription:
    text: str
    words: list[WordTimestamp]
    language: str | None


class WhisperASR:
    """One resident WhisperModel. Input never needs an intermediate WAV file."""

    def __init__(self, model_path: str, device: str = "cpu", language: str | None = "en"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[runtime][asr] loading Whisper once: {model_path} ({device}/{compute_type})")
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        self.language = language
        self.transcription_count = 0
        print("[runtime][asr] Whisper ready")

    @staticmethod
    def pcm_to_float32(pcm_s16le: bytes) -> np.ndarray:
        if len(pcm_s16le) % 2:
            raise ValueError("pcm_s16le byte length must be even")
        return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0

    def transcribe_pcm(self, pcm_s16le: bytes) -> Transcription:
        return self.transcribe_waveform(self.pcm_to_float32(pcm_s16le))

    def transcribe_waveform(self, waveform: np.ndarray) -> Transcription:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return Transcription(text="", words=[], language=self.language)

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            word_timestamps=True,
            vad_filter=False,
        )
        words: list[WordTimestamp] = []
        segment_text: list[str] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                segment_text.append(text)
            for word in segment.words or []:
                token = (word.word or "").strip()
                if token and word.start is not None and word.end is not None:
                    words.append(WordTimestamp(token, float(word.start), float(word.end)))

        self.transcription_count += 1
        text = " ".join(segment_text).strip()
        if not text and words:
            text = " ".join(word.text for word in words)
        language = getattr(info, "language", None) or self.language
        print(
            f"[runtime][asr] transcription #{self.transcription_count}: "
            f"text={text!r}, words={len(words)}"
        )
        return Transcription(text=text, words=words, language=language)
