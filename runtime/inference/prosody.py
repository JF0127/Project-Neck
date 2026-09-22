"""Simple explainable prosody features derived from complete TTS PCM."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any

import numpy as np

from .speech_alignment import SpeechAlignment, SpeechSegment

PROSODY_FRAME_SEC = 0.01
LOW_RELATIVE_ENERGY = 0.85
HIGH_RELATIVE_ENERGY = 1.15
SLOW_RATE_CHARS_SEC = 4.0
FAST_RATE_CHARS_SEC = 7.0
_LEXICAL_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")


class ProsodyError(ValueError):
    """TTS PCM or aligned segments cannot produce safe prosody features."""


@dataclass(frozen=True)
class SegmentProsody:
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    pause_before_sec: float
    pause_after_sec: float
    rms_mean: float
    rms_peak: float
    relative_energy: float
    energy_level: str
    speech_rate_chars_sec: float
    rate_level: str
    energy_peak_time_sec: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "text": self.text,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "prosody": {
                "duration_sec": self.duration_sec,
                "pause_before_sec": self.pause_before_sec,
                "pause_after_sec": self.pause_after_sec,
                "rms_mean": self.rms_mean,
                "rms_peak": self.rms_peak,
                "relative_energy": self.relative_energy,
                "energy_level": self.energy_level,
                "speech_rate_chars_sec": self.speech_rate_chars_sec,
                "rate_level": self.rate_level,
                "energy_peak_time_sec": self.energy_peak_time_sec,
            },
        }


@dataclass(frozen=True)
class ProsodyAnalysis:
    duration_sec: float
    sample_rate: int
    segments: tuple[SegmentProsody, ...]
    latency_sec: float

    def payload_segments(self) -> list[dict[str, Any]]:
        return [segment.to_dict() for segment in self.segments]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "segments": self.payload_segments(),
            "latency_sec": self.latency_sec,
            "unavailable_features": {
                "f0": "not_extracted_no_reliable_installed_dependency"
            },
        }


def _level(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "medium"


def _rate_level(value: float) -> str:
    if value < SLOW_RATE_CHARS_SEC:
        return "slow"
    if value > FAST_RATE_CHARS_SEC:
        return "fast"
    return "normal"


def _pcm_array(pcm_s16le: bytes) -> np.ndarray:
    if not isinstance(pcm_s16le, bytes) or not pcm_s16le or len(pcm_s16le) % 2:
        raise ProsodyError("PCM must contain complete non-empty s16le samples")
    return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float64) / 32768.0


def _segment_samples(
    samples: np.ndarray,
    segment: SpeechSegment,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    start = max(0, min(len(samples), int(round(segment.start * sample_rate))))
    end = max(start + 1, min(len(samples), int(round(segment.end * sample_rate))))
    if start >= len(samples) or end <= start:
        raise ProsodyError(f"aligned segment {segment.text!r} has no PCM samples")
    return samples[start:end], start


def _frame_rms(samples: np.ndarray, frame_samples: int) -> np.ndarray:
    values = []
    for start in range(0, len(samples), frame_samples):
        frame = samples[start : start + frame_samples]
        if len(frame):
            values.append(float(np.sqrt(np.mean(np.square(frame)))))
    return np.asarray(values, dtype=np.float64)


def extract_prosody(
    pcm_s16le: bytes,
    sample_rate: int,
    alignment: SpeechAlignment,
) -> ProsodyAnalysis:
    """Extract segment timing, energy and lexical-rate features without F0."""
    started = time.perf_counter()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ProsodyError("sample_rate must be a positive integer")
    if not isinstance(alignment, SpeechAlignment) or not alignment.segments:
        raise ProsodyError("alignment must contain at least one SpeechSegment")
    samples = _pcm_array(pcm_s16le)
    audio_duration = len(samples) / sample_rate
    frame_samples = max(1, round(sample_rate * PROSODY_FRAME_SEC))

    raw: list[tuple[SpeechSegment, np.ndarray, int, np.ndarray]] = []
    speech_square_sum = 0.0
    speech_sample_count = 0
    for segment in alignment.segments:
        if segment.start < 0.0 or segment.end > audio_duration + 1.0 / sample_rate:
            raise ProsodyError("aligned segment lies outside PCM duration")
        segment_pcm, sample_start = _segment_samples(samples, segment, sample_rate)
        rms = _frame_rms(segment_pcm, frame_samples)
        raw.append((segment, segment_pcm, sample_start, rms))
        speech_square_sum += float(np.sum(np.square(segment_pcm)))
        speech_sample_count += len(segment_pcm)
    utterance_rms = (
        math.sqrt(speech_square_sum / speech_sample_count)
        if speech_sample_count
        else 0.0
    )

    output: list[SegmentProsody] = []
    for index, (segment, _, sample_start, rms) in enumerate(raw):
        duration = segment.end - segment.start
        rms_mean = float(np.mean(rms)) if len(rms) else 0.0
        rms_peak = float(np.max(rms)) if len(rms) else 0.0
        relative_energy = rms_mean / utterance_rms if utterance_rms > 1e-12 else 0.0
        peak_index = int(np.argmax(rms)) if len(rms) else 0
        peak_time = min(
            segment.end,
            segment.start + (peak_index + 0.5) * frame_samples / sample_rate,
        )
        previous_end = raw[index - 1][0].end if index else 0.0
        next_start = raw[index + 1][0].start if index + 1 < len(raw) else audio_duration
        lexical_count = len(_LEXICAL_RE.findall(segment.text))
        speech_rate = lexical_count / duration if duration > 0.0 else 0.0
        output.append(
            SegmentProsody(
                text=segment.text,
                start_sec=float(segment.start),
                end_sec=float(segment.end),
                duration_sec=float(duration),
                pause_before_sec=max(0.0, float(segment.start - previous_end)),
                pause_after_sec=max(0.0, float(next_start - segment.end)),
                rms_mean=rms_mean,
                rms_peak=rms_peak,
                relative_energy=relative_energy,
                energy_level=_level(
                    relative_energy, LOW_RELATIVE_ENERGY, HIGH_RELATIVE_ENERGY
                ),
                speech_rate_chars_sec=float(speech_rate),
                rate_level=_rate_level(speech_rate),
                energy_peak_time_sec=float(peak_time),
            )
        )
    return ProsodyAnalysis(
        duration_sec=float(audio_duration),
        sample_rate=sample_rate,
        segments=tuple(output),
        latency_sec=time.perf_counter() - started,
    )
