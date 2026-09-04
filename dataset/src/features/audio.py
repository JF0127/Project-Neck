"""Speech V1 audio extraction and WAV/video alignment validation."""

from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_CODEC = "pcm_s16le"
AUDIO_ALIGNMENT_TOLERANCE_SECONDS = 0.10


class AudioExtractionError(RuntimeError):
    """Raised when standardized audio cannot be extracted or validated."""


def probe_video_duration(video_path: Path) -> float:
    """Read the video-stream duration, falling back to container duration."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration:format=duration", "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioExtractionError(f"ffprobe failed: {result.stderr.strip()[:1000]}")
    try:
        payload = json.loads(result.stdout)
        candidates = [
            *((stream.get("duration") for stream in payload.get("streams", []))),
            payload.get("format", {}).get("duration"),
        ]
        duration = next(
            float(value)
            for value in candidates
            if value not in (None, "N/A") and math.isfinite(float(value)) and float(value) > 0
        )
    except (ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        raise AudioExtractionError(f"could not determine video duration: {video_path}") from exc
    return duration


def probe_wav(wav_path: Path) -> dict[str, Any]:
    """Validate and describe an uncompressed WAV using its header."""
    try:
        with wave.open(str(wav_path), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            frame_count = source.getnframes()
            compression = source.getcomptype()
    except (OSError, wave.Error) as exc:
        raise AudioExtractionError(f"invalid WAV file {wav_path}: {exc}") from exc
    duration = frame_count / sample_rate if sample_rate > 0 else 0.0
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise AudioExtractionError(f"expected 16000 Hz WAV, got {sample_rate}")
    if channels != AUDIO_CHANNELS:
        raise AudioExtractionError(f"expected mono WAV, got {channels} channels")
    if sample_width != AUDIO_SAMPLE_WIDTH_BYTES:
        raise AudioExtractionError(f"expected signed 16-bit WAV, got {sample_width * 8} bit")
    if compression != "NONE":
        raise AudioExtractionError(f"expected PCM WAV, got compression {compression}")
    if not math.isfinite(duration) or duration <= 0:
        raise AudioExtractionError("WAV duration must be positive")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "frame_count": frame_count,
        "format": AUDIO_CODEC,
        "duration": duration,
    }


def extract_standard_audio(video_path: Path, wav_path: Path) -> dict[str, Any]:
    """Extract clip-local 16 kHz mono PCM audio without trimming or normalization."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-map", "0:a:0", "-vn",
        "-af", "asetpts=PTS-STARTPTS", "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS), "-c:a", AUDIO_CODEC, str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
        wav_path.unlink(missing_ok=True)
        message = result.stderr.strip() or "ffmpeg produced no audio"
        raise AudioExtractionError(f"ffmpeg audio extraction failed: {message[:1000]}")
    return probe_wav(wav_path)


def validate_audio_alignment(
    video_duration: float,
    audio_duration: float,
    tolerance: float = AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
) -> float:
    """Return signed audio-minus-video duration difference, rejecting large mismatch."""
    difference = float(audio_duration - video_duration)
    if abs(difference) > tolerance:
        raise AudioExtractionError(
            f"alignment error: video={video_duration:.6f}s audio={audio_duration:.6f}s "
            f"difference={difference:+.6f}s exceeds {tolerance:.2f}s"
        )
    return difference
