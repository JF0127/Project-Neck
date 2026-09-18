"""Persist one generated turn as robot audio + trajectory + metadata files.

Files are written atomically and metadata is written last, so a complete
metadata file marks a complete generation (the handoff point for playback).
"""

from __future__ import annotations

import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from ..contracts import RobotSpeech

SAMPLE_RATE = 16_000
AUDIO_FILENAME = "robot.wav"
TRAJECTORY_FILENAME = "trajectory.json"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class GenerationArtifacts:
    directory: Path
    audio_path: Path
    trajectory_path: Path | None
    metadata_path: Path


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_wav_atomic(path: str | Path, pcm_s16le: bytes) -> Path:
    """Atomically write 16 kHz mono PCM s16le bytes as a WAV file."""
    if not pcm_s16le or len(pcm_s16le) % 2:
        raise ValueError("robot pcm_s16le must contain an even non-zero byte count")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm_s16le)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_generation(
    directory: str | Path,
    speech: RobotSpeech,
    document: dict | None,
    metadata: dict,
) -> GenerationArtifacts:
    """Write ``robot.wav``, optional ``trajectory.json`` and ``metadata.json``."""
    if not isinstance(speech, RobotSpeech):
        raise TypeError("save_generation requires RobotSpeech")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a mapping")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    audio_path = directory / AUDIO_FILENAME
    write_wav_atomic(audio_path, speech.pcm_s16le)

    trajectory_path: Path | None = None
    if document is not None:
        if not isinstance(document, dict):
            raise TypeError("trajectory document must be a mapping")
        trajectory_path = directory / TRAJECTORY_FILENAME
        _write_json_atomic(trajectory_path, document)

    complete = dict(metadata)
    complete.setdefault("complete", True)
    complete.setdefault("saved_unix_sec", time.time())
    metadata_path = directory / METADATA_FILENAME
    _write_json_atomic(metadata_path, complete)

    return GenerationArtifacts(
        directory=directory,
        audio_path=audio_path,
        trajectory_path=trajectory_path,
        metadata_path=metadata_path,
    )
