"""Create pending-review Qwen3-ASR transcripts with forced-alignment timestamps."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "qwen"
DEFAULT_ALIGNER = PROJECT_ROOT / "models" / "qwen3-forced-aligner"
OUTPUT_SUFFIX = ".qwen_asr_v1.json"
SAMPLE_RATE = 16_000
_SUPPORTED_SUFFIXES = {".wav", ".mp4"}


class QwenTranscriptionError(RuntimeError):
    """Raised when a Qwen transcript cannot be generated or published safely."""


def _validate_wav(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            audio_format = (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getcomptype(),
            )
            frame_count = source.getnframes()
    except (OSError, wave.Error) as exc:
        raise QwenTranscriptionError(f"could not read WAV: {path}") from exc
    if audio_format != (16000, 1, 2, "NONE") or frame_count <= 0:
        raise QwenTranscriptionError(
            f"WAV must be non-empty 16 kHz mono PCM s16: {path}"
        )
    return frame_count / 16000.0


def _model_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not (resolved / "config.json").is_file():
        raise QwenTranscriptionError(f"{label} model directory is invalid: {resolved}")
    return resolved


def _audio_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    if not files:
        raise QwenTranscriptionError(f"no WAV or MP4 files found in: {root}")
    return files


def _decode_mp4(path: Path, ffmpeg: str) -> tuple[np.ndarray, float]:
    """Read an MP4 audio track through an ffmpeg pipe without writing a WAV."""
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise QwenTranscriptionError(
            f"ffmpeg audio decode failed for {path}: "
            f"{details[-1] if details else 'unknown error'}"
        )
    if not result.stdout or len(result.stdout) % 2:
        raise QwenTranscriptionError(
            f"ffmpeg returned empty or invalid PCM audio for: {path}"
        )
    waveform = (
        np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    )
    return waveform, len(waveform) / SAMPLE_RATE


def _timestamp_items(value: Any, audio: Path) -> list[dict[str, Any]]:
    if value is None:
        raise QwenTranscriptionError(f"forced aligner returned no timestamps for: {audio}")
    items: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, stamp in enumerate(value, 1):
        try:
            text = str(stamp.text)
            start = float(stamp.start_time)
            end = float(stamp.end_time)
        except (AttributeError, TypeError, ValueError) as exc:
            raise QwenTranscriptionError(
                f"invalid timestamp for {audio} at index {index}"
            ) from exc
        if (
            not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end - 1e-6
            or end < start
        ):
            raise QwenTranscriptionError(
                f"invalid timestamp values for {audio} at index {index}"
            )
        items.append({"text": text, "start_time": start, "end_time": end})
        previous_end = end
    if not items:
        raise QwenTranscriptionError(f"forced aligner returned empty timestamps for: {audio}")
    return items


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(document, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def transcribe_directory(
    root: Path,
    model_path: Path = DEFAULT_MODEL,
    aligner_path: Path = DEFAULT_ALIGNER,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    force: bool = False,
) -> list[Path]:
    """Transcribe every WAV or MP4 below ``root`` into an adjacent Qwen JSON file."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise QwenTranscriptionError(f"input directory not found: {root}")
    if device not in {"cpu", "cuda"}:
        raise QwenTranscriptionError("device must be cpu or cuda")
    if max_new_tokens <= 0:
        raise QwenTranscriptionError("max_new_tokens must be greater than zero")

    model_path = _model_directory(model_path, "Qwen3-ASR")
    aligner_path = _model_directory(aligner_path, "forced aligner")
    audio_files = _audio_files(root)
    outputs = [audio.with_name(f"{audio.stem}{OUTPUT_SUFFIX}") for audio in audio_files]
    if len(set(outputs)) != len(outputs):
        raise QwenTranscriptionError(
            "WAV and MP4 inputs with the same stem would overwrite one Qwen transcript"
        )
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise QwenTranscriptionError(
            "Qwen transcript already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    durations = {
        audio: _validate_wav(audio)
        for audio in audio_files
        if audio.suffix.lower() == ".wav"
    }
    ffmpeg = shutil.which("ffmpeg") if any(
        audio.suffix.lower() == ".mp4" for audio in audio_files
    ) else None
    if any(audio.suffix.lower() == ".mp4" for audio in audio_files) and ffmpeg is None:
        raise QwenTranscriptionError("ffmpeg is required for MP4 audio decoding")

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise QwenTranscriptionError(
            "Qwen ASR dependencies are missing; install dataset/requirements.txt"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise QwenTranscriptionError("CUDA was requested but is unavailable")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
        forced_aligner=str(aligner_path),
        forced_aligner_kwargs={"dtype": dtype, "device_map": device_map},
    )

    published: list[Path] = []
    for audio, output in zip(audio_files, outputs):
        if audio.suffix.lower() == ".mp4":
            assert ffmpeg is not None
            model_audio, audio_duration = _decode_mp4(audio, ffmpeg)
            asr_input: Any = (model_audio, SAMPLE_RATE)
        else:
            audio_duration = durations[audio]
            asr_input = str(audio)
        result = model.transcribe(
            audio=asr_input,
            language="Chinese",
            return_time_stamps=True,
        )
        if not isinstance(result, list) or len(result) != 1:
            raise QwenTranscriptionError(f"unexpected ASR result for: {audio}")
        text = str(result[0].text).strip()
        if not text:
            raise QwenTranscriptionError(f"Qwen ASR returned empty text for: {audio}")
        document = {
            "version": "qwen_asr_v1",
            "audio": audio.name,
            "audio_duration_sec": audio_duration,
            "language": "Chinese",
            "timeline": "original_audio_video",
            "time_unit": "second",
            "source": {
                "type": "automatic_asr",
                "asr_model": model_path.name,
                "forced_aligner": aligner_path.name,
            },
            "review": {"status": "pending"},
            "text": text,
            "timestamps": _timestamp_items(result[0].time_stamps, audio),
        }
        _write_json_atomic(output, document)
        published.append(output)
        print(f"Transcript: {output}", flush=True)
    return published


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe WAV or MP4 files with local Qwen3-ASR and forced alignment"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="directory containing WAV or MP4 files"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--aligner", type=Path, default=DEFAULT_ALIGNER)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--force", action="store_true", help="replace existing Qwen JSON files")
    args = parser.parse_args()
    try:
        outputs = transcribe_directory(
            args.input,
            args.model,
            args.aligner,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            force=args.force,
        )
    except (OSError, QwenTranscriptionError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Media files transcribed: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
