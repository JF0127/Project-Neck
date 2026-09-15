"""Create one reviewable rough-transcript JSON for each local video/audio pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "asr"
_MODEL_DIRECTORIES = {
    "asr_model": "paraformer-zh",
    "vad_model": "fsmn-vad",
    "punctuation_model": "ct-punc",
}


class RoughTranscriptionError(RuntimeError):
    """Raised when local ASR output cannot be safely published."""


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            audio_format = (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getcomptype(),
            )
            frames = source.getnframes()
    except (OSError, wave.Error) as exc:
        raise RoughTranscriptionError(f"could not read WAV: {path}") from exc
    if audio_format != (16000, 1, 2, "NONE") or frames <= 0:
        raise RoughTranscriptionError(
            f"WAV must be non-empty 16 kHz mono PCM s16: {path}"
        )
    return frames / 16000.0


def _video_audio_pairs(root: Path) -> list[tuple[Path, Path, Path]]:
    videos = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise RoughTranscriptionError(f"no top-level MP4 files found in: {root}")

    pairs: list[tuple[Path, Path, Path]] = []
    for video in videos:
        audio = root / f"{video.stem}.wav"
        if not audio.is_file():
            raise RoughTranscriptionError(f"matching WAV not found for: {video.name}")
        pairs.append((video, audio, root / f"{video.stem}.json"))
    return pairs


def _check_models(model_root: Path) -> dict[str, Path]:
    paths = {
        name: (model_root / directory).resolve()
        for name, directory in _MODEL_DIRECTORIES.items()
    }
    required_files = {
        "asr_model": ("model.pt", "config.yaml", "tokens.json", "seg_dict", "am.mvn"),
        "vad_model": ("model.pt", "config.yaml", "am.mvn"),
        "punctuation_model": ("model.pt", "config.yaml", "tokens.json"),
    }
    missing = [
        str(path / filename)
        for name, path in paths.items()
        for filename in required_files[name]
        if not (path / filename).is_file()
    ]
    if missing:
        raise RoughTranscriptionError(
            "local ASR model files are missing: " + ", ".join(missing)
        )
    return paths


def _segments(result: dict[str, Any], stem: str, audio_duration: float) -> list[dict[str, Any]]:
    sentence_info = result.get("sentence_info")
    if not isinstance(sentence_info, list) or not sentence_info:
        raise RoughTranscriptionError(f"ASR returned no sentence timestamps for: {stem}")

    segments: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, sentence in enumerate(sentence_info, 1):
        try:
            text = str(sentence["text"]).strip()
            start = float(sentence["start"]) / 1000.0
            end = float(sentence["end"]) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            raise RoughTranscriptionError(
                f"invalid sentence timestamp returned for {stem} at index {index}"
            ) from exc
        if (
            not text
            or not all(math.isfinite(value) for value in (start, end))
            or start < previous_end - 1e-6
            or end <= start
            or end > audio_duration + 0.1
        ):
            raise RoughTranscriptionError(
                f"invalid sentence values returned for {stem} at index {index}"
            )
        segments.append(
            {
                "id": f"{stem}_{index:03d}",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": text,
            }
        )
        previous_end = end
    return segments


def _write_json(path: Path, document: dict[str, Any]) -> None:
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
    model_root: Path = DEFAULT_MODEL_ROOT,
    *,
    device: str = "auto",
    force: bool = False,
) -> list[Path]:
    """Transcribe top-level MP4/WAV pairs into per-video pending-review JSON."""
    root = root.resolve()
    if not root.is_dir():
        raise RoughTranscriptionError(f"input directory not found: {root}")
    if device not in {"auto", "cpu", "cuda"}:
        raise RoughTranscriptionError("device must be auto, cpu, or cuda")

    pairs = _video_audio_pairs(root)
    existing = [output for _video, _audio, output in pairs if output.exists()]
    if existing and not force:
        raise RoughTranscriptionError(
            "transcript JSON already exists; refusing to overwrite: "
            + ", ".join(path.name for path in existing)
        )
    durations = {audio: _audio_duration(audio) for _video, audio, _output in pairs}
    models = _check_models(model_root.resolve())

    try:
        import torch
        from funasr import AutoModel
    except ImportError as exc:
        raise RoughTranscriptionError(
            "FunASR dependencies are missing; install dataset/requirements.txt"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RoughTranscriptionError("CUDA was requested but is unavailable")
    actual_device = "cuda:0" if device == "cuda" or (
        device == "auto" and torch.cuda.is_available()
    ) else "cpu"
    model = AutoModel(
        model=str(models["asr_model"]),
        vad_model=str(models["vad_model"]),
        punc_model=str(models["punctuation_model"]),
        device=actual_device,
        disable_update=True,
    )

    outputs: list[Path] = []
    for video, audio, output in pairs:
        result = model.generate(
            input=str(audio),
            batch_size_s=300,
            sentence_timestamp=True,
        )
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
            raise RoughTranscriptionError(f"unexpected ASR result for: {audio.name}")
        document = {
            "version": "rough_transcript_v1",
            "video": video.name,
            "audio": audio.name,
            "language": "zh",
            "timeline": "original_audio_video",
            "time_unit": "second",
            "source": {
                "type": "automatic_asr",
                **_MODEL_DIRECTORIES,
            },
            "review": {"status": "pending"},
            "segments": _segments(result[0], video.stem, durations[audio]),
        }
        _write_json(output, document)
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create reviewable sentence-timestamp JSON using local FunASR models"
    )
    parser.add_argument("--input", required=True, type=Path, help="MP4/WAV directory")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace existing transcript JSON")
    args = parser.parse_args()
    try:
        outputs = transcribe_directory(
            args.input,
            args.model_root,
            device=args.device,
            force=args.force,
        )
    except (OSError, RoughTranscriptionError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for output in outputs:
        print(f"Transcript: {output}")
    print(f"Videos transcribed: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
