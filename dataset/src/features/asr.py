"""Speech V1 Chinese ASR, timestamp validation, resume, and batch orchestration."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.features.audio import (
    AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
    AudioExtractionError,
    extract_standard_audio,
    probe_video_duration,
    probe_wav,
    validate_audio_alignment,
)

FEATURE_VERSION = "speech_v1"
DEFAULT_MODEL = "large-v3"
LANGUAGE = "zh"
TIMESTAMP_TOLERANCE_SECONDS = 0.10
_REQUIRED_FILES = ("audio.wav", "transcript.json", "metadata.json")


class ASRError(RuntimeError):
    """Raised when ASR initialization, transcription, or output conversion fails."""


class SpeechStageError(RuntimeError):
    """A clip-local error carrying its failed Speech V1 stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def select_device(
    requested_device: str | None = None, requested_compute_type: str | None = None
) -> tuple[str, str]:
    """Select CUDA through CTranslate2 capability checks, without importing PyTorch."""
    normalized_device = requested_device.lower() if requested_device else None
    if normalized_device and normalized_device != "auto":
        device = normalized_device
    else:
        device = "cpu"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except (ImportError, RuntimeError, OSError, AttributeError):
            device = "cpu"
    compute_type = requested_compute_type or ("float16" if device == "cuda" else "int8")
    return device, compute_type


def create_asr_model(model_name: str, device: str, compute_type: str) -> Any:
    """Create faster-whisper lazily so help/tests do not download or load a model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRError(f"faster-whisper is unavailable: {exc}") from exc
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise ASRError(f"could not initialize model {model_name}: {exc}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def transcribe_audio(model: Any, wav_path: Path, clip_id: str) -> dict[str, Any]:
    """Transcribe one standardized WAV and preserve faster-whisper word units verbatim."""
    try:
        segment_iterator, info = model.transcribe(
            str(wav_path),
            language=LANGUAGE,
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
        )
        segments: list[dict[str, Any]] = []
        flat_words: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for fallback_id, segment in enumerate(segment_iterator):
            segment_id = int(segment.id) if getattr(segment, "id", None) is not None else fallback_id
            segment_words: list[dict[str, Any]] = []
            for word in getattr(segment, "words", None) or []:
                item: dict[str, Any] = {
                    "text": str(getattr(word, "word", "")),
                    "start": _optional_float(getattr(word, "start", None)),
                    "end": _optional_float(getattr(word, "end", None)),
                }
                probability = _optional_float(getattr(word, "probability", None))
                if probability is not None:
                    item["probability"] = probability
                segment_words.append(item)
                flat_words.append({**item, "segment_id": segment_id})
            segment_text = str(getattr(segment, "text", ""))
            text_parts.append(segment_text)
            segments.append(
                {
                    "id": segment_id,
                    "start": _optional_float(getattr(segment, "start", None)),
                    "end": _optional_float(getattr(segment, "end", None)),
                    "text": segment_text,
                    "words": segment_words,
                }
            )
    except Exception as exc:
        if isinstance(exc, ASRError):
            raise
        raise ASRError(f"faster-whisper transcription failed: {exc}") from exc
    detected_language = getattr(info, "language", None)
    language_probability = _optional_float(getattr(info, "language_probability", None))
    transcript: dict[str, Any] = {
        "clip_id": clip_id,
        "language": LANGUAGE,
        "text": "".join(text_parts),
        "segments": segments,
        "words": flat_words,
    }
    if detected_language is not None:
        transcript["detected_language"] = str(detected_language)
    if language_probability is not None:
        transcript["language_probability"] = language_probability
    return transcript


def _validate_timed_items(
    items: list[dict[str, Any]], label: str, duration: float, tolerance: float
) -> None:
    previous_start = -math.inf
    previous_end = -math.inf
    for index, item in enumerate(items):
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ASRError(f"{label} {index} has missing or invalid timestamps") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ASRError(f"{label} {index} has non-finite timestamps")
        if start < 0.0 or end < start or end > duration + tolerance:
            raise ASRError(
                f"{label} {index} timestamp [{start:.6f}, {end:.6f}] is outside "
                f"clip-local duration {duration:.6f}s"
            )
        if start < previous_start - 1e-9 or end < previous_end - 1e-9:
            raise ASRError(f"{label} timestamps are not monotonically ordered at index {index}")
        previous_start, previous_end = start, end


def validate_transcript(
    transcript: dict[str, Any], audio_duration: float,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> None:
    """Validate clip-local segment/word boundaries and monotonic ordering."""
    if not isinstance(transcript, dict) or not isinstance(transcript.get("text"), str):
        raise ASRError("transcript must contain complete text")
    segments = transcript.get("segments")
    words = transcript.get("words")
    if not isinstance(segments, list) or not isinstance(words, list):
        raise ASRError("transcript segments and words must be lists")
    _validate_timed_items(segments, "segment", audio_duration, tolerance)
    _validate_timed_items(words, "word", audio_duration, tolerance)
    for segment_index, segment in enumerate(segments):
        segment_words = segment.get("words")
        if not isinstance(segment_words, list) or not isinstance(segment.get("text"), str):
            raise ASRError(f"segment {segment_index} has invalid text or words")
        _validate_timed_items(segment_words, f"segment {segment_index} word", audio_duration, tolerance)
        for word_index, word in enumerate(segment_words):
            if not isinstance(word.get("text"), str):
                raise ASRError(f"segment {segment_index} word {word_index} has invalid text")
    for word_index, word in enumerate(words):
        if not isinstance(word.get("text"), str) or "segment_id" not in word:
            raise ASRError(f"word {word_index} has invalid text or segment_id")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize strict JSON with UTF-8 Chinese text preserved."""
    with path.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def _neck_alignment(neck_root: Path | None, clip_id: str, audio_duration: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "tolerance_seconds": AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
        "alignment_error": False,
    }
    if neck_root is None:
        return result
    timestamps_path = neck_root / clip_id / "timestamps.npy"
    if not timestamps_path.is_file():
        result["warning"] = "corresponding Neck Pose V1 timestamps not found"
        return result
    try:
        timestamps = np.load(timestamps_path, allow_pickle=False)
        if timestamps.ndim != 1 or len(timestamps) == 0 or not np.isfinite(timestamps).all():
            raise ValueError("invalid neck timestamps array")
        neck_end = float(timestamps[-1])
        difference = neck_end - audio_duration
        result.update(
            {
                "available": True,
                "neck_last_timestamp": neck_end,
                "audio_duration": audio_duration,
                "difference_seconds": difference,
                "alignment_error": abs(difference) > AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
            }
        )
        if result["alignment_error"]:
            result["warning"] = "neck last timestamp and audio duration differ by more than 0.10s"
    except (OSError, ValueError) as exc:
        result.update({"alignment_error": True, "warning": f"neck alignment check failed: {exc}"})
    return result


def is_complete_speech_output(directory: Path) -> dict[str, Any] | None:
    """Return completed metadata only after basic WAV and transcript checks pass."""
    if not all((directory / name).is_file() for name in _REQUIRED_FILES):
        return None
    try:
        with (directory / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        with (directory / "transcript.json").open("r", encoding="utf-8") as source:
            transcript = json.load(source)
        if metadata.get("status") != "completed" or metadata.get("feature_version") != FEATURE_VERSION:
            return None
        audio = probe_wav(directory / "audio.wav")
        validate_transcript(transcript, float(audio["duration"]))
        metadata_audio = metadata["audio"]
        metadata_asr = metadata["asr"]
        if (
            metadata.get("clip_id") != directory.name
            or transcript.get("clip_id") != directory.name
            or int(metadata_audio["sample_rate"]) != audio["sample_rate"]
            or int(metadata_audio["channels"]) != audio["channels"]
            or metadata_audio["format"] != audio["format"]
            or abs(float(metadata_audio["duration"]) - float(audio["duration"])) > 1e-6
            or int(metadata_asr["segment_count"]) != len(transcript["segments"])
            or int(metadata_asr["word_count"]) != len(transcript["words"])
        ):
            return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, AudioExtractionError, ASRError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ASRError(f"invalid JSONL record in {path}")
                records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _global_record(metadata: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    return {
        "clip_id": metadata["clip_id"],
        "feature_path": str(feature_path.resolve()),
        "audio_duration": metadata["audio"]["duration"],
        "video_duration": metadata["video_duration"],
        "duration_difference": metadata["duration_difference"],
        "language": metadata["asr"]["language"],
        "segment_count": metadata["asr"]["segment_count"],
        "word_count": metadata["asr"]["word_count"],
        "asr_model": metadata["asr"]["model"],
    }


def process_speech_clip(
    video_path: Path,
    output_dir: Path,
    model: Any,
    model_name: str,
    device: str,
    compute_type: str,
    neck_root: Path | None = None,
) -> dict[str, Any]:
    """Extract standardized audio and direct Chinese ASR for one Clean V1 clip."""
    clip_id = video_path.stem
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output_dir.parent))
    try:
        try:
            video_duration = probe_video_duration(video_path)
            audio = extract_standard_audio(video_path, temporary / "audio.wav")
        except AudioExtractionError as exc:
            raise SpeechStageError("audio", str(exc)) from exc
        try:
            duration_difference = validate_audio_alignment(video_duration, float(audio["duration"]))
        except AudioExtractionError as exc:
            raise SpeechStageError("validation", str(exc)) from exc
        try:
            transcript = transcribe_audio(model, temporary / "audio.wav", clip_id)
        except ASRError as exc:
            raise SpeechStageError("asr", str(exc)) from exc
        try:
            validate_transcript(transcript, float(audio["duration"]))
        except ASRError as exc:
            raise SpeechStageError("validation", str(exc)) from exc
        write_json(temporary / "transcript.json", transcript)
        neck_alignment = _neck_alignment(neck_root, clip_id, float(audio["duration"]))
        metadata: dict[str, Any] = {
            "clip_id": clip_id,
            "source_video": str(video_path.resolve()),
            "audio": {
                "sample_rate": audio["sample_rate"],
                "channels": audio["channels"],
                "sample_width_bits": audio["sample_width_bits"],
                "format": audio["format"],
                "duration": audio["duration"],
            },
            "video_duration": video_duration,
            "duration_difference": duration_difference,
            "asr": {
                "engine": "faster-whisper",
                "model": model_name,
                "language": LANGUAGE,
                "task": "transcribe",
                "beam_size": 5,
                "word_timestamps": True,
                "device": device,
                "compute_type": compute_type,
                "segment_count": len(transcript["segments"]),
                "word_count": len(transcript["words"]),
            },
            "text_length": len(transcript["text"]),
            "neck_pose_alignment": neck_alignment,
            "feature_version": FEATURE_VERSION,
            "status": "completed",
        }
        write_json(temporary / "metadata.json", metadata)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
        return metadata
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _print_clip(metadata: dict[str, Any], status: str) -> None:
    print("Audio:")
    print(f"  duration: {metadata['audio']['duration']:.2f}s")
    print(f"  sample rate: {metadata['audio']['sample_rate']}")
    print(f"  channels: {metadata['audio']['channels']}")
    print("ASR:")
    print(f"  language: {metadata['asr']['language']}")
    print(f"  segments: {metadata['asr']['segment_count']}")
    print(f"  words: {metadata['asr']['word_count']}")
    alignment = metadata.get("neck_pose_alignment", {})
    if alignment.get("warning"):
        print(f"Warning: {alignment['warning']}")
    print(f"Status: {status}", flush=True)


def extract_speech_clips(
    video_paths: list[Path],
    output_root: Path,
    model_name: str = DEFAULT_MODEL,
    requested_device: str | None = None,
    requested_compute_type: str | None = None,
    force: bool = False,
    neck_root: Path | None = None,
) -> dict[str, Any]:
    """Run Speech V1 with one shared ASR model and clip-level failure isolation."""
    output_root = output_root.resolve()
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"
    failed_path = output_root / "failed.jsonl"
    successful = {str(item["clip_id"]): item for item in _read_jsonl(metadata_path) if item.get("clip_id")}
    failures = {str(item["clip_id"]): item for item in _read_jsonl(failed_path) if item.get("clip_id")}
    device, compute_type = select_device(requested_device, requested_compute_type)
    model: Any | None = None
    model_initialization_error: ASRError | None = None
    processed = skipped = failed = 0
    total_duration = 0.0
    total_segments = total_words = 0

    for index, video_path in enumerate(video_paths, start=1):
        video_path = video_path.resolve()
        clip_id = video_path.stem
        output_dir = clips_root / clip_id
        print(f"[{index:03d}/{len(video_paths):03d}] {video_path.name}", flush=True)
        completed = is_complete_speech_output(output_dir)
        if completed is not None and not force:
            skipped += 1
            successful[clip_id] = _global_record(completed, output_dir)
            failures.pop(clip_id, None)
            total_duration += float(completed["audio"]["duration"])
            total_segments += int(completed["asr"]["segment_count"])
            total_words += int(completed["asr"]["word_count"])
            _print_clip(completed, "skipped")
            continue
        try:
            if model is None:
                if model_initialization_error is not None:
                    raise SpeechStageError("asr", str(model_initialization_error))
                try:
                    model = create_asr_model(model_name, device, compute_type)
                except ASRError as exc:
                    model_initialization_error = exc
                    raise SpeechStageError("asr", str(exc)) from exc
            metadata = process_speech_clip(
                video_path, output_dir, model, model_name, device, compute_type, neck_root
            )
            successful[clip_id] = _global_record(metadata, output_dir)
            failures.pop(clip_id, None)
            processed += 1
            total_duration += float(metadata["audio"]["duration"])
            total_segments += int(metadata["asr"]["segment_count"])
            total_words += int(metadata["asr"]["word_count"])
            _print_clip(metadata, "done")
        except Exception as exc:
            failed += 1
            stage = exc.stage if isinstance(exc, SpeechStageError) else "validation"
            successful.pop(clip_id, None)
            failures[clip_id] = {
                "clip_id": clip_id,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
            print("Status: failed")
            print(f"Stage: {stage}")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        _write_jsonl(metadata_path, list(successful.values()))
        _write_jsonl(failed_path, list(failures.values()))

    _write_jsonl(metadata_path, list(successful.values()))
    _write_jsonl(failed_path, list(failures.values()))
    return {
        "input_clips": len(video_paths),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_audio_duration": total_duration,
        "total_segments": total_segments,
        "total_words": total_words,
        "device": device,
        "compute_type": compute_type,
    }
