"""MediaPipe Visual Features V1: video frames to unprocessed face features."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FEATURE_VERSION = "mediapipe_v1"
_REQUIRED_FILES = (
    "timestamps.npy",
    "valid.npy",
    "landmarks.npy",
    "blendshapes.npy",
    "transforms.npy",
    "metadata.json",
)


class MediaPipeFeatureError(RuntimeError):
    """Raised when MediaPipe V1 cannot be started or a clip is malformed."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MediaPipeFeatureError(f"invalid object at {path}:{line_number}")
                records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _completed_metadata(directory: Path) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in _REQUIRED_FILES):
        return None
    try:
        with (directory / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        if (
            metadata.get("status") != "completed"
            or metadata.get("feature_version") != FEATURE_VERSION
        ):
            return None
        frame_count = int(metadata["frame_count"])
        expected = {
            "timestamps.npy": ((frame_count,), np.dtype(np.float64)),
            "valid.npy": ((frame_count,), np.dtype(np.bool_)),
            "landmarks.npy": (
                (frame_count, int(metadata["landmark_count"]), 3), np.dtype(np.float32)
            ),
            "blendshapes.npy": (
                (frame_count, int(metadata["blendshape_count"])), np.dtype(np.float32)
            ),
            "transforms.npy": (
                (frame_count, *(int(value) for value in metadata["transform_shape"])),
                np.dtype(np.float32),
            ),
        }
        for name, (shape, dtype) in expected.items():
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _clean_metadata(input_dir: Path) -> dict[str, dict[str, Any]]:
    path = input_dir.parent / "metadata.jsonl"
    return {
        str(record["clip_id"]): record
        for record in _read_jsonl(path)
        if record.get("clip_id")
    }


def _timestamp_seconds(capture: cv2.VideoCapture, index: int, fps: float, previous: float) -> float:
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    if not math.isfinite(timestamp) or timestamp < 0 or (index and timestamp <= previous):
        timestamp = index / fps
    if index and timestamp <= previous:
        timestamp = previous + 1.0 / fps
    return timestamp


def _category_map(categories: Any) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for category in categories:
        name = getattr(category, "category_name", None)
        score = getattr(category, "score", None)
        if not name or name in values or score is None or not math.isfinite(float(score)):
            return None
        values[str(name)] = float(score)
    return values or None


def _frame_features(result: Any) -> tuple[np.ndarray, dict[str, float], np.ndarray] | None:
    landmarks_faces = getattr(result, "face_landmarks", None)
    blendshape_faces = getattr(result, "face_blendshapes", None)
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not landmarks_faces or not blendshape_faces or not matrices:
        return None
    if len(landmarks_faces) != 1 or len(blendshape_faces) != 1 or len(matrices) != 1:
        return None

    landmarks = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks_faces[0]], dtype=np.float32
    )
    blendshapes = _category_map(blendshape_faces[0])
    transform = np.asarray(matrices[0], dtype=np.float32)
    if (
        landmarks.ndim != 2
        or landmarks.shape[0] == 0
        or landmarks.shape[1] != 3
        or blendshapes is None
        or transform.ndim != 2
        or 0 in transform.shape
        or not np.isfinite(landmarks).all()
        or not np.isfinite(transform).all()
    ):
        return None
    return landmarks, blendshapes, transform


def _extract_clip(
    clip_path: Path,
    output_dir: Path,
    model_path: Path,
    clean_record: dict[str, Any] | None,
) -> dict[str, Any]:
    # Imported lazily so CLI help and unrelated commands do not require MediaPipe.
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except (ImportError, AttributeError) as exc:
        raise MediaPipeFeatureError(f"MediaPipe Tasks API is unavailable: {exc}") from exc

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        capture.release()
        raise MediaPipeFeatureError(f"could not open video: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise MediaPipeFeatureError(f"invalid video FPS: {fps}")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    timestamps: list[float] = []
    detected: list[tuple[np.ndarray, dict[str, float], np.ndarray] | None] = []
    landmark_shape: tuple[int, int] | None = None
    transform_shape: tuple[int, int] | None = None
    blendshape_names: list[str] | None = None
    previous_timestamp = -1.0
    mediapipe_timestamp_ms = -1

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = _timestamp_seconds(capture, frame_index, fps, previous_timestamp)
                previous_timestamp = timestamp
                timestamps.append(timestamp)
                timestamp_ms = max(int(round(timestamp * 1000.0)), mediapipe_timestamp_ms + 1)
                mediapipe_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                parsed = _frame_features(landmarker.detect_for_video(image, timestamp_ms))
                if parsed is not None:
                    landmarks, categories, transform = parsed
                    if landmark_shape is None:
                        landmark_shape = landmarks.shape
                        transform_shape = transform.shape
                        blendshape_names = list(categories)
                    if (
                        landmarks.shape != landmark_shape
                        or transform.shape != transform_shape
                        or set(categories) != set(blendshape_names or ())
                    ):
                        parsed = None
                detected.append(parsed)
                frame_index += 1
    finally:
        capture.release()

    frame_count = len(timestamps)
    if frame_count == 0:
        raise MediaPipeFeatureError("video contains no decodable frames")
    if reported_frame_count > 0 and frame_count != reported_frame_count:
        raise MediaPipeFeatureError(
            f"decoded frame count {frame_count} does not match container frame count "
            f"{reported_frame_count}"
        )
    if landmark_shape is None or transform_shape is None or blendshape_names is None:
        raise MediaPipeFeatureError("no frame produced a complete MediaPipe result")

    valid = np.zeros(frame_count, dtype=np.bool_)
    landmarks_array = np.full((frame_count, *landmark_shape), np.nan, dtype=np.float32)
    blendshapes_array = np.full((frame_count, len(blendshape_names)), np.nan, dtype=np.float32)
    transforms_array = np.full((frame_count, *transform_shape), np.nan, dtype=np.float32)
    for index, parsed in enumerate(detected):
        if parsed is None:
            continue
        landmarks, categories, transform = parsed
        coefficients = np.asarray([categories[name] for name in blendshape_names], dtype=np.float32)
        if not np.isfinite(coefficients).all():
            continue
        valid[index] = True
        landmarks_array[index] = landmarks
        blendshapes_array[index] = coefficients
        transforms_array[index] = transform

    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    arrays = (valid, landmarks_array, blendshapes_array, transforms_array)
    if any(array.shape[0] != frame_count for array in arrays):
        raise MediaPipeFeatureError("feature/frame alignment invariant failed")
    invalid = ~valid
    if (
        not np.isnan(landmarks_array[invalid]).all()
        or not np.isnan(blendshapes_array[invalid]).all()
        or not np.isnan(transforms_array[invalid]).all()
    ):
        raise MediaPipeFeatureError("invalid frames do not contain NaN features")

    valid_frames = int(valid.sum())
    duration = float(timestamps_array[-1] + (1.0 / fps))
    clip_id = clip_path.stem
    source_video_id = (
        str(clean_record["source_video_id"])
        if clean_record and clean_record.get("source_video_id")
        else clip_id.rsplit("_", 1)[0]
    )
    source_path = (
        str(clean_record["source_path"])
        if clean_record and clean_record.get("source_path")
        else str(clip_path.resolve())
    )
    metadata: dict[str, Any] = {
        "clip_id": clip_id,
        "source_video_id": source_video_id,
        "source_path": source_path,
        "clip_path": str(clip_path.resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "valid_frames": valid_frames,
        "invalid_frames": frame_count - valid_frames,
        "valid_ratio": valid_frames / frame_count,
        "landmark_count": landmark_shape[0],
        "landmark_coordinate_system": "MediaPipe Face Landmarker normalized x/y and relative z (unmodified)",
        "blendshape_count": len(blendshape_names),
        "blendshape_names": blendshape_names,
        "transform_shape": list(transform_shape),
        "mediapipe_model": str(model_path.resolve()),
        "feature_version": FEATURE_VERSION,
        "status": "completed",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output_dir.parent))
    try:
        np.save(temporary / "timestamps.npy", timestamps_array, allow_pickle=False)
        np.save(temporary / "valid.npy", valid, allow_pickle=False)
        np.save(temporary / "landmarks.npy", landmarks_array, allow_pickle=False)
        np.save(temporary / "blendshapes.npy", blendshapes_array, allow_pickle=False)
        np.save(temporary / "transforms.npy", transforms_array, allow_pickle=False)
        with (temporary / "metadata.json").open("w", encoding="utf-8") as target:
            json.dump(metadata, target, ensure_ascii=False, indent=2)
            target.write("\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def _global_record(metadata: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    keys = (
        "clip_id", "source_video_id", "source_path", "clip_path", "fps",
        "frame_count", "duration", "valid_frames", "invalid_frames", "valid_ratio",
    )
    return {**{key: metadata[key] for key in keys}, "feature_path": str(feature_path.resolve())}


def _print_clip(metadata: dict[str, Any], status: str) -> None:
    print(f"Frames: {metadata['frame_count']}")
    print(f"Valid: {metadata['valid_frames']}")
    print(f"Invalid: {metadata['invalid_frames']}")
    print(f"Valid ratio: {float(metadata['valid_ratio']):.2%}")
    print(f"Status: {status}", flush=True)


def extract_mediapipe_clips(
    clip_paths: list[Path], output_root: Path, model_path: Path, force: bool = False
) -> dict[str, Any]:
    """Extract raw Face Landmarker outputs with clip-level resume and isolation."""
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise MediaPipeFeatureError(f"Face Landmarker model not found: {model_path}")
    output_root = output_root.resolve()
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"
    failed_path = output_root / "failed.jsonl"
    successful = {str(item["clip_id"]): item for item in _read_jsonl(metadata_path) if item.get("clip_id")}
    failures = {str(item["clip_id"]): item for item in _read_jsonl(failed_path) if item.get("clip_id")}
    clean_records = _clean_metadata(clip_paths[0].parent) if clip_paths else {}

    processed = skipped = failed = 0
    total_frames = valid_frames = 0
    for index, clip_path in enumerate(clip_paths, start=1):
        clip_path = clip_path.resolve()
        clip_id = clip_path.stem
        feature_path = clips_root / clip_id
        print(f"[{index:03d}/{len(clip_paths):03d}] {clip_path.name}", flush=True)
        completed = _completed_metadata(feature_path)
        if completed is not None and not force:
            skipped += 1
            successful[clip_id] = _global_record(completed, feature_path)
            failures.pop(clip_id, None)
            total_frames += int(completed["frame_count"])
            valid_frames += int(completed["valid_frames"])
            _print_clip(completed, "skipped")
            continue
        try:
            metadata = _extract_clip(clip_path, feature_path, model_path, clean_records.get(clip_id))
            successful[clip_id] = _global_record(metadata, feature_path)
            failures.pop(clip_id, None)
            processed += 1
            total_frames += int(metadata["frame_count"])
            valid_frames += int(metadata["valid_frames"])
            _print_clip(metadata, "done")
        except Exception as exc:
            failed += 1
            successful.pop(clip_id, None)
            failures[clip_id] = {
                "clip_id": clip_id,
                "source_video_id": clip_id.rsplit("_", 1)[0],
                "source_path": str(clip_path),
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "feature_version": FEATURE_VERSION,
            }
            print("Status: failed")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        _write_jsonl(metadata_path, list(successful.values()))
        _write_jsonl(failed_path, list(failures.values()))

    # Also materialize empty index files on a first run with no failures.
    _write_jsonl(metadata_path, list(successful.values()))
    _write_jsonl(failed_path, list(failures.values()))
    invalid_frames = total_frames - valid_frames
    return {
        "input_clips": len(clip_paths),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "invalid_frames": invalid_frames,
        "valid_ratio": valid_frames / total_frames if total_frames else 0.0,
    }
