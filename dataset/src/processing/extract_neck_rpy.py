"""Extract frame-aligned head/neck RPY for manual sentence annotations."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.processing.manual_annotations import ManualSegment, load_annotations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/mediapipe/face_landmarker.task"
AXES = ("roll", "pitch", "yaw")


def _rotation(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("face transform must be a finite 4x4 matrix")
    u, _, vh = np.linalg.svd(value[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _rpy(rotation: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw] for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _face_rotation(landmarker: Any, frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect_for_video(image, timestamp_ms)
    matrices = result.facial_transformation_matrixes
    if len(matrices) != 1:
        return None
    try:
        return _rotation(matrices[0])
    except (ValueError, np.linalg.LinAlgError):
        return None


def _segment_index(timestamp: float, segments: list[ManualSegment], start_index: int) -> int | None:
    index = start_index
    while index < len(segments) and timestamp >= segments[index].end - 1e-9:
        index += 1
    if index < len(segments) and segments[index].start - 1e-9 <= timestamp < segments[index].end - 1e-9:
        return index
    return None


def _save_segment(
    directory: Path,
    video_times: list[float],
    values: list[np.ndarray],
    start: float,
) -> None:
    times = np.asarray(video_times, dtype=np.float64)
    rpy = np.asarray(values, dtype=np.float32)
    if rpy.shape != (len(times), 3) or len(times) == 0:
        raise RuntimeError("annotation segment produced no frame-aligned RPY data")
    valid = np.isfinite(rpy).all(axis=1)
    local_times = times - start
    directory.mkdir()
    np.savez_compressed(
        directory / "rpy.npz",
        video_timestamps=times,
        local_timestamps=local_times,
        valid=valid,
        values=rpy,
        order=np.asarray(AXES),
        unit=np.asarray("radian"),
    )
    for column, axis in enumerate(AXES):
        np.savez_compressed(
            directory / f"{axis}.npz",
            video_timestamps=times,
            local_timestamps=local_times,
            valid=valid,
            values=rpy[:, column],
            unit=np.asarray("radian"),
        )


def extract(video: Path, annotations_path: Path, output: Path, model_path: Path) -> list[dict[str, Any]]:
    segments = load_annotations(annotations_path, validate_audio=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video has invalid FPS or frame count")
    duration = frame_count / fps
    if segments[-1].end > duration + 0.05:
        capture.release()
        raise ValueError("manual annotation extends beyond video duration")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    video_times: list[list[float]] = [[] for _ in segments]
    values: list[list[np.ndarray]] = [[] for _ in segments]
    neutral: np.ndarray | None = None
    current_segment = 0
    decoded = 0
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            for frame_index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                timestamp = frame_index / fps
                index = _segment_index(timestamp, segments, current_segment)
                while current_segment < len(segments) and timestamp >= segments[current_segment].end - 1e-9:
                    current_segment += 1
                if frame_index != 0 and index is None:
                    continue
                rotation = _face_rotation(landmarker, frame, round(timestamp * 1000))
                if frame_index == 0:
                    if rotation is None:
                        raise RuntimeError("video frame zero has no valid face transform")
                    neutral = rotation
                if index is not None:
                    video_times[index].append(timestamp)
                    values[index].append(
                        np.full(3, np.nan, dtype=np.float32)
                        if rotation is None
                        else _rpy(neutral.T @ rotation)
                    )
    finally:
        capture.release()
    if decoded != frame_count:
        raise RuntimeError(f"decoded {decoded} frames but video reports {frame_count}")
    assert neutral is not None

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    summary: list[dict[str, Any]] = []
    try:
        for segment, times, segment_values in zip(segments, video_times, values):
            _save_segment(staging / segment.id, times, segment_values, segment.start)
            valid_frames = sum(bool(np.isfinite(value).all()) for value in segment_values)
            summary.append(
                {
                    "id": segment.id,
                    "start_sec": segment.start,
                    "end_sec": segment.end,
                    "frames": len(segment_values),
                    "valid_frames": valid_frames,
                }
            )
        with (staging / "manifest.json").open("w", encoding="utf-8") as target:
            json.dump(
                {
                    "video": video.name,
                    "annotations": annotations_path.name,
                    "reference": "video_frame_0",
                    "order": list(AXES),
                    "unit": "radian",
                    "segments": summary,
                },
                target,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            target.write("\n")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RPY for manually annotated segments")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path, help="manual metadata.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    video = args.video.resolve()
    annotations = args.annotations.resolve()
    model = args.model.resolve()
    for label, path in (("video", video), ("manual annotations", annotations), ("face model", model)):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    summary = extract(video, annotations, args.output.resolve(), model)
    print(f"Segments: {len(summary)}")
    print(f"Frames: {sum(item['frames'] for item in summary)}")
    print(f"Valid frames: {sum(item['valid_frames'] for item in summary)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
