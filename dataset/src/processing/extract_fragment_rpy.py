"""Extract frame-aligned head/neck RPY for human-reviewed sentence fragments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.processing.extract_neck_rpy import AXES, DEFAULT_MODEL, _face_rotation, _rpy, _save_segment

ANNOTATION_SUFFIX = ".corrected_fragments.json"
OUTPUT_SUFFIX = ".neck_rpy"
_REVIEWED_STATUSES = {"human_corrected", "human_segmented"}


class FragmentRPYError(RuntimeError):
    """Raised when reviewed fragments or their video cannot be processed safely."""


@dataclass(frozen=True)
class ReviewedFragment:
    id: str
    start: float
    end: float
    duration: float
    text: str


def _wav_duration(path: Path) -> float:
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
        raise FragmentRPYError(f"could not read WAV: {path}") from exc
    if audio_format != (16000, 1, 2, "NONE") or frame_count <= 0:
        raise FragmentRPYError(f"WAV must be non-empty 16 kHz mono PCM s16: {path}")
    return frame_count / 16000.0


def load_reviewed_fragments(path: Path) -> tuple[str, list[ReviewedFragment]]:
    """Validate one human-reviewed corrected-fragments document."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentRPYError(f"could not read corrected fragments: {path}") from exc
    if not isinstance(document, dict):
        raise FragmentRPYError(f"corrected fragments root must be an object: {path}")

    if document.get("version") != "qwen_asr_v1_corrected_fragments":
        raise FragmentRPYError(f"unsupported corrected fragments version: {path}")

    review = document.get("review")
    status = review.get("status") if isinstance(review, dict) else None
    if status not in _REVIEWED_STATUSES:
        raise FragmentRPYError(
            f"corrected fragments are not marked as human reviewed: {path}"
        )
    if document.get("timeline") != "original_audio_video" or document.get("time_unit") != "second":
        raise FragmentRPYError(f"corrected fragments must use the original second timeline: {path}")

    audio_name = document.get("audio")
    if not isinstance(audio_name, str) or Path(audio_name).name != audio_name or not audio_name.lower().endswith(".wav"):
        raise FragmentRPYError(f"invalid audio filename in corrected fragments: {path}")
    audio_path = path.parent / audio_name
    if not audio_path.is_file():
        raise FragmentRPYError(f"referenced WAV not found: {audio_path}")
    actual_audio_duration = _wav_duration(audio_path)
    try:
        declared_audio_duration = float(document["audio_duration_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FragmentRPYError(f"invalid audio duration in corrected fragments: {path}") from exc
    if not math.isfinite(declared_audio_duration) or abs(declared_audio_duration - actual_audio_duration) > 1.5 / 16000:
        raise FragmentRPYError(f"audio duration mismatch in corrected fragments: {path}")

    raw_fragments = document.get("fragments")
    if not isinstance(raw_fragments, list) or not raw_fragments:
        raise FragmentRPYError(f"corrected fragments contain no fragments: {path}")
    if review.get("fragment_count") != len(raw_fragments):
        raise FragmentRPYError(f"review fragment_count does not match fragments: {path}")

    fragments: list[ReviewedFragment] = []
    identifiers: set[str] = set()
    previous_end = 0.0
    for index, item in enumerate(raw_fragments, 1):
        try:
            identifier = str(item["id"]).strip()
            start = float(item["start_time"])
            end = float(item["end_time"])
            duration = float(item["duration"])
            text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise FragmentRPYError(f"invalid fragment at {path}, index {index}") from exc
        if (
            not identifier
            or identifier in identifiers
            or Path(identifier).name != identifier
            or identifier in {".", ".."}
            or not text
            or not all(math.isfinite(value) for value in (start, end, duration))
            or start < previous_end - 1e-6
            or end <= start
            or end > actual_audio_duration + 1e-6
            or abs(duration - (end - start)) > 1e-3
        ):
            raise FragmentRPYError(f"invalid fragment values at {path}, index {index}")
        fragments.append(ReviewedFragment(identifier, start, end, duration, text))
        identifiers.add(identifier)
        previous_end = end
    return audio_name, fragments


def _fragment_index(timestamp: float, fragments: list[ReviewedFragment], start_index: int) -> int | None:
    index = start_index
    while index < len(fragments) and timestamp >= fragments[index].end - 1e-9:
        index += 1
    if index < len(fragments) and fragments[index].start - 1e-9 <= timestamp < fragments[index].end - 1e-9:
        return index
    return None


def extract_fragment_rpy(
    video: Path,
    annotations: Path,
    output: Path,
    model_path: Path = DEFAULT_MODEL,
    *,
    device: str = "gpu",
) -> dict[str, Any]:
    """Extract one video's reviewed fragments into an atomically published directory."""
    if device not in {"cpu", "gpu"}:
        raise FragmentRPYError("device must be cpu or gpu")
    audio_name, fragments = load_reviewed_fragments(annotations)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FragmentRPYError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise FragmentRPYError(f"video has invalid FPS or frame count: {video}")
    video_duration = frame_count / fps
    if fragments[-1].end > video_duration + 0.05:
        capture.release()
        raise FragmentRPYError(f"last fragment extends beyond video duration: {video}")

    delegate = (
        mp.tasks.BaseOptions.Delegate.GPU
        if device == "gpu"
        else mp.tasks.BaseOptions.Delegate.CPU
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    video_times: list[list[float]] = [[] for _ in fragments]
    values: list[list[np.ndarray]] = [[] for _ in fragments]
    neutral: np.ndarray | None = None
    current_fragment = 0
    decoded = 0
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            for frame_index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                timestamp = frame_index / fps
                index = _fragment_index(timestamp, fragments, current_fragment)
                while current_fragment < len(fragments) and timestamp >= fragments[current_fragment].end - 1e-9:
                    current_fragment += 1
                if frame_index != 0 and index is None:
                    continue
                rotation = _face_rotation(landmarker, frame, round(timestamp * 1000))
                if frame_index == 0:
                    if rotation is None:
                        raise FragmentRPYError(f"video frame zero has no valid face transform: {video}")
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
        raise FragmentRPYError(
            f"decoded {decoded} frames but video reports {frame_count}: {video}"
        )
    if neutral is None:
        raise FragmentRPYError(f"could not establish frame-zero reference: {video}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    summaries: list[dict[str, Any]] = []
    try:
        for fragment, times, fragment_values in zip(fragments, video_times, values):
            _save_segment(staging / fragment.id, times, fragment_values, fragment.start)
            valid_frames = sum(bool(np.isfinite(value).all()) for value in fragment_values)
            summaries.append(
                {
                    "id": fragment.id,
                    "start_time": fragment.start,
                    "end_time": fragment.end,
                    "duration": fragment.duration,
                    "text": fragment.text,
                    "frames": len(fragment_values),
                    "valid_frames": valid_frames,
                }
            )
        manifest = {
            "version": "fragment_neck_rpy_v1",
            "complete": True,
            "video": video.name,
            "audio": audio_name,
            "annotations": annotations.name,
            "reference": "video_frame_0",
            "fps": fps,
            "mediapipe_delegate": device,
            "order": list(AXES),
            "unit": "radian",
            "fragments": summaries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def process_directory(
    root: Path,
    model_path: Path = DEFAULT_MODEL,
    *,
    device: str = "gpu",
    force: bool = False,
) -> list[Path]:
    """Process every corrected-fragments JSON recursively below ``root``."""
    root = root.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not root.is_dir():
        raise FragmentRPYError(f"input directory not found: {root}")
    if device not in {"cpu", "gpu"}:
        raise FragmentRPYError("device must be cpu or gpu")
    if not model_path.is_file():
        raise FragmentRPYError(f"face landmarker model not found: {model_path}")
    annotations_files = sorted(root.rglob(f"*{ANNOTATION_SUFFIX}"))
    if not annotations_files:
        raise FragmentRPYError(f"no corrected fragments found in: {root}")

    jobs: list[tuple[Path, Path, Path]] = []
    for annotations in annotations_files:
        stem = annotations.name[: -len(ANNOTATION_SUFFIX)]
        video = annotations.parent / f"{stem}.mp4"
        output = annotations.parent / f"{stem}{OUTPUT_SUFFIX}"
        if not video.is_file():
            raise FragmentRPYError(f"matching MP4 not found: {video}")
        jobs.append((video, annotations, output))
    existing = [output for _video, _annotations, output in jobs if output.exists()]
    if existing and not force:
        raise FragmentRPYError(
            "RPY output already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    outputs: list[Path] = []
    for video, annotations, output in jobs:
        manifest = extract_fragment_rpy(
            video, annotations, output, model_path, device=device
        )
        outputs.append(output)
        valid = sum(item["valid_frames"] for item in manifest["fragments"])
        total = sum(item["frames"] for item in manifest["fragments"])
        print(
            f"RPY: {output} fragments={len(manifest['fragments'])} "
            f"valid_frames={valid}/{total}",
            flush=True,
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-fragment RPY from human-reviewed corrected fragments"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--force", action="store_true", help="replace existing RPY directories")
    args = parser.parse_args()
    try:
        outputs = process_directory(
            args.input, args.model, device=args.device, force=args.force
        )
    except (OSError, ValueError, FragmentRPYError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Videos processed: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
