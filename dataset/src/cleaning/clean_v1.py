"""Clean V1: shot detection and stable upper-body clip extraction."""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scenedetect import ContentDetector, SceneManager, open_video
from ultralytics import YOLO

from src.crawler.storage import load_jsonl, save_jsonl

_FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
_POSE_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"
)
_REJECT_REASONS = {
    "too_short",
    "no_person",
    "multiple_people",
    "no_face",
    "multiple_faces",
    "face_too_small",
    "shoulders_not_visible",
    "unstable_detection",
    "crop_invalid",
    "export_failed",
}


class CleanError(RuntimeError):
    """Raised when Clean V1 cannot be started."""


def _download_model(url: str, cache_path: Path, minimum_size: int) -> Path:
    if cache_path.is_file() and cache_path.stat().st_size > minimum_size:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size <= minimum_size:
            raise CleanError(f"downloaded model is unexpectedly small: {cache_path.name}")
        temporary.replace(cache_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, CleanError):
            raise
        raise CleanError(f"could not download model {cache_path.name}: {exc}") from exc
    return cache_path


def _face_model_path() -> Path:
    cache_path = Path.home() / ".cache/youtube_crawler/face_detection_yunet_2023mar.onnx"
    return _download_model(_FACE_MODEL_URL, cache_path, 100_000)


def _pose_model_path() -> Path:
    cache_path = Path.home() / ".cache/youtube_crawler/yolo11n-pose.pt"
    return _download_model(_POSE_MODEL_URL, cache_path, 1_000_000)


def _detect_scenes(path: Path) -> list[tuple[float, float]]:
    video = open_video(str(path))
    fps = float(video.frame_rate)
    manager = SceneManager()
    # A moderately conservative threshold avoids treating normal presenter motion as a cut.
    manager.add_detector(ContentDetector(threshold=30.0, min_scene_len=max(15, round(fps))))
    manager.detect_scenes(video, show_progress=False)
    return [
        (start.get_seconds(), end.get_seconds())
        for start, end in manager.get_scene_list(start_in_scene=True)
    ]


def _sample_times(start: float, end: float, sample_fps: float) -> list[float]:
    count = max(1, math.ceil((end - start) * sample_fps))
    step = (end - start) / count
    return [start + (index + 0.5) * step for index in range(count)]


def _faces(detector: Any, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(frame)
    if detections is None:
        return []
    return [tuple(float(value) for value in row[:4]) for row in detections]


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _stable_crop(
    observations: list[dict[str, Any]], frame_width: int, frame_height: int
) -> dict[str, int] | None:
    if not observations:
        return None
    face_heights = np.array([item["face"][3] for item in observations])
    face_tops = np.array([item["face"][1] for item in observations])
    lefts = np.array(
        [min(item["face"][0], item["left_shoulder"][0], item["right_shoulder"][0]) for item in observations]
    )
    rights = np.array(
        [
            max(
                item["face"][0] + item["face"][2],
                item["left_shoulder"][0],
                item["right_shoulder"][0],
            )
            for item in observations
        ]
    )
    shoulder_y = np.array(
        [max(item["left_shoulder"][1], item["right_shoulder"][1]) for item in observations]
    )
    face_height = float(np.median(face_heights))
    left = float(np.percentile(lefts, 10)) - 0.12 * face_height
    right = float(np.percentile(rights, 90)) + 0.12 * face_height
    top = float(np.percentile(face_tops, 10)) - 0.18 * face_height
    bottom = float(np.percentile(shoulder_y, 90)) + 1.45 * face_height

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    width = max(right - left, 2.8 * face_height)
    height = max(bottom - top, width / 0.8)  # Unified 4:5 crop.
    width = max(width, height * 0.8)
    if width > frame_width or height > frame_height:
        scale = min(frame_width / width, frame_height / height)
        width *= scale
        height *= scale
    width, height = _even(width), _even(height)
    if width < 64 or height < 64:
        return None
    x = int(round(center_x - width / 2))
    y = int(round(center_y - height / 2))
    x = min(max(0, x), frame_width - width)
    y = min(max(0, y), frame_height - height)
    x, y = x // 2 * 2, y // 2 * 2
    return {"x": x, "y": y, "w": width, "h": height}


def _reject_reason(
    person_counts: list[int],
    face_counts: list[int],
    single_person_ratio: float,
    single_face_ratio: float,
    shoulder_ratio: float,
    face_size_ratio: float,
) -> str | None:
    if single_person_ratio < 0.90:
        if sum(count > 1 for count in person_counts) / len(person_counts) > 0.10:
            return "multiple_people"
        if sum(count == 0 for count in person_counts) / len(person_counts) >= 0.50:
            return "no_person"
        return "unstable_detection"
    if single_face_ratio < 0.90:
        if sum(count > 1 for count in face_counts) / len(face_counts) > 0.10:
            return "multiple_faces"
        if sum(count == 0 for count in face_counts) / len(face_counts) >= 0.50:
            return "no_face"
        return "unstable_detection"
    if shoulder_ratio < 0.80:
        return "shoulders_not_visible"
    if face_size_ratio < 0.15:
        return "face_too_small"
    return None


def _draw_debug(
    frame: np.ndarray, observation: dict[str, Any], crop: dict[str, int], output: Path
) -> None:
    image = frame.copy()
    px1, py1, px2, py2 = (int(v) for v in observation["person"])
    fx, fy, fw, fh = (int(v) for v in observation["face"])
    cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 2)
    cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    for key in ("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"):
        point = observation.get(key)
        if point:
            cv2.circle(image, (round(point[0]), round(point[1])), 4, (0, 255, 255), -1)
    cv2.rectangle(
        image,
        (crop["x"], crop["y"]),
        (crop["x"] + crop["w"], crop["y"] + crop["h"]),
        (0, 0, 255),
        3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"could not write debug preview: {output}")


def _export_clip(
    source: Path, start: float, duration: float, crop: dict[str, int], output: Path
) -> str | None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        return (lines[-1] if lines else "ffmpeg did not create a valid clip")[:1000]
    return None


def _clean_video_batch(
    video_paths: list[Path], output_root: Path, sample_fps: float = 3.0
) -> dict[str, Any]:
    """Run the unchanged Clean V1 algorithm for a batch of videos."""
    if shutil.which("ffmpeg") is None:
        raise CleanError("ffmpeg executable was not found")
    output_root = output_root.resolve()
    clips_dir, debug_dir = output_root / "clips", output_root / "debug"
    clips_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    face_detector = cv2.FaceDetectorYN.create(
        str(_face_model_path()), "", (320, 320), 0.75, 0.3, 5000
    )
    pose_model = YOLO(str(_pose_model_path()))
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    detected_shots = 0

    for video_number, source in enumerate(video_paths, start=1):
        video_id = source.stem
        print(f"[{video_number}/{len(video_paths)}] Scene detection: {video_id}", flush=True)
        scenes = _detect_scenes(source)
        detected_shots += len(scenes)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise CleanError(f"could not open input video: {source}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for shot_index, (start, end) in enumerate(scenes):
            duration = end - start
            base = {
                "source_video_id": video_id,
                "source_path": str(source.resolve()),
                "shot_index": shot_index,
                "start_time": round(start, 6),
                "end_time": round(end, 6),
                "duration": round(duration, 6),
            }
            if duration < 2.0:
                rejected.append({**base, "reject_reason": "too_short"})
                continue

            person_counts: list[int] = []
            face_counts: list[int] = []
            observations: list[dict[str, Any]] = []
            face_sizes: list[float] = []
            representative: tuple[np.ndarray, dict[str, Any]] | None = None
            for timestamp in _sample_times(start, end, sample_fps):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    person_counts.append(0)
                    face_counts.append(0)
                    continue
                result = pose_model.predict(
                    frame, conf=0.35, imgsz=640, device=device, verbose=False
                )[0]
                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0
                person_counts.append(count)
                faces = _faces(face_detector, frame)
                face_counts.append(len(faces))
                if len(faces) == 1:
                    face_sizes.append(faces[0][3] / frame_height)
                if count != 1 or len(faces) != 1 or result.keypoints is None:
                    continue
                confidence = result.keypoints.conf
                coordinates = result.keypoints.xy
                if confidence is None or len(confidence) != 1:
                    continue
                conf = confidence[0].cpu().numpy()
                points = coordinates[0].cpu().numpy()
                if conf[5] < 0.40 or conf[6] < 0.40:
                    continue
                box = boxes.xyxy[0].cpu().numpy().tolist()
                observation = {
                    "person": box,
                    "face": faces[0],
                    "nose": points[0].tolist() if conf[0] >= 0.40 else None,
                    "left_eye": points[1].tolist() if conf[1] >= 0.40 else None,
                    "right_eye": points[2].tolist() if conf[2] >= 0.40 else None,
                    "left_shoulder": points[5].tolist(),
                    "right_shoulder": points[6].tolist(),
                }
                observations.append(observation)
                if representative is None:
                    representative = (frame.copy(), observation)

            sample_count = len(person_counts)
            if sample_count == 0:
                rejected.append({**base, "reject_reason": "unstable_detection"})
                continue
            single_person_ratio = person_counts.count(1) / sample_count
            single_face_ratio = face_counts.count(1) / sample_count
            shoulder_ratio = len(observations) / sample_count
            face_size_ratio = float(np.median(face_sizes)) if face_sizes else 0.0
            metrics = {
                "single_person_ratio": round(single_person_ratio, 6),
                "single_face_ratio": round(single_face_ratio, 6),
                "shoulder_visible_ratio": round(shoulder_ratio, 6),
                "face_size_ratio": round(face_size_ratio, 6),
                "sampled_frames": sample_count,
            }
            reason = _reject_reason(
                person_counts,
                face_counts,
                single_person_ratio,
                single_face_ratio,
                shoulder_ratio,
                face_size_ratio,
            )
            if reason:
                rejected.append({**base, **metrics, "reject_reason": reason})
                continue
            crop = _stable_crop(observations, frame_width, frame_height)
            if crop is None or representative is None:
                rejected.append({**base, **metrics, "reject_reason": "crop_invalid"})
                continue

            clip_id = f"{video_id}_{shot_index:04d}"
            output_path = clips_dir / f"{clip_id}.mp4"
            export_error = _export_clip(source, start, duration, crop, output_path)
            if export_error:
                rejected.append(
                    {**base, **metrics, "crop": crop, "reject_reason": "export_failed", "error": export_error}
                )
                continue
            _draw_debug(representative[0], representative[1], crop, debug_dir / f"{clip_id}.jpg")
            accepted.append(
                {
                    "clip_id": clip_id,
                    **base,
                    **metrics,
                    "crop": crop,
                    "output_path": str(output_path.resolve()),
                }
            )
            print(f"  Accepted {clip_id} ({duration:.2f}s)", flush=True)
        capture.release()

    unknown = {item["reject_reason"] for item in rejected} - _REJECT_REASONS
    if unknown:
        raise AssertionError(f"unknown rejection reasons: {unknown}")
    return {
        "input_videos": len(video_paths),
        "detected_shots": detected_shots,
        "accepted": accepted,
        "rejected": rejected,
        "reject_counts": Counter(item["reject_reason"] for item in rejected),
        "clips_dir": clips_dir,
        "debug_dir": debug_dir,
    }


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frames / fps if fps > 0 and frames > 0 else 0.0


def clean_videos(
    video_paths: list[Path],
    output_root: Path,
    sample_fps: float = 3.0,
    force: bool = False,
) -> dict[str, Any]:
    """Run Clean V1 with source-level resume and failure isolation."""
    output_root = output_root.resolve()
    clips_dir, debug_dir = output_root / "clips", output_root / "debug"
    metadata_path, rejected_path = output_root / "metadata.jsonl", output_root / "rejected.jsonl"
    clips_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    accepted = load_jsonl(metadata_path) if metadata_path.exists() else []
    rejected = load_jsonl(rejected_path) if rejected_path.exists() else []
    selected_ids = {path.stem for path in video_paths}
    processed_ids = {
        str(item["source_video_id"])
        for item in accepted + rejected
        if item.get("source_video_id")
    }
    source_durations = {path.stem: _video_duration(path) for path in video_paths}
    skipped = processed = failed = 0

    for index, source in enumerate(video_paths, start=1):
        video_id = source.stem
        print(f"[{index:03d}/{len(video_paths):03d}] {source.name}", flush=True)
        if video_id in processed_ids and not force:
            source_accepted = sum(item.get("source_video_id") == video_id for item in accepted)
            source_rejected = sum(item.get("source_video_id") == video_id for item in rejected)
            skipped += 1
            print("status: skipped", flush=True)
            print(f"accepted shots: {source_accepted}", flush=True)
            print(f"rejected shots: {source_rejected}", flush=True)
            continue

        if force and video_id in processed_ids:
            accepted = [item for item in accepted if item.get("source_video_id") != video_id]
            rejected = [item for item in rejected if item.get("source_video_id") != video_id]
            for path in clips_dir.glob(f"{video_id}_*.mp4"):
                path.unlink()
            for path in debug_dir.glob(f"{video_id}_*.jpg"):
                path.unlink()
            save_jsonl(accepted, metadata_path)
            save_jsonl(rejected, rejected_path)

        print("status: processing", flush=True)
        try:
            result = _clean_video_batch([source], output_root, sample_fps)
            source_accepted_records = result["accepted"]
            source_rejected_records = result["rejected"]
            accepted.extend(source_accepted_records)
            rejected.extend(source_rejected_records)
            processed += 1
            print("status: done", flush=True)
            print(f"accepted shots: {len(source_accepted_records)}", flush=True)
            print(f"rejected shots: {len(source_rejected_records)}", flush=True)
        except Exception as exc:
            failed += 1
            rejected.append(
                {
                    "source_video_id": video_id,
                    "source_path": str(source.resolve()),
                    "shot_index": -1,
                    "start_time": 0.0,
                    "end_time": round(source_durations[video_id], 6),
                    "duration": round(source_durations[video_id], 6),
                    "reject_reason": "unstable_detection",
                    "error": f"source processing failed: {type(exc).__name__}: {exc}"[:1000],
                }
            )
            print("status: failed", flush=True)
            print("accepted shots: 0", flush=True)
            print("rejected shots: 1", flush=True)
            print(f"error: {type(exc).__name__}: {exc}", flush=True)
        save_jsonl(accepted, metadata_path)
        save_jsonl(rejected, rejected_path)

    selected_accepted = [item for item in accepted if item.get("source_video_id") in selected_ids]
    selected_rejected = [item for item in rejected if item.get("source_video_id") in selected_ids]
    detected_shots = sum(
        int(item.get("shot_index", -1)) >= 0
        for item in selected_accepted + selected_rejected
    )
    return {
        "input_videos": len(video_paths),
        "source_total_duration": sum(source_durations.values()),
        "detected_shots": detected_shots,
        "accepted": selected_accepted,
        "rejected": selected_rejected,
        "reject_counts": Counter(item["reject_reason"] for item in selected_rejected),
        "skipped_sources": skipped,
        "processed_sources": processed,
        "failed_sources": failed,
        "clips_dir": clips_dir,
        "debug_dir": debug_dir,
    }
