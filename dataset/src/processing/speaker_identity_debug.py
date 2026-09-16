"""Inspect presenter-face selection on five fixed timestamps from ten videos.

This is a diagnostic-only tool. It reads source videos without modifying them,
does not cluster identities, and does not persist face embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.processing.clean_videos import (
    DEFAULT_CONFIG,
    CleaningConfig,
    _gender_value,
    _normalized_embedding,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "videos"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "speaker_identity_debug"
)
TIMESTAMPS_SEC = (0.0, 0.5, 1.0, 1.5, 2.0)
VIDEO_COUNT = 10
DEFAULT_SEED = 42
_UPPER_FACE_CENTER_RATIO = 0.6
_PANEL_WIDTH = 360
_PANEL_HEIGHT = 360
_REQUIRED_BUFFALO_FILES = ("det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")


class SpeakerIdentityDebugError(RuntimeError):
    """Raised when the diagnostic cannot run or publish safely."""


def _create_face_model(config: CleaningConfig, requested_device: str) -> tuple[Any, str]:
    """Initialize the local Buffalo-L detection, recognition, and gender models."""
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SpeakerIdentityDebugError(
            "InsightFace dependencies are missing; install dataset/requirements.txt"
        ) from exc

    if requested_device not in {"auto", "cpu", "cuda"}:
        raise SpeakerIdentityDebugError("device must be auto, cpu, or cuda")
    if requested_device != "cpu" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    if requested_device == "cuda" and not cuda_available:
        raise SpeakerIdentityDebugError(
            "CUDA was requested but ONNX Runtime CUDAExecutionProvider is unavailable"
        )
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and cuda_available
    )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )

    model_dir = config.insightface_root / "models" / config.insightface_name
    missing = [name for name in _REQUIRED_BUFFALO_FILES if not (model_dir / name).is_file()]
    if missing:
        raise SpeakerIdentityDebugError(
            "local InsightFace model files are missing: "
            + ", ".join(str(model_dir / name) for name in missing)
        )

    try:
        model = FaceAnalysis(
            name=config.insightface_name,
            root=str(config.insightface_root),
            allowed_modules=["detection", "recognition", "genderage"],
            providers=providers,
        )
        model.prepare(
            ctx_id=0 if use_cuda else -1,
            det_size=(config.face_detection_size, config.face_detection_size),
        )
    except Exception as exc:
        raise SpeakerIdentityDebugError(
            f"could not initialize InsightFace model {config.insightface_name}: {exc}"
        ) from exc
    return model, "cuda" if use_cuda else "cpu"


def _valid_face_box(face: Any) -> np.ndarray | None:
    bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float32)
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        return None
    return bbox


def _face_area(face: Any) -> float:
    bbox = _valid_face_box(face)
    if bbox is None:
        return -1.0
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _select_presenter_face(faces: list[Any], image_height: int) -> tuple[Any | None, str | None]:
    valid = [face for face in faces if _valid_face_box(face) is not None]
    upper = [
        face
        for face in valid
        if float((_valid_face_box(face)[1] + _valid_face_box(face)[3]) / 2.0)
        < _UPPER_FACE_CENTER_RATIO * image_height
    ]
    if upper:
        return max(upper, key=_face_area), "upper"
    if valid:
        return max(valid, key=_face_area), "full_frame_fallback"
    return None, None


def _safe_score(face: Any) -> float | None:
    try:
        score = float(face.det_score)
    except (AttributeError, TypeError, ValueError):
        return None
    return round(score, 6) if math.isfinite(score) else None


def _frame_record(
    timestamp: float,
    frame: np.ndarray,
    faces: list[Any],
    selected: Any | None,
    selection_scope: str | None,
    decoded_frame_index: int | None,
    fps: float,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    record: dict[str, Any] = {
        "timestamp_sec": timestamp,
        "decode_success": True,
        "decoded_frame_index": decoded_frame_index,
        "decoded_timestamp_sec": (
            round(decoded_frame_index / fps, 6)
            if decoded_frame_index is not None and fps > 0
            else None
        ),
        "image_size": [width, height],
        "detected_face_count": len(faces),
        "selection_scope": selection_scope,
        "bbox_xyxy": None,
        "det_score": None,
        "gender": None,
        "normed_embedding_available": False,
        "error": None,
    }
    if selected is None:
        record["error"] = "no_valid_face"
        return record

    bbox = _valid_face_box(selected)
    assert bbox is not None
    try:
        gender = _gender_value(selected)
    except (TypeError, ValueError, OverflowError):
        gender = None
    record.update(
        {
            "bbox_xyxy": [round(float(value), 2) for value in bbox],
            "det_score": _safe_score(selected),
            "gender": gender,
            "normed_embedding_available": _normalized_embedding(selected) is not None,
        }
    )
    return record


def _draw_selection(frame: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    image = frame.copy()
    bbox = record["bbox_xyxy"]
    if bbox is not None:
        x1, y1, x2, y2 = (round(float(value)) for value in bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    gender = record["gender"] or "unknown"
    scope = record["selection_scope"] or "none"
    label = f"t={record['timestamp_sec']:.1f}s gender={gender} select={scope}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if bbox is not None else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _failure_panel(timestamp: float, message: str) -> np.ndarray:
    panel = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 32, dtype=np.uint8)
    cv2.putText(
        panel,
        f"t={timestamp:.1f}s",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        message[:40],
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _panel(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(_PANEL_WIDTH / width, _PANEL_HEIGHT / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((_PANEL_HEIGHT, _PANEL_WIDTH, 3), dtype=np.uint8)
    x = (_PANEL_WIDTH - resized_width) // 2
    y = (_PANEL_HEIGHT - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def _inspect_video(video: Path, face_model: Any) -> tuple[dict[str, Any], np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        frames = [
            {
                "timestamp_sec": timestamp,
                "decode_success": False,
                "decoded_frame_index": None,
                "decoded_timestamp_sec": None,
                "image_size": None,
                "detected_face_count": 0,
                "selection_scope": None,
                "bbox_xyxy": None,
                "det_score": None,
                "gender": None,
                "normed_embedding_available": False,
                "error": "video_open_failed",
            }
            for timestamp in TIMESTAMPS_SEC
        ]
        panels = [_failure_panel(timestamp, "video open failed") for timestamp in TIMESTAMPS_SEC]
        return {"video": video.name, "frames": frames}, np.hstack(panels)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    records: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    try:
        for timestamp in TIMESTAMPS_SEC:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                records.append(
                    {
                        "timestamp_sec": timestamp,
                        "decode_success": False,
                        "decoded_frame_index": None,
                        "decoded_timestamp_sec": None,
                        "image_size": None,
                        "detected_face_count": 0,
                        "selection_scope": None,
                        "bbox_xyxy": None,
                        "det_score": None,
                        "gender": None,
                        "normed_embedding_available": False,
                        "error": "frame_decode_failed",
                    }
                )
                panels.append(_failure_panel(timestamp, "frame decode failed"))
                continue

            position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            decoded_frame_index = (
                max(0, round(position) - 1) if math.isfinite(position) else None
            )
            faces = list(face_model.get(frame))
            selected, scope = _select_presenter_face(faces, frame.shape[0])
            record = _frame_record(
                timestamp, frame, faces, selected, scope, decoded_frame_index, fps
            )
            records.append(record)
            panels.append(_panel(_draw_selection(frame, record)))
    finally:
        capture.release()

    return {"video": video.name, "frames": records}, np.hstack(panels)


def _publish(staging: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise SpeakerIdentityDebugError(
            f"output already exists (use --force to replace it): {output}"
        )
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def inspect_random_videos(
    input_root: Path,
    output: Path,
    config: CleaningConfig,
    *,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Select ten videos reproducibly and publish contact sheets plus JSONL."""
    input_root = input_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not input_root.is_dir():
        raise SpeakerIdentityDebugError(f"input directory not found: {input_root}")
    videos = sorted(
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() in config.video_extensions
    )
    if len(videos) < VIDEO_COUNT:
        raise SpeakerIdentityDebugError(
            f"need at least {VIDEO_COUNT} input videos, found {len(videos)}"
        )
    if output.exists() and not force:
        raise SpeakerIdentityDebugError(
            f"output already exists (use --force to replace it): {output}"
        )

    selected_videos = random.Random(seed).sample(videos, VIDEO_COUNT)
    face_model, actual_device = _create_face_model(config, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    records: list[dict[str, Any]] = []
    try:
        for index, video in enumerate(selected_videos, 1):
            print(f"[{index}/{VIDEO_COUNT}] {video.name}", flush=True)
            record, contact_sheet = _inspect_video(video, face_model)
            record["contact_sheet"] = f"{video.stem}.jpg"
            records.append(record)
            if not cv2.imwrite(
                str(staging / record["contact_sheet"]),
                contact_sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise SpeakerIdentityDebugError(
                    f"could not write contact sheet for: {video.name}"
                )

        with (staging / "results.jsonl").open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        manifest = {
            "version": "speaker_identity_debug_v1",
            "complete": True,
            "input_root": str(input_root),
            "seed": seed,
            "video_count": VIDEO_COUNT,
            "timestamps_sec": list(TIMESTAMPS_SEC),
            "model": config.insightface_name,
            "face_detection_size": config.face_detection_size,
            "device": actual_device,
            "selection": {
                "prefer_face_center_above_image_ratio": _UPPER_FACE_CENTER_RATIO,
                "rank": "largest_bbox_area",
                "fallback": "largest_bbox_area_in_full_frame",
            },
            "videos": [video.name for video in selected_videos],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output, force)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect presenter-face selection on five frames from ten random videos"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace existing debug output")
    args = parser.parse_args(argv)
    try:
        manifest = inspect_random_videos(
            args.input,
            args.output,
            load_config(args.config.resolve()),
            seed=args.seed,
            device=args.device,
            force=args.force,
        )
    except (OSError, ValueError, SpeakerIdentityDebugError, yaml.YAMLError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Videos inspected: {manifest['video_count']}")
    print(f"Seed: {manifest['seed']}")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
