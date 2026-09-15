"""Strict whole-video filtering and automatic presenter grouping.

The largest person in the first frame is locked as the primary presenter and
then followed by ArcFace identity. Other people and faces, including
picture-in-picture windows, are ignored. At the first frame where the presenter
face or neck becomes unavailable, that frame and the remainder are discarded;
the valid prefix is retained. Outputs are grouped by model-predicted gender and
identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.crawler.storage import save_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cleaning.yaml"

_REJECT_MESSAGES = {
    "invalid_video": "视频无法打开，或帧率/画面尺寸无效",
    "empty_video": "视频没有可解码帧",
    "decode_incomplete": "实际解码帧数少于视频声明帧数，存在未检查帧",
    "no_person_detected": "该帧没有检测到主主持人候选",
    "primary_presenter_too_small": "首帧最大人物面积不足，不能确认为主主持人",
    "primary_face_not_detected": "主主持人的人物框内没有检测到人脸",
    "primary_identity_match_failed": "当前帧没有人脸能与已锁定主主持人身份匹配",
    "primary_face_not_fully_visible": "主主持人人脸框触及画面边缘，面部不完整",
    "primary_shoulders_not_detected": "主主持人的双肩关键点缺失或置信度不足",
    "primary_neck_geometry_invalid": "主主持人的脸与双肩几何关系不能证明脖子可见",
    "primary_identity_unavailable": "无法提取主主持人的人脸身份特征",
    "primary_gender_unavailable": "无法得到主主持人的性别模型结果",
}


class VideoCleaningError(RuntimeError):
    """Raised when strict video cleaning cannot be completed."""


@dataclass(frozen=True)
class CleaningConfig:
    pose_path: Path
    pose_url: str
    insightface_root: Path
    insightface_name: str
    person_confidence: float
    pose_image_size: int
    shoulder_confidence: float
    face_detection_size: int
    face_edge_margin_ratio: float
    primary_min_person_area_ratio: float
    primary_tracking_similarity_threshold: float
    min_shoulder_to_face_width_ratio: float
    min_face_to_shoulder_gap_ratio: float
    max_face_to_shoulder_gap_ratio: float
    identity_similarity_threshold: float
    video_extensions: tuple[str, ...]


@dataclass
class AcceptedVideo:
    source: Path
    source_id: str
    relative_source: str
    frame_count: int
    fps: float
    width: int
    height: int
    gender: str
    gender_vote_ratio: float
    embedding: np.ndarray
    original_frame_count: int
    truncated: bool
    termination: dict[str, Any] | None = None
    anchor_id: str = ""


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> CleaningConfig:
    """Load and validate the strict cleaner configuration."""
    with path.open("r", encoding="utf-8") as source:
        raw = yaml.safe_load(source)
    try:
        models = raw["models"]
        detection = raw["detection"]
        classification = raw["classification"]
        extensions = tuple(str(value).lower() for value in raw["video_extensions"])
        config = CleaningConfig(
            pose_path=_project_path(models["pose_path"]),
            pose_url=str(models["pose_url"]),
            insightface_root=_project_path(models["insightface_root"]),
            insightface_name=str(models["insightface_name"]),
            person_confidence=float(detection["person_confidence"]),
            pose_image_size=int(detection["pose_image_size"]),
            shoulder_confidence=float(detection["shoulder_confidence"]),
            face_detection_size=int(detection["face_detection_size"]),
            face_edge_margin_ratio=float(detection["face_edge_margin_ratio"]),
            primary_min_person_area_ratio=float(
                detection["primary_min_person_area_ratio"]
            ),
            primary_tracking_similarity_threshold=float(
                detection["primary_tracking_similarity_threshold"]
            ),
            min_shoulder_to_face_width_ratio=float(
                detection["min_shoulder_to_face_width_ratio"]
            ),
            min_face_to_shoulder_gap_ratio=float(
                detection["min_face_to_shoulder_gap_ratio"]
            ),
            max_face_to_shoulder_gap_ratio=float(
                detection["max_face_to_shoulder_gap_ratio"]
            ),
            identity_similarity_threshold=float(
                classification["identity_similarity_threshold"]
            ),
            video_extensions=extensions,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoCleaningError(f"invalid cleaning config: {path}") from exc

    probabilities = (
        config.person_confidence,
        config.shoulder_confidence,
        config.face_edge_margin_ratio,
        config.primary_min_person_area_ratio,
        config.primary_tracking_similarity_threshold,
        config.identity_similarity_threshold,
    )
    if (
        not isinstance(raw, dict)
        or not extensions
        or any(not extension.startswith(".") for extension in extensions)
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or config.pose_image_size <= 0
        or config.face_detection_size <= 0
        or config.min_shoulder_to_face_width_ratio <= 0.0
        or config.min_face_to_shoulder_gap_ratio < 0.0
        or config.max_face_to_shoulder_gap_ratio
        <= config.min_face_to_shoulder_gap_ratio
    ):
        raise VideoCleaningError(f"invalid cleaning config values: {path}")
    return config


def _download_model(url: str, destination: Path, minimum_size: int = 1_000_000) -> Path:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        print(f"Downloading model: {url}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < minimum_size:
            raise VideoCleaningError(f"downloaded model is unexpectedly small: {destination}")
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, VideoCleaningError):
            raise
        raise VideoCleaningError(f"could not download pose model: {exc}") from exc
    return destination


def _create_models(config: CleaningConfig, requested_device: str) -> tuple[Any, Any, str]:
    """Create YOLO-Pose and InsightFace without importing them for other commands."""
    try:
        import onnxruntime as ort
        import torch
        from insightface.app import FaceAnalysis
        from ultralytics import YOLO
    except ImportError as exc:
        raise VideoCleaningError(
            "cleaning dependencies are missing; install dataset/requirements.txt"
        ) from exc

    # ONNX Runtime GPU wheels use CUDA 12/cuDNN 9; preload their pip-installed
    # NVIDIA libraries so they can coexist with this environment's Torch CUDA.
    if requested_device != "cpu" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    cuda_ort = "CUDAExecutionProvider" in ort.get_available_providers()
    cuda_torch = bool(torch.cuda.is_available())
    if requested_device == "cuda" and (not cuda_torch or not cuda_ort):
        raise VideoCleaningError(
            "CUDA was requested but both Torch CUDA and ONNX Runtime CUDA providers are required"
        )
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and cuda_torch and cuda_ort
    )
    device = "0" if use_cuda else "cpu"
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )

    pose_path = _download_model(config.pose_url, config.pose_path)
    pose_model = YOLO(str(pose_path))
    config.insightface_root.mkdir(parents=True, exist_ok=True)
    try:
        face_model = FaceAnalysis(
            name=config.insightface_name,
            root=str(config.insightface_root),
            providers=providers,
        )
        face_model.prepare(
            ctx_id=0 if use_cuda else -1,
            det_size=(config.face_detection_size, config.face_detection_size),
        )
    except Exception as exc:
        raise VideoCleaningError(
            f"could not initialize InsightFace model {config.insightface_name}: {exc}"
        ) from exc
    return pose_model, face_model, device


def _source_id(path: Path, input_root: Path) -> tuple[str, str]:
    relative = path.relative_to(input_root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in path.stem
    )
    return f"{safe_stem}_{digest}", relative


def _normalized_embedding(face: Any) -> np.ndarray | None:
    value = getattr(face, "normed_embedding", None)
    if value is None:
        value = getattr(face, "embedding", None)
    if value is None:
        return None
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(embedding).all() or norm <= 0.0:
        return None
    return embedding / norm


def _face_is_fully_visible(face: Any, width: int, height: int, margin_ratio: float) -> bool:
    bbox = np.asarray(face.bbox, dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return False
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    x1, y1, x2, y2 = bbox
    return (
        x1 >= margin_x
        and y1 >= margin_y
        and x2 <= width - margin_x
        and y2 <= height - margin_y
        and x2 > x1
        and y2 > y1
    )


def _neck_diagnostics(
    face: Any,
    points: np.ndarray,
    confidence: np.ndarray,
    config: CleaningConfig,
) -> dict[str, Any]:
    """Return the complete evidence used for the neck-visibility decision."""
    face_box = np.asarray(face.bbox, dtype=np.float32)
    details: dict[str, Any] = {
        "face_bbox_xyxy": [round(float(value), 2) for value in face_box],
        "shoulder_confidence_threshold": config.shoulder_confidence,
    }
    if points.shape[0] <= 6 or confidence.shape[0] <= 6:
        return {**details, "shoulders_available": False, "visible": False}

    left, right = points[5], points[6]
    left_confidence = float(confidence[5])
    right_confidence = float(confidence[6])
    shoulders_available = bool(
        np.isfinite(left).all()
        and np.isfinite(right).all()
        and left_confidence >= config.shoulder_confidence
        and right_confidence >= config.shoulder_confidence
    )
    details.update(
        {
            "left_shoulder_xy": [round(float(value), 2) for value in left],
            "right_shoulder_xy": [round(float(value), 2) for value in right],
            "left_shoulder_confidence": round(left_confidence, 6),
            "right_shoulder_confidence": round(right_confidence, 6),
            "shoulders_available": shoulders_available,
        }
    )
    if not shoulders_available:
        return {**details, "visible": False}

    x1, _y1, x2, y2 = face_box
    face_width = float(x2 - x1)
    face_height = float(face_box[3] - face_box[1])
    shoulder_width = float(abs(right[0] - left[0]))
    shoulder_y = float((left[1] + right[1]) / 2.0)
    gap_ratio = float((shoulder_y - float(y2)) / max(face_height, 1.0))
    shoulder_to_face_width_ratio = shoulder_width / max(face_width, 1.0)
    face_center_x = float((x1 + x2) / 2.0)
    shoulder_min_x = float(min(left[0], right[0]))
    shoulder_max_x = float(max(left[0], right[0]))
    horizontal_margin = shoulder_width * 0.5
    horizontally_aligned = bool(
        shoulder_min_x - horizontal_margin
        <= face_center_x
        <= shoulder_max_x + horizontal_margin
    )
    visible = bool(
        shoulder_to_face_width_ratio
        >= config.min_shoulder_to_face_width_ratio
        and config.min_face_to_shoulder_gap_ratio
        <= gap_ratio
        <= config.max_face_to_shoulder_gap_ratio
        and horizontally_aligned
    )
    return {
        **details,
        "shoulder_to_face_width_ratio": round(shoulder_to_face_width_ratio, 6),
        "min_shoulder_to_face_width_ratio": config.min_shoulder_to_face_width_ratio,
        "face_to_shoulder_gap_ratio": round(gap_ratio, 6),
        "allowed_face_to_shoulder_gap_ratio": [
            config.min_face_to_shoulder_gap_ratio,
            config.max_face_to_shoulder_gap_ratio,
        ],
        "face_horizontally_between_shoulders": horizontally_aligned,
        "visible": visible,
    }


def _gender_value(face: Any) -> str | None:
    value = getattr(face, "gender", None)
    if value is None:
        value = getattr(face, "sex", None)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"m", "male"}:
            return "male"
        if normalized in {"f", "female"}:
            return "female"
        return None
    if value is None:
        return None
    return "male" if int(value) == 1 else "female" if int(value) == 0 else None


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _faces_inside_person(faces: list[Any], person_box: np.ndarray) -> list[Any]:
    """Associate faces with a pose box, allowing YOLO to omit the head region."""
    x1, y1, x2, y2 = person_box
    person_width = float(x2 - x1)
    person_height = float(y2 - y1)
    # YOLO Pose often starts an upper-body box at the neck even when facial
    # keypoints are valid. Expand upward for association without changing the
    # recorded person box or the subsequent neck checks.
    x1 -= 0.10 * person_width
    x2 += 0.10 * person_width
    y1 -= 0.35 * person_height
    matches: list[Any] = []
    for face in faces:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            continue
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float((bbox[1] + bbox[3]) / 2.0)
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            matches.append(face)
    return sorted(
        matches,
        key=lambda item: _box_area(np.asarray(item.bbox, dtype=np.float32)),
        reverse=True,
    )


def _pose_arrays(result: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), None, None
    boxes = np.asarray(result.boxes.xyxy.cpu().numpy(), dtype=np.float32)
    if result.keypoints is None or result.keypoints.conf is None:
        return boxes, None, None
    points = np.asarray(result.keypoints.xy.cpu().numpy(), dtype=np.float32)
    confidence = np.asarray(result.keypoints.conf.cpu().numpy(), dtype=np.float32)
    if len(points) != len(boxes) or len(confidence) != len(boxes):
        return boxes, None, None
    return boxes, points, confidence


def _best_tracked_presenter(
    boxes: np.ndarray,
    faces: list[Any],
    presenter_embedding: np.ndarray,
    all_points: np.ndarray | None,
    all_confidence: np.ndarray | None,
    config: CleaningConfig,
) -> tuple[int, Any, np.ndarray, float] | None:
    """Match identity first, then choose the pose box with best neck evidence."""
    face_candidates: list[tuple[float, Any, np.ndarray]] = []
    for face in faces:
        embedding = _normalized_embedding(face)
        if embedding is None or embedding.shape != presenter_embedding.shape:
            continue
        if not any(_faces_inside_person([face], box) for box in boxes):
            continue
        similarity = float(np.dot(embedding, presenter_embedding))
        face_candidates.append((similarity, face, embedding))
    if not face_candidates:
        return None

    similarity, face, embedding = max(face_candidates, key=lambda item: item[0])
    person_candidates: list[tuple[tuple[bool, bool, float], int]] = []
    for person_index, box in enumerate(boxes):
        if not _faces_inside_person([face], box):
            continue
        if all_points is None or all_confidence is None:
            rank = (False, False, 0.0)
        else:
            neck = _neck_diagnostics(
                face, all_points[person_index], all_confidence[person_index], config
            )
            minimum_shoulder_confidence = min(
                float(all_confidence[person_index][5]),
                float(all_confidence[person_index][6]),
            )
            rank = (
                bool(neck["visible"]),
                bool(neck["shoulders_available"]),
                minimum_shoulder_confidence,
            )
        person_candidates.append((rank, person_index))
    if not person_candidates:
        return None
    person_index = max(person_candidates, key=lambda item: item[0])[1]
    return person_index, face, embedding, similarity


def _write_failure_preview(
    frame: np.ndarray,
    path: Path,
    reason: str,
    frame_index: int,
    timestamp: float,
    boxes: np.ndarray,
    faces: list[Any],
    primary_person_index: int | None,
) -> None:
    image = frame.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(float(value)) for value in box)
        color = (0, 0, 255) if index == primary_person_index else (255, 128, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"person {index}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = (
            round(float(value)) for value in np.asarray(face.bbox).reshape(4)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"face {index}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    lines = [
        f"REJECT frame={frame_index} time={timestamp:.3f}s",
        reason[:48],
        reason[48:96],
    ]
    lines = [line for line in lines if line]
    cv2.rectangle(image, (0, 0), (image.shape[1], 25 * len(lines) + 8), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (8, 22 + 25 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _inspect_video(
    path: Path,
    input_root: Path,
    pose_model: Any,
    face_model: Any,
    device: str,
    config: CleaningConfig,
    rejection_preview: Path,
) -> tuple[AcceptedVideo | None, dict[str, Any] | None]:
    source_id, relative_source = _source_id(path, input_root)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None, {
            "source_id": source_id,
            "source": relative_source,
            "reject_reason": "invalid_video",
            "reject_message": _REJECT_MESSAGES["invalid_video"],
            "failure_frame": 0,
            "failure_time_sec": 0.0,
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0.0 or width <= 0 or height <= 0:
        capture.release()
        return None, {
            "source_id": source_id,
            "source": relative_source,
            "reject_reason": "invalid_video",
            "reject_message": _REJECT_MESSAGES["invalid_video"],
            "failure_frame": 0,
            "failure_time_sec": 0.0,
        }

    frame_index = 0
    embedding_sum: np.ndarray | None = None
    gender_counts = {"male": 0, "female": 0}
    failure_reason: str | None = None
    failure_details: dict[str, Any] = {}
    failed_frame: np.ndarray | None = None
    failure_boxes = np.empty((0, 4), dtype=np.float32)
    failure_faces: list[Any] = []
    failure_primary_index: int | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pose_model.predict(
            frame,
            conf=config.person_confidence,
            imgsz=config.pose_image_size,
            device=device,
            verbose=False,
        )[0]
        boxes, all_points, all_confidence = _pose_arrays(result)
        faces = list(face_model.get(frame)) if len(boxes) else []
        person_index: int | None = None
        face: Any | None = None
        embedding: np.ndarray | None = None

        if len(boxes) == 0:
            failure_reason = "no_person_detected"
            failure_details = {"person_confidence_threshold": config.person_confidence}
        elif embedding_sum is None:
            # The dominant person establishes the presenter. Smaller people in
            # inset windows are deliberately ignored.
            person_index = int(np.argmax([_box_area(box) for box in boxes]))
            primary_area_ratio = _box_area(boxes[person_index]) / (width * height)
            matched_faces = _faces_inside_person(faces, boxes[person_index])
            failure_details = {
                "selection": "largest_person_in_first_frame",
                "primary_person_bbox_xyxy": [
                    round(float(value), 2) for value in boxes[person_index]
                ],
                "primary_person_area_ratio": round(primary_area_ratio, 6),
                "minimum_primary_person_area_ratio": (
                    config.primary_min_person_area_ratio
                ),
                "faces_inside_primary_person": len(matched_faces),
            }
            if primary_area_ratio < config.primary_min_person_area_ratio:
                failure_reason = "primary_presenter_too_small"
            elif not matched_faces:
                failure_reason = "primary_face_not_detected"
            else:
                face = matched_faces[0]
                embedding = _normalized_embedding(face)
        else:
            presenter_embedding = embedding_sum / np.linalg.norm(embedding_sum)
            tracked = _best_tracked_presenter(
                boxes,
                faces,
                presenter_embedding,
                all_points,
                all_confidence,
                config,
            )
            if tracked is None:
                failure_reason = "primary_identity_match_failed"
                failure_details = {
                    "candidate_face_person_pairs": 0,
                    "required_identity_similarity": (
                        config.primary_tracking_similarity_threshold
                    ),
                }
            else:
                person_index, face, embedding, similarity = tracked
                failure_details = {
                    "primary_person_bbox_xyxy": [
                        round(float(value), 2) for value in boxes[person_index]
                    ],
                    "identity_similarity": round(similarity, 6),
                    "required_identity_similarity": (
                        config.primary_tracking_similarity_threshold
                    ),
                }
                if similarity < config.primary_tracking_similarity_threshold:
                    failure_reason = "primary_identity_match_failed"

        if failure_reason is None:
            assert person_index is not None and face is not None
            face_box = [round(float(value), 2) for value in np.asarray(face.bbox)]
            failure_details["primary_face_bbox_xyxy"] = face_box
            if not _face_is_fully_visible(
                face, width, height, config.face_edge_margin_ratio
            ):
                failure_reason = "primary_face_not_fully_visible"
                failure_details["face_edge_margin_ratio"] = (
                    config.face_edge_margin_ratio
                )
            elif all_points is None or all_confidence is None:
                failure_reason = "primary_shoulders_not_detected"
                failure_details["pose_keypoints_available"] = False
            else:
                neck = _neck_diagnostics(
                    face,
                    all_points[person_index],
                    all_confidence[person_index],
                    config,
                )
                failure_details["neck_evidence"] = neck
                if not neck["shoulders_available"]:
                    failure_reason = "primary_shoulders_not_detected"
                elif not neck["visible"]:
                    failure_reason = "primary_neck_geometry_invalid"
                elif embedding is None:
                    failure_reason = "primary_identity_unavailable"
                else:
                    gender = _gender_value(face)
                    if gender is None:
                        failure_reason = "primary_gender_unavailable"
                    elif embedding_sum is not None and embedding.shape != embedding_sum.shape:
                        failure_reason = "primary_identity_unavailable"
                    else:
                        if embedding_sum is None:
                            embedding_sum = np.zeros_like(embedding)
                        embedding_sum += embedding
                        gender_counts[gender] += 1

        if failure_reason is not None:
            failed_frame = frame.copy()
            failure_boxes = boxes.copy()
            failure_faces = faces
            failure_primary_index = person_index
            break
        frame_index += 1

    capture.release()
    if failure_reason is not None:
        assert failed_frame is not None
        timestamp = frame_index / fps
        preview_relative = f"diagnostic_previews/{source_id}.jpg"
        _write_failure_preview(
            failed_frame,
            rejection_preview / f"{source_id}.jpg",
            failure_reason,
            frame_index,
            timestamp,
            failure_boxes,
            failure_faces,
            failure_primary_index,
        )
        diagnostic = {
            "source_id": source_id,
            "source": relative_source,
            "termination_reason": failure_reason,
            "termination_message": _REJECT_MESSAGES[failure_reason],
            "failure_frame": frame_index,
            "failure_time_sec": round(timestamp, 6),
            "retained_frames": frame_index,
            "retained_duration_sec": round(timestamp, 6),
            "decoded_frames": frame_index + 1,
            "fps": round(fps, 6),
            "frame_size": [width, height],
            "detected_person_count": len(failure_boxes),
            "detected_face_count": len(failure_faces),
            "detected_person_bboxes_xyxy": [
                [round(float(value), 2) for value in box] for box in failure_boxes
            ],
            "detected_face_bboxes_xyxy": [
                [round(float(value), 2) for value in np.asarray(face.bbox)]
                for face in failure_faces
            ],
            "details": failure_details,
            "preview": preview_relative,
        }
        if frame_index == 0 or embedding_sum is None:
            return None, {
                **diagnostic,
                "reject_reason": failure_reason,
                "reject_message": _REJECT_MESSAGES[failure_reason],
            }
        embedding_norm = float(np.linalg.norm(embedding_sum))
        if embedding_norm <= 0.0 or not math.isfinite(embedding_norm):
            return None, {
                **diagnostic,
                "reject_reason": "primary_identity_unavailable",
                "reject_message": _REJECT_MESSAGES["primary_identity_unavailable"],
            }
        gender = max(gender_counts, key=gender_counts.get)
        gender_votes = gender_counts[gender]
        accepted = AcceptedVideo(
            source=path,
            source_id=source_id,
            relative_source=relative_source,
            frame_count=frame_index,
            fps=fps,
            width=width,
            height=height,
            gender=gender,
            gender_vote_ratio=gender_votes / frame_index,
            embedding=embedding_sum / embedding_norm,
            original_frame_count=declared_frames,
            truncated=True,
            termination=diagnostic,
        )
        return accepted, diagnostic

    # A short decode means at least one declared frame was not inspected, so reject it.
    if frame_index == 0 or (declared_frames > 0 and frame_index < declared_frames):
        return None, {
            "source_id": source_id,
            "source": relative_source,
            "reject_reason": "decode_incomplete" if frame_index else "empty_video",
            "reject_message": _REJECT_MESSAGES[
                "decode_incomplete" if frame_index else "empty_video"
            ],
            "failure_frame": frame_index,
            "failure_time_sec": round(frame_index / fps, 6),
            "decoded_frames": frame_index,
            "declared_frames": declared_frames,
            "fps": fps,
        }
    if embedding_sum is None:
        raise AssertionError("accepted frame loop produced no identity embedding")
    embedding_norm = float(np.linalg.norm(embedding_sum))
    if embedding_norm <= 0.0 or not math.isfinite(embedding_norm):
        return None, {
            "source_id": source_id,
            "source": relative_source,
            "reject_reason": "primary_identity_unavailable",
            "reject_message": _REJECT_MESSAGES["primary_identity_unavailable"],
            "failure_frame": frame_index,
            "failure_time_sec": round(frame_index / fps, 6),
        }

    gender = max(gender_counts, key=gender_counts.get)
    gender_votes = gender_counts[gender]
    accepted = AcceptedVideo(
        source=path,
        source_id=source_id,
        relative_source=relative_source,
        frame_count=frame_index,
        fps=fps,
        width=width,
        height=height,
        gender=gender,
        gender_vote_ratio=gender_votes / frame_index,
        embedding=embedding_sum / embedding_norm,
        original_frame_count=declared_frames,
        truncated=False,
    )
    return accepted, None


def _assign_anchors(videos: list[AcceptedVideo], threshold: float) -> None:
    """Greedily cluster normalized video embeddings in deterministic source order."""
    for gender in ("male", "female"):
        clusters: list[dict[str, Any]] = []
        gender_videos = sorted(
            (item for item in videos if item.gender == gender),
            key=lambda item: item.relative_source,
        )
        for video in gender_videos:
            similarities = [
                float(np.dot(video.embedding, cluster["centroid"]))
                for cluster in clusters
            ]
            best = int(np.argmax(similarities)) if similarities else -1
            if best >= 0 and similarities[best] >= threshold:
                cluster = clusters[best]
                cluster["members"].append(video)
                total = np.sum([member.embedding for member in cluster["members"]], axis=0)
                cluster["centroid"] = total / np.linalg.norm(total)
                video.anchor_id = f"anchor_{best + 1:04d}"
            else:
                clusters.append({"members": [video], "centroid": video.embedding.copy()})
                video.anchor_id = f"anchor_{len(clusters):04d}"


def _export_prefix(source: Path, destination: Path, duration: float) -> None:
    """Re-encode an exact valid prefix, excluding the first failing frame."""
    if shutil.which("ffmpeg") is None:
        raise VideoCleaningError("ffmpeg is required to export truncated videos")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        message = result.stderr.strip().splitlines()
        raise VideoCleaningError(
            f"could not export valid prefix for {source.name}: "
            f"{message[-1] if message else 'ffmpeg failed'}"
        )


def _replace_directory(staging: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise VideoCleaningError(f"output already exists (use --force to replace it): {output}")
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


def clean_videos(
    input_root: Path,
    output_root: Path,
    config: CleaningConfig,
    *,
    device: str = "auto",
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Retain valid presenter prefixes and publish gender/identity groups atomically."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise VideoCleaningError(f"input directory not found: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise VideoCleaningError("output must not be the input directory or one of its children")
    if device not in {"auto", "cpu", "cuda"}:
        raise VideoCleaningError("device must be auto, cpu, or cuda")
    if limit is not None and limit <= 0:
        raise VideoCleaningError("limit must be greater than zero")

    videos = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in config.video_extensions
    )
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise VideoCleaningError(f"no supported videos found in: {input_root}")
    if output_root.exists() and not force:
        raise VideoCleaningError(
            f"output already exists (use --force to replace it): {output_root}"
        )

    pose_model, face_model, actual_device = _create_models(config, device)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    accepted: list[AcceptedVideo] = []
    truncated_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        for index, path in enumerate(videos, 1):
            print(f"[{index}/{len(videos)}] {path.relative_to(input_root)}", flush=True)
            item, diagnostic = _inspect_video(
                path,
                input_root,
                pose_model,
                face_model,
                actual_device,
                config,
                staging / "diagnostic_previews",
            )
            if item is not None:
                accepted.append(item)
                if diagnostic is None:
                    print("  accepted complete video", flush=True)
                else:
                    truncated_records.append(diagnostic)
                    save_jsonl(truncated_records, staging / "truncated.jsonl")
                    details_path = (
                        staging
                        / "truncated_details"
                        / f"{diagnostic['source_id']}.json"
                    )
                    details_path.parent.mkdir(parents=True, exist_ok=True)
                    details_path.write_text(
                        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        "  retained prefix through "
                        f"frame {diagnostic['failure_frame'] - 1} "
                        f"({diagnostic['failure_time_sec']:.3f}s); removed remainder: "
                        f"{diagnostic['termination_reason']} - "
                        f"{diagnostic['termination_message']}",
                        flush=True,
                    )
            else:
                assert diagnostic is not None
                rejected.append(diagnostic)
                save_jsonl(rejected, staging / "rejected.jsonl")
                details_path = (
                    staging / "rejected_details" / f"{diagnostic['source_id']}.json"
                )
                details_path.parent.mkdir(parents=True, exist_ok=True)
                details_path.write_text(
                    json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    "  no usable prefix; rejected at "
                    f"frame {diagnostic['failure_frame']} "
                    f"({diagnostic['failure_time_sec']:.3f}s): "
                    f"{diagnostic['reject_reason']} - {diagnostic['reject_message']}",
                    flush=True,
                )

        _assign_anchors(accepted, config.identity_similarity_threshold)
        accepted_records: list[dict[str, Any]] = []
        for item in accepted:
            destination_dir = staging / item.gender / item.anchor_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_suffix = ".mp4" if item.truncated else item.source.suffix.lower()
            destination = destination_dir / f"{item.source_id}{destination_suffix}"
            if item.truncated:
                _export_prefix(item.source, destination, item.frame_count / item.fps)
            else:
                shutil.copy2(item.source, destination)
            accepted_records.append(
                {
                    "source_id": item.source_id,
                    "source": item.relative_source,
                    "gender": item.gender,
                    "gender_vote_ratio": round(item.gender_vote_ratio, 6),
                    "anchor_id": item.anchor_id,
                    "output": destination.relative_to(staging).as_posix(),
                    "retained_frame_count": item.frame_count,
                    "retained_duration_sec": round(item.frame_count / item.fps, 6),
                    "original_declared_frame_count": item.original_frame_count,
                    "truncated": item.truncated,
                    "termination_reason": (
                        item.termination["termination_reason"]
                        if item.termination is not None
                        else None
                    ),
                    "fps": round(item.fps, 6),
                    "width": item.width,
                    "height": item.height,
                }
            )

        save_jsonl(accepted_records, staging / "accepted.jsonl")
        save_jsonl(truncated_records, staging / "truncated.jsonl")
        save_jsonl(rejected, staging / "rejected.jsonl")
        anchor_count = len({(item.gender, item.anchor_id) for item in accepted})
        manifest = {
            "version": "clean_v2",
            "complete": True,
            "input_root": str(input_root),
            "models": {
                "pose": str(config.pose_path),
                "face_identity_gender": config.insightface_name,
            },
            "rules": {
                "inspect_every_decoded_frame": True,
                "primary_presenter_selection": "largest_person_in_first_frame",
                "primary_presenter_tracking": "arcface_identity",
                "ignore_non_primary_people_and_faces": True,
                "require_primary_full_face": True,
                "require_primary_both_shoulders_and_neck_geometry": True,
                "retain_valid_prefix_before_first_failure": True,
                "discard_first_failure_frame_and_all_following_frames": True,
                "pose_image_size": config.pose_image_size,
                "tracking_similarity_threshold": (
                    config.primary_tracking_similarity_threshold
                ),
                "identity_similarity_threshold": config.identity_similarity_threshold,
            },
            "input_videos": len(videos),
            "accepted_videos": len(accepted_records),
            "truncated_videos": len(truncated_records),
            "rejected_without_usable_prefix": len(rejected),
            "anchors": anchor_count,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _replace_directory(staging, output_root, force)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retain valid presenter prefixes and group by gender and identity"
    )
    parser.add_argument("--input", required=True, type=Path, help="source video directory")
    parser.add_argument("--output", required=True, type=Path, help="new cleaned dataset directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="process only the first N videos")
    parser.add_argument("--force", action="store_true", help="atomically replace existing output")
    args = parser.parse_args(argv)
    try:
        manifest = clean_videos(
            args.input,
            args.output,
            load_config(args.config.resolve()),
            device=args.device,
            limit=args.limit,
            force=args.force,
        )
    except (VideoCleaningError, OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
