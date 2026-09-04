"""Experimental visualization for validating a candidate MediaPipe RPY convention."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VERSION = "neck_pose_v0"
_REQUIRED_OUTPUTS = (
    "rotation.npy",
    "raw_rpy.npy",
    "rpy_curve.png",
    "debug_pose.mp4",
    "metadata.json",
)


class NeckPoseV0Error(RuntimeError):
    """Raised when a diagnostic clip cannot be generated safely."""


def project_to_so3(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a finite 3x3 matrix to its nearest proper rotation using SVD."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("SO(3) projection requires a finite 3x3 matrix")
    u, singular_values, vh = np.linalg.svd(value)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, singular_values


def rotation_to_zyx_rpy(rotation: np.ndarray, gimbal_epsilon: float = 1e-7) -> np.ndarray:
    """Return [roll_x, pitch_y, yaw_z] for R=Rz(yaw)Ry(pitch)Rx(roll)."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Euler decomposition requires a finite 3x3 matrix")
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > gimbal_epsilon:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        # At either pole roll/yaw are not independently identifiable. This
        # deterministic candidate chooses roll=0 and preserves an equivalent yaw.
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _load_inputs(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ("timestamps.npy", "valid.npy", "landmarks.npy", "transforms.npy")
    missing = [name for name in names if not (feature_dir / name).is_file()]
    if missing:
        raise NeckPoseV0Error(f"missing MediaPipe input(s): {', '.join(missing)}")
    timestamps = np.load(feature_dir / "timestamps.npy", allow_pickle=False)
    valid = np.load(feature_dir / "valid.npy", allow_pickle=False)
    landmarks = np.load(feature_dir / "landmarks.npy", allow_pickle=False)
    transforms = np.load(feature_dir / "transforms.npy", allow_pickle=False)
    if timestamps.ndim != 1 or valid.ndim != 1:
        raise NeckPoseV0Error("timestamps and valid must be one-dimensional")
    frame_count = len(timestamps)
    lengths = (frame_count, len(valid), landmarks.shape[0], transforms.shape[0])
    if len(set(lengths)) != 1:
        raise NeckPoseV0Error(
            "alignment error: timestamps/valid/landmarks/transforms lengths are "
            + "/".join(str(value) for value in lengths)
        )
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        raise NeckPoseV0Error(f"unexpected landmarks shape: {landmarks.shape}")
    if transforms.ndim != 3 or transforms.shape[1] < 3 or transforms.shape[2] < 3:
        raise NeckPoseV0Error(f"unexpected transforms shape: {transforms.shape}")
    if valid.dtype != np.bool_:
        raise NeckPoseV0Error(f"valid mask must have bool dtype, got {valid.dtype}")
    if not np.isfinite(timestamps).all():
        raise NeckPoseV0Error("timestamps contain non-finite values")
    return timestamps, valid, landmarks, transforms


def _rotation_features(
    valid: np.ndarray, transforms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_count = len(valid)
    rotations = np.full((frame_count, 3, 3), np.nan, dtype=np.float32)
    raw_rpy = np.full((frame_count, 3), np.nan, dtype=np.float32)
    raw_determinants: list[float] = []
    projected_determinants: list[float] = []
    raw_orthogonality: list[float] = []
    projected_orthogonality: list[float] = []
    singular_values: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float64)

    for index in np.flatnonzero(valid):
        linear = np.asarray(transforms[index, :3, :3], dtype=np.float64)
        if not np.isfinite(linear).all():
            raise NeckPoseV0Error(f"valid frame {index} has a non-finite transform")
        try:
            rotation, values = project_to_so3(linear)
            rpy = rotation_to_zyx_rpy(rotation)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise NeckPoseV0Error(f"rotation failed at valid frame {index}: {exc}") from exc
        determinant = float(np.linalg.det(rotation))
        orthogonality = float(np.linalg.norm(rotation.T @ rotation - identity, ord="fro"))
        if (
            not np.isfinite(orthogonality)
            or abs(determinant - 1.0) > 1e-5
            or orthogonality > 1e-5
        ):
            raise NeckPoseV0Error(f"invalid projected rotation at frame {index}")
        rotations[index] = rotation.astype(np.float32)
        raw_rpy[index] = rpy.astype(np.float32)
        raw_determinants.append(float(np.linalg.det(linear)))
        projected_determinants.append(determinant)
        raw_orthogonality.append(float(np.linalg.norm(linear.T @ linear - identity, ord="fro")))
        projected_orthogonality.append(orthogonality)
        singular_values.append(values)

    def aggregate(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }

    diagnostics: dict[str, Any] = {
        "raw_determinant": aggregate(raw_determinants),
        "projected_determinant": aggregate(projected_determinants),
        "raw_orthogonality_error_frobenius": aggregate(raw_orthogonality),
        "projected_orthogonality_error_frobenius": aggregate(projected_orthogonality),
        "mean_singular_values": (
            np.mean(np.stack(singular_values), axis=0).tolist() if singular_values else None
        ),
    }
    return rotations, raw_rpy, diagnostics


def _plot_curve(
    output: Path, clip_id: str, timestamps: np.ndarray, raw_rpy: np.ndarray, valid_ratio: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = np.rad2deg(raw_rpy)
    figure, axis = plt.subplots(figsize=(12, 5), dpi=140)
    for column, label, color in (
        (0, "roll_x", "red"),
        (1, "pitch_y", "green"),
        (2, "yaw_z", "blue"),
    ):
        axis.plot(timestamps, degrees[:, column], label=label, color=color, linewidth=1.0)
    axis.set_xlabel("time / second")
    axis.set_ylabel("angle / degree")
    axis.set_title(
        f"{clip_id} | valid ratio={valid_ratio:.2%} | Euler convention = ZYX candidate"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _draw_axes(frame: np.ndarray, rotation: np.ndarray) -> None:
    height, width = frame.shape[:2]
    origin = (max(55, width - 90), min(90, max(55, height // 5)))
    scale = max(30, min(width, height) // 9)
    cv2.putText(
        frame, "MediaPipe rotation diagnostic", (max(5, width - 315), 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for column, label, color in (
        (0, "X", (0, 0, 255)),
        (1, "Y", (0, 255, 0)),
        (2, "Z", (255, 0, 0)),
    ):
        vector = rotation[:, column]
        endpoint = (
            int(round(origin[0] + scale * vector[0])),
            int(round(origin[1] - scale * vector[1])),
        )
        cv2.arrowedLine(frame, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(frame, endpoint, 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame, label, (endpoint[0] + 4, endpoint[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )


def _draw_face_bbox(frame: np.ndarray, landmarks: np.ndarray) -> None:
    finite = np.isfinite(landmarks).all(axis=1)
    if not finite.any():
        return
    height, width = frame.shape[:2]
    points = landmarks[finite, :2]
    x1 = int(np.clip(np.min(points[:, 0]) * width, 0, width - 1))
    y1 = int(np.clip(np.min(points[:, 1]) * height, 0, height - 1))
    x2 = int(np.clip(np.max(points[:, 0]) * width, 0, width - 1))
    y2 = int(np.clip(np.max(points[:, 1]) * height, 0, height - 1))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)


def _write_debug_video(
    video_path: Path,
    output: Path,
    timestamps: np.ndarray,
    valid: np.ndarray,
    landmarks: np.ndarray,
    rotations: np.ndarray,
    raw_rpy: np.ndarray,
) -> tuple[float, int, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise NeckPoseV0Error(f"could not open Clean V1 clip: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_count = len(timestamps)
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise NeckPoseV0Error("Clean V1 clip has invalid FPS or resolution")
    if reported_count > 0 and reported_count != frame_count:
        capture.release()
        raise NeckPoseV0Error(
            f"alignment error: video reports {reported_count} frames, features have {frame_count}"
        )
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise NeckPoseV0Error(f"could not create debug video: {output}")
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise NeckPoseV0Error(
                    f"alignment error: video ended at frame {index}, expected {frame_count}"
                )
            cv2.putText(
                frame, f"frame = {index}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"timestamp = {timestamps[index]:.3f} s", (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if valid[index]:
                degrees = np.rad2deg(raw_rpy[index])
                for line, (label, value) in enumerate(zip(("roll_x", "pitch_y", "yaw_z"), degrees)):
                    cv2.putText(
                        frame, f"{label} = {value:.1f} deg", (15, 82 + line * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                cv2.putText(
                    frame, "VALID = TRUE", (15, 171), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )
                _draw_axes(frame, rotations[index])
                _draw_face_bbox(frame, landmarks[index])
            else:
                cv2.putText(
                    frame, "RPY = N/A", (15, 88), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "VALID = FALSE", (15, 118), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
            writer.write(frame)
        extra, _ = capture.read()
        if extra:
            raise NeckPoseV0Error(
                f"alignment error: video has more than {frame_count} feature-aligned frames"
            )
    finally:
        writer.release()
        capture.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise NeckPoseV0Error("debug video write produced an empty file")
    return fps, width, height, frame_count / fps


def _completed(directory: Path) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in _REQUIRED_OUTPUTS):
        return None
    try:
        with (directory / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        frame_count = int(metadata["frame_count"])
        rotation = np.load(directory / "rotation.npy", mmap_mode="r", allow_pickle=False)
        raw_rpy = np.load(directory / "raw_rpy.npy", mmap_mode="r", allow_pickle=False)
        if (
            metadata.get("status") != "completed"
            or metadata.get("analysis_version") != VERSION
            or rotation.shape != (frame_count, 3, 3)
            or raw_rpy.shape != (frame_count, 3)
            or rotation.dtype != np.float32
            or raw_rpy.dtype != np.float32
            or (directory / "rpy_curve.png").stat().st_size == 0
            or (directory / "debug_pose.mp4").stat().st_size == 0
        ):
            return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def process_neck_pose_clip(feature_dir: Path, video_path: Path, output_dir: Path) -> dict[str, Any]:
    """Create one experimental RPY diagnostic package with strict frame alignment."""
    timestamps, valid, landmarks, transforms = _load_inputs(feature_dir)
    rotations, raw_rpy, diagnostics = _rotation_features(valid, transforms)
    clip_id = feature_dir.name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output_dir.parent))
    try:
        np.save(temporary / "rotation.npy", rotations, allow_pickle=False)
        np.save(temporary / "raw_rpy.npy", raw_rpy, allow_pickle=False)
        valid_frames = int(valid.sum())
        valid_ratio = valid_frames / len(valid) if len(valid) else 0.0
        _plot_curve(temporary / "rpy_curve.png", clip_id, timestamps, raw_rpy, valid_ratio)
        fps, width, height, duration = _write_debug_video(
            video_path, temporary / "debug_pose.mp4", timestamps, valid,
            landmarks, rotations, raw_rpy,
        )
        metadata: dict[str, Any] = {
            "clip_id": clip_id,
            "frame_count": len(valid),
            "fps": fps,
            "duration": duration,
            "video_resolution": [width, height],
            "valid_frames": valid_frames,
            "invalid_frames": len(valid) - valid_frames,
            "valid_ratio": valid_ratio,
            "source_transform_shape": list(transforms.shape[1:]),
            "rotation_extraction": "SVD nearest SO(3)",
            "rotation_diagnostics": diagnostics,
            "candidate_euler_convention": {
                "composition": "R = Rz(yaw) Ry(pitch) Rx(roll)",
                "output_order": ["roll_x", "pitch_y", "yaw_z"],
                "unit": "radian",
                "status": "candidate_not_final",
            },
            "smoothing": False,
            "interpolation": False,
            "neutral_normalization": False,
            "analysis_version": VERSION,
            "status": "completed",
        }
        with (temporary / "metadata.json").open("w", encoding="utf-8") as target:
            json.dump(metadata, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
        return metadata
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_neck_pose_clips(
    feature_dirs: list[Path], video_root: Path, output_root: Path, force: bool = False
) -> dict[str, int]:
    """Generate V0 diagnostics with clip-level resume and failure isolation."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processed = skipped = failed = 0
    for index, feature_dir in enumerate(feature_dirs, start=1):
        clip_id = feature_dir.name
        output_dir = output_root / clip_id
        print(f"[{index:03d}/{len(feature_dirs):03d}] {clip_id}", flush=True)
        existing = _completed(output_dir)
        if existing is not None and not force:
            skipped += 1
            print("Status: skipped", flush=True)
            continue
        video_path = video_root / f"{clip_id}.mp4"
        try:
            if not video_path.is_file():
                raise NeckPoseV0Error(f"Clean V1 clip not found: {video_path}")
            metadata = process_neck_pose_clip(feature_dir, video_path, output_dir)
            processed += 1
            print(f"Frames: {metadata['frame_count']}")
            print(f"Valid ratio: {metadata['valid_ratio']:.2%}")
            print("Status: done", flush=True)
        except Exception as exc:
            failed += 1
            print("Status: failed")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
    return {"input_clips": len(feature_dirs), "processed": processed, "skipped": skipped, "failed": failed}
