"""Neck Pose V1 ground-truth extraction from stored MediaPipe transforms."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

FEATURE_VERSION = "neck_pose_v1"
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 1.5
MAX_INTERPOLATION_GAP_SECONDS = 0.20
_REQUIRED_FILES = (
    "timestamps.npy",
    "valid.npy",
    "relative_rotation.npy",
    "relative_rpy_raw.npy",
    "neck_rpy.npy",
    "metadata.json",
)


class NeckPoseError(RuntimeError):
    """Raised when Neck Pose V1 cannot safely process a clip."""


def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper 3D rotation to a finite 3x3 matrix."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("SO(3) projection requires a finite 3x3 matrix")
    u, _, vh = np.linalg.svd(value)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    determinant = float(np.linalg.det(rotation))
    error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    if abs(determinant - 1.0) > 1e-6 or error > 1e-6:
        raise ValueError("SVD projection did not produce a valid SO(3) rotation")
    return rotation


def rotation_to_zyx_rpy(rotation: np.ndarray, gimbal_epsilon: float = 1e-7) -> np.ndarray:
    """Return candidate [roll_x,pitch_y,yaw_z] for R=Rz Ry Rx."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Euler decomposition requires a finite 3x3 matrix")
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > gimbal_epsilon:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def semantic_candidate_to_human(candidate_rpy: np.ndarray) -> np.ndarray:
    """Map [roll_x,pitch_y,yaw_z] to human [roll,pitch,yaw], without sign flips."""
    candidate = np.asarray(candidate_rpy)
    if candidate.shape[-1:] != (3,):
        raise ValueError("candidate RPY must have a final dimension of three")
    return candidate[..., [2, 0, 1]]


def relative_rotations(
    rotations: np.ndarray, valid: np.ndarray, neutral_index: int | None = None
) -> tuple[np.ndarray, int]:
    """Compute R0.T @ R(t), using the first valid frame unless explicitly supplied."""
    values = np.asarray(rotations, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != (len(mask), 3, 3):
        raise ValueError("rotations and valid must have shapes [T,3,3] and [T]")
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise ValueError("clip has no valid rotation")
    index = int(valid_indices[0]) if neutral_index is None else int(neutral_index)
    if index < 0 or index >= len(mask) or not mask[index]:
        raise ValueError("neutral frame must be valid")
    relative = np.full_like(values, np.nan)
    neutral_transpose = values[index].T
    relative[mask] = np.einsum("ij,tjk->tik", neutral_transpose, values[mask])
    return relative, index


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert SO(3) to a normalized [w,x,y,z] quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * scale, (r[2, 1] - r[1, 2]) / scale,
             (r[0, 2] - r[2, 0]) / scale, (r[1, 0] - r[0, 1]) / scale]
        )
    else:
        diagonal = np.diag(r)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            quaternion = np.array(
                [(r[2, 1] - r[1, 2]) / scale, 0.25 * scale,
                 (r[0, 1] + r[1, 0]) / scale, (r[0, 2] + r[2, 0]) / scale]
            )
        elif axis == 1:
            scale = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            quaternion = np.array(
                [(r[0, 2] - r[2, 0]) / scale, (r[0, 1] + r[1, 0]) / scale,
                 0.25 * scale, (r[1, 2] + r[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            quaternion = np.array(
                [(r[1, 0] - r[0, 1]) / scale, (r[0, 2] + r[2, 0]) / scale,
                 (r[1, 2] + r[2, 1]) / scale, 0.25 * scale]
            )
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("could not convert rotation to quaternion")
    return quaternion / norm


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path quaternion SLERP, returning a rotation matrix."""
    q0 = _rotation_to_quaternion(first)
    q1 = _rotation_to_quaternion(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = q0 + fraction * (q1 - q0)
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        quaternion = (
            math.sin((1.0 - fraction) * angle) / sin_angle * q0
            + math.sin(fraction * angle) / sin_angle * q1
        )
    return project_to_so3(_quaternion_to_rotation(quaternion))


def _invalid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([True], np.asarray(valid, dtype=bool), [True])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == -1)
    ends = np.flatnonzero(transitions == 1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _gap_duration(
    start: int, end: int, timestamps: np.ndarray, median_dt: float
) -> float:
    if end < len(timestamps):
        return float(timestamps[end] - timestamps[start])
    if start > 0:
        return float(timestamps[end - 1] - timestamps[start - 1])
    return float((end - start) * median_dt)


def interpolate_short_rotation_gaps(
    rotations: np.ndarray,
    source_valid: np.ndarray,
    timestamps: np.ndarray,
    max_gap_seconds: float = MAX_INTERPOLATION_GAP_SECONDS,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """SLERP bounded internal invalid runs no longer than max_gap_seconds."""
    values = np.asarray(rotations, dtype=np.float64).copy()
    valid = np.asarray(source_valid, dtype=bool).copy()
    times = np.asarray(timestamps, dtype=np.float64)
    if values.shape != (len(valid), 3, 3) or times.shape != (len(valid),):
        raise ValueError("rotation interpolation inputs are not frame-aligned")
    differences = np.diff(times)
    if len(times) < 2 or np.any(differences <= 0) or not np.isfinite(times).all():
        raise ValueError("timestamps must be finite and strictly increasing")
    median_dt = float(np.median(differences))
    interpolated = 0
    max_invalid_gap = 0.0
    for start, end in _invalid_runs(valid):
        duration = _gap_duration(start, end, times, median_dt)
        max_invalid_gap = max(max_invalid_gap, duration)
        if (
            start == 0
            or end == len(valid)
            or duration > max_gap_seconds + 1e-12
            or not valid[start - 1]
            or not valid[end]
        ):
            continue
        interval = float(times[end] - times[start - 1])
        for index in range(start, end):
            fraction = float((times[index] - times[start - 1]) / interval)
            values[index] = quaternion_slerp(values[start - 1], values[end], fraction)
            valid[index] = True
            interpolated += 1
    values[~valid] = np.nan
    return values, valid, interpolated, max_invalid_gap


def _valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(valid, dtype=bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    return [
        (int(start), int(end))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    ]


def _sos_padlen(sos: np.ndarray) -> int:
    return 3 * (
        2 * len(sos) + 1
        - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    )


def filter_neck_rpy(
    raw_rpy: np.ndarray,
    valid: np.ndarray,
    sampling_frequency_hz: float,
    cutoff_hz: float = FILTER_CUTOFF_HZ,
    order: int = FILTER_ORDER,
) -> tuple[np.ndarray, int]:
    """Apply zero-phase low-pass independently inside each final-valid run."""
    values = np.asarray(raw_rpy, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != (len(mask), 3):
        raise ValueError("RPY and valid must have shapes [T,3] and [T]")
    if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
        raise ValueError("RPY finite/NaN values do not agree with valid")
    if cutoff_hz <= 0 or cutoff_hz >= sampling_frequency_hz / 2.0:
        raise ValueError("low-pass cutoff must be below Nyquist")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sampling_frequency_hz, output="sos")
    padlen = _sos_padlen(sos)
    output = values.copy()
    short_segments = 0
    for start, end in _valid_runs(mask):
        if end - start <= padlen:
            short_segments += 1
        else:
            output[start:end] = sosfiltfilt(sos, output[start:end], axis=0)
    output[~mask] = np.nan
    return output.astype(np.float32), short_segments


def _load_inputs(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = {name: feature_dir / name for name in ("timestamps.npy", "valid.npy", "transforms.npy", "metadata.json")}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise NeckPoseError(f"missing MediaPipe input(s): {', '.join(missing)}")
    timestamps = np.load(paths["timestamps.npy"], allow_pickle=False)
    valid = np.load(paths["valid.npy"], allow_pickle=False)
    transforms = np.load(paths["transforms.npy"], allow_pickle=False)
    with paths["metadata.json"].open("r", encoding="utf-8") as source:
        source_metadata = json.load(source)
    if timestamps.ndim != 1 or valid.shape != timestamps.shape or valid.dtype != np.bool_:
        raise NeckPoseError("timestamps and bool valid mask must have shape [T]")
    if transforms.shape != (len(timestamps), 4, 4):
        raise NeckPoseError(f"expected transforms [T,4,4], got {transforms.shape}")
    if len(timestamps) < 2:
        raise NeckPoseError("at least two frames are required")
    differences = np.diff(np.asarray(timestamps, dtype=np.float64))
    if not np.isfinite(timestamps).all() or np.any(differences <= 0):
        raise NeckPoseError("timestamps must be finite and strictly increasing")
    if not np.isfinite(transforms[valid, :3, :3]).all():
        raise NeckPoseError("a source-valid transform contains non-finite values")
    if int(source_metadata.get("frame_count", -1)) != len(timestamps):
        raise NeckPoseError("MediaPipe metadata frame count does not match arrays")
    return timestamps.astype(np.float64, copy=False), valid, transforms, source_metadata


def _quality_statistics(
    neck_rpy: np.ndarray, timestamps: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    degrees = np.rad2deg(np.asarray(neck_rpy, dtype=np.float64))
    pair_valid = valid[1:] & valid[:-1]
    dt = np.diff(timestamps)
    result: dict[str, Any] = {}
    for column, axis in enumerate(("roll", "pitch", "yaw")):
        angles = degrees[valid, column]
        velocity = np.diff(degrees[:, column])[pair_valid] / dt[pair_valid]
        absolute_velocity = np.abs(velocity)
        result[axis] = {
            "min_degree": float(np.min(angles)) if angles.size else None,
            "max_degree": float(np.max(angles)) if angles.size else None,
            "mean_degree": float(np.mean(angles)) if angles.size else None,
            "std_degree": float(np.std(angles)) if angles.size else None,
            "mean_absolute_angular_velocity_degree_per_second": (
                float(np.mean(absolute_velocity)) if absolute_velocity.size else None
            ),
            "p95_absolute_angular_velocity_degree_per_second": (
                float(np.percentile(absolute_velocity, 95)) if absolute_velocity.size else None
            ),
            "max_absolute_angular_velocity_degree_per_second": (
                float(np.max(absolute_velocity)) if absolute_velocity.size else None
            ),
        }
    return result


def _completed(directory: Path) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in _REQUIRED_FILES):
        return None
    try:
        with (directory / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        frame_count = int(metadata["frame_count"])
        expected = {
            "timestamps.npy": ((frame_count,), np.dtype(np.float64)),
            "valid.npy": ((frame_count,), np.dtype(np.bool_)),
            "relative_rotation.npy": ((frame_count, 3, 3), np.dtype(np.float32)),
            "relative_rpy_raw.npy": ((frame_count, 3), np.dtype(np.float32)),
            "neck_rpy.npy": ((frame_count, 3), np.dtype(np.float32)),
        }
        if metadata.get("status") != "completed" or metadata.get("feature_version") != FEATURE_VERSION:
            return None
        for name, (shape, dtype) in expected.items():
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise NeckPoseError(f"invalid JSONL record in {path}")
                records.append(record)
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
        "frame_count": metadata["frame_count"],
        "duration": metadata["duration"],
        "sampling_frequency_hz": metadata["sampling_frequency_hz"],
        "source_valid_ratio": metadata["source_valid_ratio"],
        "final_valid_ratio": metadata["final_valid_ratio"],
        "interpolated_frames": metadata["interpolated_frames"],
        "remaining_invalid_frames": metadata["remaining_invalid_frames"],
        "neutral_frame_index": metadata["neutral_pose"]["frame_index"],
    }


def process_neck_pose_clip(feature_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Generate one complete Neck Pose V1 package without invoking MediaPipe."""
    timestamps, source_valid, transforms, _ = _load_inputs(feature_dir)
    frame_count = len(timestamps)
    differences = np.diff(timestamps)
    median_dt = float(np.median(differences))
    sampling_frequency_hz = 1.0 / median_dt

    rotations = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    for index in np.flatnonzero(source_valid):
        try:
            rotations[index] = project_to_so3(transforms[index, :3, :3])
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise NeckPoseError(f"SO(3) projection failed at frame {index}: {exc}") from exc
    if not source_valid.any():
        raise NeckPoseError("clip has no valid MediaPipe frame")
    neutral_index = int(np.flatnonzero(source_valid)[0])

    interpolated_rotations, final_valid, interpolated_frames, max_invalid_gap = (
        interpolate_short_rotation_gaps(
            rotations, source_valid, timestamps, MAX_INTERPOLATION_GAP_SECONDS
        )
    )
    relative, neutral_index = relative_rotations(
        interpolated_rotations, final_valid, neutral_index=neutral_index
    )
    if not np.allclose(relative[neutral_index], np.eye(3), atol=1e-6):
        raise NeckPoseError("neutral relative rotation is not identity")

    candidate = np.full((frame_count, 3), np.nan, dtype=np.float64)
    for index in np.flatnonzero(final_valid):
        candidate[index] = rotation_to_zyx_rpy(relative[index])
    human_raw = semantic_candidate_to_human(candidate).astype(np.float32)
    human_raw[~final_valid] = np.nan
    neck_rpy, short_filter_segments = filter_neck_rpy(
        human_raw, final_valid, sampling_frequency_hz
    )

    source_valid_frames = int(source_valid.sum())
    final_valid_frames = int(final_valid.sum())
    remaining_invalid_frames = frame_count - final_valid_frames
    duration = float(timestamps[-1] - timestamps[0] + median_dt)
    metadata: dict[str, Any] = {
        "clip_id": feature_dir.name,
        "frame_count": frame_count,
        "duration": duration,
        "sampling_frequency_hz": sampling_frequency_hz,
        "source_valid_frames": source_valid_frames,
        "source_valid_ratio": source_valid_frames / frame_count,
        "interpolated_frames": interpolated_frames,
        "remaining_invalid_frames": remaining_invalid_frames,
        "max_invalid_gap_seconds": max_invalid_gap,
        "final_valid_frames": final_valid_frames,
        "final_valid_ratio": final_valid_frames / frame_count,
        "neutral_pose": {
            "method": "first_valid_frame",
            "frame_index": neutral_index,
            "timestamp": float(timestamps[neutral_index]),
        },
        "source_rotation_projection": "SVD nearest SO(3)",
        "relative_rotation": "R0.T @ R(t)",
        "candidate_euler_convention": "R = Rz(yaw_z) Ry(pitch_y) Rx(roll_x)",
        "semantic_axis_mapping": {
            "human_roll": "candidate_yaw_z",
            "human_pitch": "candidate_roll_x",
            "human_yaw": "candidate_pitch_y",
        },
        "output_order": ["roll", "pitch", "yaw"],
        "unit": "radian",
        "filter": {
            "type": "Butterworth low-pass",
            "order": FILTER_ORDER,
            "cutoff_hz": FILTER_CUTOFF_HZ,
            "zero_phase": True,
            "implementation": "scipy.signal.sosfiltfilt",
            "short_segment_count": short_filter_segments,
        },
        "interpolation": {
            "method": "rotation SLERP",
            "max_gap_seconds": MAX_INTERPOLATION_GAP_SECONDS,
        },
        "quality_statistics": _quality_statistics(neck_rpy, timestamps, final_valid),
        "feature_version": FEATURE_VERSION,
        "status": "completed",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{feature_dir.name}.", dir=output_dir.parent))
    try:
        np.save(temporary / "timestamps.npy", timestamps, allow_pickle=False)
        np.save(temporary / "valid.npy", final_valid.astype(np.bool_), allow_pickle=False)
        np.save(temporary / "relative_rotation.npy", relative.astype(np.float32), allow_pickle=False)
        np.save(temporary / "relative_rpy_raw.npy", human_raw, allow_pickle=False)
        np.save(temporary / "neck_rpy.npy", neck_rpy, allow_pickle=False)
        with (temporary / "metadata.json").open("w", encoding="utf-8") as target:
            json.dump(metadata, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def _print_clip(metadata: dict[str, Any], status: str) -> None:
    print(f"Frames: {metadata['frame_count']}")
    print(f"Source valid: {metadata['source_valid_frames']}")
    print(f"Interpolated: {metadata['interpolated_frames']}")
    print(f"Final valid: {metadata['final_valid_frames']}")
    print(f"Final valid ratio: {metadata['final_valid_ratio']:.2%}")
    print(f"Neutral frame: {metadata['neutral_pose']['frame_index']}")
    print(f"Status: {status}", flush=True)


def extract_neck_pose_clips(
    feature_dirs: list[Path], output_root: Path, force: bool = False
) -> dict[str, Any]:
    """Run Neck Pose V1 with atomic clip output, resume, and failure isolation."""
    output_root = output_root.resolve()
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"
    failed_path = output_root / "failed.jsonl"
    successful = {str(item["clip_id"]): item for item in _read_jsonl(metadata_path) if item.get("clip_id")}
    failures = {str(item["clip_id"]): item for item in _read_jsonl(failed_path) if item.get("clip_id")}
    processed = skipped = failed = 0
    total_frames = source_valid_total = interpolated_total = final_valid_total = 0

    for index, feature_dir in enumerate(feature_dirs, start=1):
        clip_id = feature_dir.name
        output_dir = clips_root / clip_id
        print(f"[{index:03d}/{len(feature_dirs):03d}] {clip_id}", flush=True)
        completed = _completed(output_dir)
        if completed is not None and not force:
            skipped += 1
            successful[clip_id] = _global_record(completed, output_dir)
            failures.pop(clip_id, None)
            total_frames += int(completed["frame_count"])
            source_valid_total += int(completed["source_valid_frames"])
            interpolated_total += int(completed["interpolated_frames"])
            final_valid_total += int(completed["final_valid_frames"])
            _print_clip(completed, "skipped")
            continue
        try:
            metadata = process_neck_pose_clip(feature_dir, output_dir)
            successful[clip_id] = _global_record(metadata, output_dir)
            failures.pop(clip_id, None)
            processed += 1
            total_frames += int(metadata["frame_count"])
            source_valid_total += int(metadata["source_valid_frames"])
            interpolated_total += int(metadata["interpolated_frames"])
            final_valid_total += int(metadata["final_valid_frames"])
            _print_clip(metadata, "done")
        except Exception as exc:
            failed += 1
            successful.pop(clip_id, None)
            failures[clip_id] = {
                "clip_id": clip_id,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "feature_version": FEATURE_VERSION,
            }
            print("Status: failed")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        _write_jsonl(metadata_path, list(successful.values()))
        _write_jsonl(failed_path, list(failures.values()))

    _write_jsonl(metadata_path, list(successful.values()))
    _write_jsonl(failed_path, list(failures.values()))
    return {
        "input_clips": len(feature_dirs),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_frames": total_frames,
        "source_valid_frames": source_valid_total,
        "interpolated_frames": interpolated_total,
        "remaining_invalid_frames": total_frames - final_valid_total,
        "final_valid_ratio": final_valid_total / total_frames if total_frames else 0.0,
    }
