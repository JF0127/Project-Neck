"""Experimental comparison of Butterworth smoothing for candidate neck Euler tracks."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.features.neck_pose_v0 import _draw_axes

VERSION = "neck_smoothing_v0"
FILTER_ORDER = 4
CUTOFFS_HZ = (1.0, 1.5, 2.0)
AXES = ("roll_x", "pitch_y", "yaw_z")
_MODES = (
    ("raw", "RAW", "raw.npy"),
    ("1p0hz", "LPF 1.0Hz", "lowpass_1p0hz.npy"),
    ("1p5hz", "LPF 1.5Hz", "lowpass_1p5hz.npy"),
    ("2p0hz", "LPF 2.0Hz", "lowpass_2p0hz.npy"),
)
_REQUIRED_OUTPUTS = tuple(item[2] for item in _MODES) + (
    "compare_roll_x.png",
    "compare_pitch_y.png",
    "compare_yaw_z.png",
    "overview.png",
    "debug_raw.mp4",
    "debug_1p0hz.mp4",
    "debug_1p5hz.mp4",
    "debug_2p0hz.mp4",
    "metadata.json",
)


class NeckSmoothingV0Error(RuntimeError):
    """Raised when a smoothing comparison cannot be generated safely."""


def _valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(valid, dtype=bool), [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _default_sos_padlen(sos: np.ndarray) -> int:
    # This is scipy.signal.sosfiltfilt's documented default padding expression.
    zeros_at_origin = int((sos[:, 2] == 0).sum())
    poles_at_origin = int((sos[:, 5] == 0).sum())
    return 3 * (2 * len(sos) + 1 - min(zeros_at_origin, poles_at_origin))


def filter_rpy_comparison(
    raw_rpy: np.ndarray,
    valid: np.ndarray,
    sampling_frequency_hz: float,
    cutoffs_hz: tuple[float, ...] = CUTOFFS_HZ,
    order: int = FILTER_ORDER,
) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Filter each contiguous valid run independently; short runs remain raw."""
    raw = np.asarray(raw_rpy)
    mask = np.asarray(valid)
    if raw.ndim != 2 or raw.shape[1] != 3 or mask.shape != (len(raw),):
        raise ValueError("raw_rpy and valid must have shapes [T,3] and [T]")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling frequency must be positive and finite")
    if not np.isfinite(raw[mask]).all():
        raise ValueError("valid frames must contain finite RPY values")
    if not np.isnan(raw[~mask]).all():
        raise ValueError(
            "invalid frames must contain NaN RPY values; interpolation is not allowed"
        )
    runs = _valid_runs(mask)
    outputs: dict[float, np.ndarray] = {}
    short_counts: dict[float, int] = {}
    for cutoff in cutoffs_hz:
        cutoff = float(cutoff)
        if cutoff <= 0 or cutoff >= sampling_frequency_hz / 2.0:
            raise ValueError(
                f"cutoff {cutoff:g} Hz must be below Nyquist ({sampling_frequency_hz / 2:g} Hz)"
            )
        sos = butter(order, cutoff, btype="lowpass", fs=sampling_frequency_hz, output="sos")
        padlen = _default_sos_padlen(sos)
        filtered = raw.astype(np.float64, copy=True)
        short = 0
        for start, end in runs:
            if end - start <= padlen:
                short += 1
                continue
            filtered[start:end] = sosfiltfilt(sos, filtered[start:end], axis=0)
        filtered[~mask] = np.nan
        outputs[cutoff] = filtered.astype(np.float32)
        short_counts[cutoff] = short
    return outputs, short_counts


def euler_zyx_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """Build Rz(yaw) Ry(pitch) Rx(roll) from candidate [roll,pitch,yaw]."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _load_inputs(
    pose_dir: Path, mediapipe_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = {
        "raw_rpy": pose_dir / "raw_rpy.npy",
        "pose metadata": pose_dir / "metadata.json",
        "timestamps": mediapipe_dir / "timestamps.npy",
        "valid": mediapipe_dir / "valid.npy",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise NeckSmoothingV0Error(f"missing input(s): {', '.join(missing)}")
    raw = np.load(paths["raw_rpy"], allow_pickle=False)
    timestamps = np.load(paths["timestamps"], allow_pickle=False)
    valid = np.load(paths["valid"], allow_pickle=False)
    with paths["pose metadata"].open("r", encoding="utf-8") as source:
        pose_metadata = json.load(source)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype != np.float32:
        raise NeckSmoothingV0Error(f"raw_rpy must be float32 [T,3], got {raw.shape} {raw.dtype}")
    if timestamps.ndim != 1 or valid.ndim != 1 or valid.dtype != np.bool_:
        raise NeckSmoothingV0Error("timestamps/valid must be [T] with a bool valid mask")
    lengths = (len(raw), len(timestamps), len(valid))
    if len(set(lengths)) != 1:
        raise NeckSmoothingV0Error(
            "alignment error: raw_rpy/timestamps/valid lengths are "
            + "/".join(str(value) for value in lengths)
        )
    if len(timestamps) < 2:
        raise NeckSmoothingV0Error("at least two timestamps are required")
    differences = np.diff(np.asarray(timestamps, dtype=np.float64))
    if not np.isfinite(timestamps).all() or np.any(differences <= 0):
        raise NeckSmoothingV0Error("timestamps must be finite and strictly increasing")
    if not np.isfinite(raw[valid]).all() or not np.isnan(raw[~valid]).all():
        raise NeckSmoothingV0Error("raw RPY does not agree with the valid/NaN mask")
    if int(pose_metadata.get("frame_count", -1)) != len(raw):
        raise NeckSmoothingV0Error("alignment error: Neck Pose V0 metadata frame count differs")
    return raw, timestamps.astype(np.float64, copy=False), valid, pose_metadata


def _statistics(values: np.ndarray, timestamps: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    degrees = np.rad2deg(np.asarray(values, dtype=np.float64))
    result: dict[str, Any] = {}
    for column, axis in enumerate(AXES):
        angles = degrees[valid, column]
        pair_valid = valid[1:] & valid[:-1]
        dt = np.diff(timestamps)
        velocities = np.diff(degrees[:, column])[pair_valid] / dt[pair_valid]
        triple_valid = valid[2:] & valid[1:-1] & valid[:-2]
        previous_velocity = np.diff(degrees[:, column])[:-1] / dt[:-1]
        next_velocity = np.diff(degrees[:, column])[1:] / dt[1:]
        midpoint_dt = (dt[:-1] + dt[1:]) / 2.0
        accelerations = (next_velocity - previous_velocity)[triple_valid] / midpoint_dt[triple_valid]
        absolute_velocity = np.abs(velocities)
        absolute_acceleration = np.abs(accelerations)
        result[axis] = {
            "angle_range_degree": float(np.ptp(angles)) if angles.size else None,
            "mean_absolute_angular_velocity_degree_per_second": (
                float(np.mean(absolute_velocity)) if absolute_velocity.size else None
            ),
            "p95_absolute_angular_velocity_degree_per_second": (
                float(np.percentile(absolute_velocity, 95)) if absolute_velocity.size else None
            ),
            "max_absolute_angular_velocity_degree_per_second": (
                float(np.max(absolute_velocity)) if absolute_velocity.size else None
            ),
            "mean_absolute_angular_acceleration_degree_per_second2": (
                float(np.mean(absolute_acceleration)) if absolute_acceleration.size else None
            ),
            "p95_absolute_angular_acceleration_degree_per_second2": (
                float(np.percentile(absolute_acceleration, 95)) if absolute_acceleration.size else None
            ),
        }
    return result


def _plot_comparisons(
    directory: Path,
    clip_id: str,
    timestamps: np.ndarray,
    tracks: dict[str, np.ndarray],
    sampling_frequency_hz: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = (("raw", "Raw"), ("1p0hz", "1.0 Hz"), ("1p5hz", "1.5 Hz"), ("2p0hz", "2.0 Hz"))
    for column, axis_name in enumerate(AXES):
        figure, axis = plt.subplots(figsize=(12, 5), dpi=140)
        for key, label in styles:
            axis.plot(timestamps, np.rad2deg(tracks[key][:, column]), label=label, linewidth=1.0)
        axis.set_title(f"{clip_id} | {axis_name} | fs={sampling_frequency_hz:.4f} Hz")
        axis.set_xlabel("time / second")
        axis.set_ylabel("angle / degree")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(directory / f"compare_{axis_name}.png")
        plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(13, 13), dpi=140, sharex=True)
    colors = ("red", "green", "blue")
    for plot_axis, (key, label) in zip(axes, styles):
        for column, (axis_name, color) in enumerate(zip(AXES, colors)):
            plot_axis.plot(
                timestamps, np.rad2deg(tracks[key][:, column]), label=axis_name,
                color=color, linewidth=0.9,
            )
        plot_axis.set_ylabel("degree")
        plot_axis.set_title(label, loc="left")
        plot_axis.grid(True, alpha=0.25)
        plot_axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("time / second")
    figure.suptitle(f"{clip_id} | candidate axes | fs={sampling_frequency_hz:.4f} Hz")
    figure.tight_layout()
    figure.savefig(directory / "overview.png")
    plt.close(figure)


def _draw_reconstructed_axes(frame: np.ndarray, rpy: np.ndarray) -> None:
    """Reuse the V0 XYZ projection, while identifying its reconstructed source."""
    _draw_axes(frame, euler_zyx_to_rotation(rpy))
    width = frame.shape[1]
    left = max(0, width - 335)
    cv2.rectangle(frame, (left, 4), (width - 1, 31), (0, 0, 0), -1)
    cv2.putText(
        frame, "Candidate Euler reconstructed diagnostic", (left + 5, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _annotated_video(
    source_video: Path,
    output: Path,
    timestamps: np.ndarray,
    valid: np.ndarray,
    values: np.ndarray,
    mode_label: str,
) -> tuple[float, int, int]:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        capture.release()
        raise NeckSmoothingV0Error(f"could not open Clean V1 video: {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_count = len(values)
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise NeckSmoothingV0Error("video has invalid FPS or resolution")
    if reported_count > 0 and reported_count != frame_count:
        capture.release()
        raise NeckSmoothingV0Error(
            f"alignment error: video reports {reported_count} frames, arrays have {frame_count}"
        )
    silent = output.with_name(output.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise NeckSmoothingV0Error(f"could not create temporary debug video: {silent}")
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise NeckSmoothingV0Error(
                    f"alignment error: video ended at frame {index}, expected {frame_count}"
                )
            cv2.putText(frame, f"mode = {mode_label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame = {index} | time = {timestamps[index]:.3f} s", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            if valid[index]:
                degrees = np.rad2deg(values[index])
                for line, (axis, value) in enumerate(zip(AXES, degrees)):
                    cv2.putText(frame, f"{axis} = {value:.1f} deg", (15, 88 + line * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                _draw_reconstructed_axes(frame, values[index])
            else:
                cv2.putText(frame, "VALID = FALSE | RPY = N/A", (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        extra, _ = capture.read()
        if extra:
            raise NeckSmoothingV0Error(
                f"alignment error: video contains more than {frame_count} frames"
            )
    finally:
        writer.release()
        capture.release()
    try:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent), "-i", str(source_video),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy",
            "-map_metadata", "1", "-shortest", "-movflags", "+faststart", str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            message = result.stderr.strip() or "ffmpeg produced no output"
            raise NeckSmoothingV0Error(f"audio mux failed: {message[:1000]}")
    finally:
        silent.unlink(missing_ok=True)
    return fps, width, height


def _completed(directory: Path) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in _REQUIRED_OUTPUTS):
        return None
    try:
        with (directory / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        frame_count = int(metadata["frame_count"])
        if metadata.get("status") != "completed" or metadata.get("analysis_version") != VERSION:
            return None
        for _, _, filename in _MODES:
            array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
            if array.shape != (frame_count, 3) or array.dtype != np.float32:
                return None
        if any((directory / name).stat().st_size == 0 for name in _REQUIRED_OUTPUTS[4:-1]):
            return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def process_smoothing_clip(
    pose_dir: Path, mediapipe_dir: Path, video_path: Path, output_dir: Path
) -> dict[str, Any]:
    raw, timestamps, valid, _ = _load_inputs(pose_dir, mediapipe_dir)
    dt = float(np.median(np.diff(timestamps)))
    sampling_frequency_hz = 1.0 / dt
    filtered, short_counts = filter_rpy_comparison(raw, valid, sampling_frequency_hz)
    tracks = {
        "raw": raw.copy(),
        "1p0hz": filtered[1.0],
        "1p5hz": filtered[1.5],
        "2p0hz": filtered[2.0],
    }
    clip_id = pose_dir.name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output_dir.parent))
    try:
        for key, _, filename in _MODES:
            np.save(temporary / filename, tracks[key], allow_pickle=False)
        _plot_comparisons(temporary, clip_id, timestamps, tracks, sampling_frequency_hz)
        video_info: tuple[float, int, int] | None = None
        for key, label, _ in _MODES:
            video_info = _annotated_video(
                video_path, temporary / f"debug_{key}.mp4", timestamps, valid, tracks[key], label
            )
        assert video_info is not None
        fps, width, height = video_info
        statistics = {key: _statistics(values, timestamps, valid) for key, values in tracks.items()}
        metadata: dict[str, Any] = {
            "clip_id": clip_id,
            "frame_count": len(raw),
            "sampling_frequency_hz": sampling_frequency_hz,
            "video_fps": fps,
            "video_resolution": [width, height],
            "filters": {
                "type": "Butterworth low-pass",
                "order": FILTER_ORDER,
                "zero_phase": True,
                "implementation": "scipy.signal.sosfiltfilt",
                "cutoffs_hz": list(CUTOFFS_HZ),
                "short_segment_counts": {f"{cutoff:.1f}": short_counts[cutoff] for cutoff in CUTOFFS_HZ},
            },
            "smoothness_statistics": statistics,
            "statistics_note": "Diagnostic candidate Euler statistics; velocity/acceleration never cross invalid gaps.",
            "interpolation": False,
            "neutral_normalization": False,
            "semantic_axis_mapping": False,
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


def compare_smoothing_clips(
    pose_dirs: list[Path], mediapipe_root: Path, video_root: Path,
    output_root: Path, force: bool = False,
) -> dict[str, int]:
    """Generate smoothing diagnostics with clip-level resume and failure isolation."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processed = skipped = failed = 0
    for index, pose_dir in enumerate(pose_dirs, start=1):
        clip_id = pose_dir.name
        output_dir = output_root / clip_id
        print(f"[{index:03d}/{len(pose_dirs):03d}] {clip_id}", flush=True)
        if _completed(output_dir) is not None and not force:
            skipped += 1
            print("Status: skipped", flush=True)
            continue
        try:
            metadata = process_smoothing_clip(
                pose_dir, mediapipe_root / clip_id, video_root / f"{clip_id}.mp4", output_dir
            )
            processed += 1
            print(f"Frames: {metadata['frame_count']}")
            print(f"Sampling frequency: {metadata['sampling_frequency_hz']:.4f} Hz")
            print("Status: done", flush=True)
        except Exception as exc:
            failed += 1
            print("Status: failed")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
    return {"input_clips": len(pose_dirs), "processed": processed, "skipped": skipped, "failed": failed}
