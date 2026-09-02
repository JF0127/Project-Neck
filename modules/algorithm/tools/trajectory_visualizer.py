#!/usr/bin/env python3
"""Read-only diagnostics for Algorithm -> Neck Interface V1 trajectories."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "matplotlib is required for trajectory visualization; install it with: pip install matplotlib"
    ) from exc
import numpy as np

AXES = ("roll", "pitch", "yaw")
AXIS_COLORS = {"roll": "#1f77b4", "pitch": "#ff7f0e", "yaw": "#2ca02c"}
SOURCE_STYLES = {"measured": "-", "predicted": "--", "processed": ":"}
SOURCE_COLORS = {"measured": "#1f77b4", "predicted": "#d62728", "processed": "#2ca02c"}
VALID_STATES = {"listening", "speaking", "silent"}
STATE_COLORS = {"listening": "#d9edf7", "speaking": "#fce8b2", "silent": "#eeeeee"}
RAD2DEG = 180.0 / math.pi
TOOL_VERSION = "2.0.0"
EXPERIMENT_TYPE = "v0_trajectory"
RUN_PATTERN = re.compile(r"^run_(\d{8})_(\d{3,})$")
SEGMENT_PATTERN = re.compile(r"^segment_(\d+)_")
ROLE_FROM_STATE = {"listening": "listener", "speaking": "speaker", "silent": "silent"}
VALID_ROLES = set(ROLE_FROM_STATE.values())


def load_trajectory(path: Path) -> tuple[dict, np.ndarray, list[str] | None]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("JSON root must be an object")

    fps = document.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite number greater than zero")
    if document.get("unit") != "radian":
        raise ValueError("unit must be 'radian'")
    if document.get("order") != list(AXES):
        raise ValueError('order must be ["roll", "pitch", "yaw"]')

    try:
        trajectory = np.asarray(document.get("trajectory"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must contain numeric values") from exc
    if trajectory.ndim != 2 or trajectory.shape[1:] != (3,) or len(trajectory) == 0:
        raise ValueError(f"trajectory shape must be [N,3] with N>0, got {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains NaN or Inf")

    states = document.get("states")
    if states is not None:
        if not isinstance(states, list) or len(states) != len(trajectory):
            raise ValueError("states must be an array with the same length as trajectory")
        invalid = sorted({state for state in states if state not in VALID_STATES})
        if invalid:
            raise ValueError(f"states contains unsupported values: {invalid}")
    return document, trajectory, states


def state_segments(states: list[str] | None) -> list[tuple[int, int, str]]:
    if not states:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(states)):
        if states[index] != states[start]:
            segments.append((start, index, states[start]))
            start = index
    segments.append((start, len(states), states[start]))
    return segments


def shade_states(axis, states: list[str] | None, fps: float,
                 start_sec: float = 0.0) -> None:
    labels_seen: set[str] = set()
    for start, end, state in state_segments(states):
        label = state if state not in labels_seen else None
        axis.axvspan(start_sec + start / fps, start_sec + end / fps,
                     color=STATE_COLORS[state], alpha=0.35, linewidth=0, label=label)
        labels_seen.add(state)


def save_plot(path: Path, times: np.ndarray, values: np.ndarray, ylabel: str,
              title: str, states: list[str] | None, fps: float,
              start_sec: float = 0.0) -> None:
    figure = plt.figure(figsize=(13, 6))
    axis = figure.add_axes((0.08, 0.12, 0.89, 0.80))
    shade_states(axis, states, fps, start_sec)
    for column, name in enumerate(AXES):
        axis.plot(times, values[:, column], label=name, linewidth=1.2)
    axis.set_xlabel("time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def trajectory_times(document: dict, trajectory: np.ndarray,
                     start_sec: float | None) -> tuple[np.ndarray, float]:
    if start_sec is None:
        start_sec = document.get("turn_start_sec", 0.0)
    if (isinstance(start_sec, bool) or not isinstance(start_sec, (int, float)) or
            not math.isfinite(start_sec) or start_sec < 0):
        raise ValueError("turn-relative start time must be a finite number greater than or equal to zero")
    value = float(start_sec)
    return value + np.arange(len(trajectory)) / float(document["fps"]), value


def save_rpy_combined(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]],
                      title: str, time_limits: tuple[float, float]) -> None:
    figure, axis = plt.subplots(figsize=(13, 6))
    for source, (times, trajectory) in series.items():
        for column, name in enumerate(AXES):
            axis.plot(times, trajectory[:, column] * RAD2DEG,
                      color=AXIS_COLORS[name], linestyle=SOURCE_STYLES.get(source, "-"),
                      label=f"{source} {name.capitalize()}", linewidth=1.2)
    axis.set_xlabel("Turn relative time (s)")
    axis.set_ylabel("RPY angle (degree)")
    axis.set_title(title)
    axis.set_xlim(time_limits)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_rpy_subplots(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]],
                      title: str, time_limits: tuple[float, float]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for column, (axis, name) in enumerate(zip(axes, AXES)):
        for source, (times, trajectory) in series.items():
            axis.plot(times, trajectory[:, column] * RAD2DEG,
                      color=SOURCE_COLORS.get(source), linestyle=SOURCE_STYLES.get(source, "-"),
                      label=source, linewidth=1.2)
        axis.set_ylabel(f"{name.capitalize()} (degree)")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Turn relative time (s)")
    axes[-1].set_xlim(time_limits)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_rpy_visualizations(output_dir: Path,
                            series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    end_sec = max(times[-1] for times, _ in series.values())
    time_limits = (0.0, end_sec if end_sec > 0.0 else 1.0)
    for source, values in series.items():
        source_series = {source: values}
        save_rpy_combined(
            output_dir / f"{source}_rpy_combined.png", source_series,
            f"{source.capitalize()} RPY", time_limits,
        )
        save_rpy_subplots(
            output_dir / f"{source}_rpy_axes.png", source_series,
            f"{source.capitalize()} RPY by Axis", time_limits,
        )
    if len(series) > 1:
        sources = "_".join(series)
        source_names = " and ".join(source.capitalize() for source in series)
        save_rpy_combined(
            output_dir / f"{sources}_rpy_combined.png", series,
            f"{source_names} RPY", time_limits,
        )
        save_rpy_subplots(
            output_dir / f"{sources}_rpy_axes.png", series,
            f"{source_names} RPY by Axis", time_limits,
        )


def _maximum(values: np.ndarray, frame_offset: int) -> tuple[float, int]:
    if values.size == 0:
        return 0.0, 0
    flat_index = int(np.argmax(np.abs(values)))
    frame, _ = np.unravel_index(flat_index, values.shape)
    return float(np.abs(values).reshape(-1)[flat_index]), int(frame + frame_offset)


def analyze(document: dict, trajectory: np.ndarray, states: list[str] | None) -> tuple[dict, np.ndarray, np.ndarray]:
    fps = float(document["fps"])
    delta = np.diff(trajectory, axis=0)                         # destination frame i+1
    velocity = delta * fps                                     # aligned to frames 1..N-1
    acceleration = np.diff(velocity, axis=0) * fps             # aligned to frames 2..N-1

    axes: dict[str, Any] = {}
    for column, name in enumerate(AXES):
        position = trajectory[:, column]
        axis_delta = delta[:, column]
        axis_velocity = velocity[:, column]
        axis_acceleration = acceleration[:, column]
        max_delta, delta_frame = _maximum(axis_delta[:, None], 1)
        max_velocity, velocity_frame = _maximum(axis_velocity[:, None], 1)
        max_acceleration, acceleration_frame = _maximum(axis_acceleration[:, None], 2)
        axes[name] = {
            "position_radian": {
                "min": float(position.min()),
                "max": float(position.max()),
                "mean": float(position.mean()),
                "max_absolute_amplitude": float(np.abs(position).max()),
            },
            "frame_delta_radian": {
                "max_absolute": max_delta,
                "frame_index": delta_frame,
            },
            "velocity": {
                "max_absolute_rad_s": max_velocity,
                "mean_absolute_rad_s": float(np.abs(axis_velocity).mean()) if axis_velocity.size else 0.0,
                "max_absolute_deg_s": max_velocity * RAD2DEG,
                "mean_absolute_deg_s": float(np.abs(axis_velocity).mean()) * RAD2DEG if axis_velocity.size else 0.0,
                "frame_index": velocity_frame,
            },
            "acceleration": {
                "max_absolute_rad_s2": max_acceleration,
                "mean_absolute_rad_s2": float(np.abs(axis_acceleration).mean()) if axis_acceleration.size else 0.0,
                "max_absolute_deg_s2": max_acceleration * RAD2DEG,
                "mean_absolute_deg_s2": float(np.abs(axis_acceleration).mean()) * RAD2DEG if axis_acceleration.size else 0.0,
                "frame_index": acceleration_frame,
            },
        }

    def global_anomaly(values: np.ndarray, offset: int, unit: str) -> dict:
        if values.size == 0:
            return {"axis": None, "frame_index": 0, "value": 0.0, "unit": unit}
        flat = int(np.argmax(np.abs(values)))
        frame, axis = np.unravel_index(flat, values.shape)
        return {
            "axis": AXES[axis],
            "frame_index": int(frame + offset),
            "value": float(abs(values[frame, axis])),
            "unit": unit,
        }

    summary = {
        "name": document.get("name"),
        "fps": fps,
        "frame_count": int(len(trajectory)),
        "duration_sec": float(len(trajectory) / fps),
        "unit": document["unit"],
        "order": document["order"],
        "state_segments": [
            {"state": state, "start_frame": start, "end_frame_exclusive": end,
             "start_sec": start / fps, "end_sec": end / fps}
            for start, end, state in state_segments(states)
        ],
        "axes": axes,
        "maximum_anomalies": {
            "frame_delta": global_anomaly(delta, 1, "radian/frame"),
            "velocity": global_anomaly(velocity, 1, "rad/s"),
            "acceleration": global_anomaly(acceleration, 2, "rad/s^2"),
        },
    }
    return summary, velocity, acceleration


def find_project_root() -> Path:
    """Locate Project-Neck from this script, without depending on the caller cwd."""
    script = Path(__file__).resolve()
    for candidate in script.parents:
        if (candidate / "modules/algorithm").is_dir() and (candidate / "modules/motor").is_dir():
            return candidate
    raise RuntimeError("cannot locate Project-Neck root from trajectory_visualizer.py")


def git_commit(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def create_automatic_run(runs_dir: Path, timestamp: datetime) -> tuple[Path, str, bool]:
    """Create run_YYYYMMDD_XXX atomically; numbering restarts each local day."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    date = timestamp.strftime("%Y%m%d")
    numbers = []
    for child in runs_dir.iterdir():
        match = RUN_PATTERN.match(child.name) if child.is_dir() else None
        if match and match.group(1) == date:
            numbers.append(int(match.group(2)))
    number = max(numbers, default=0) + 1
    while True:
        run_id = f"run_{date}_{number:03d}"
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir()
            return run_dir, run_id, True
        except FileExistsError:
            number += 1


def resolve_run(runs_dir: Path, requested: str | None,
                timestamp: datetime) -> tuple[Path, str, bool]:
    if requested is None:
        return create_automatic_run(runs_dir, timestamp)
    if not RUN_PATTERN.fullmatch(requested):
        raise ValueError("--run must use run_YYYYMMDD_XXX format")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / requested
    try:
        run_dir.mkdir()
        created = True
    except FileExistsError:
        if not run_dir.is_dir():
            raise ValueError(f"run path exists but is not a directory: {run_dir}")
        created = False
    return run_dir, requested, created


def create_segment(run_dir: Path, role: str) -> tuple[Path, str, int]:
    numbers = []
    for child in run_dir.iterdir():
        match = SEGMENT_PATTERN.match(child.name) if child.is_dir() else None
        if match:
            numbers.append(int(match.group(1)))
    index = max(numbers, default=0) + 1
    while True:
        segment_id = f"segment_{index:03d}_{role}"
        segment_dir = run_dir / segment_id
        try:
            segment_dir.mkdir()
            return segment_dir, segment_id, index
        except FileExistsError:
            index += 1


def segment_count(run_dir: Path) -> int:
    return sum(
        1 for child in run_dir.iterdir()
        if child.is_dir() and SEGMENT_PATTERN.match(child.name)
    )


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def infer_role(requested: str | None, states: list[str] | None) -> str:
    if requested is not None:
        return requested
    unique = set(states or [])
    if len(unique) == 1:
        return ROLE_FROM_STATE[next(iter(unique))]
    raise ValueError("--role is required when trajectory states do not identify one segment type")


def write_raw_csv(path: Path, trajectory: np.ndarray, states: list[str] | None,
                  fps: float) -> None:
    count = len(trajectory)
    delta = np.full((count, 3), np.nan, dtype=np.float64)
    velocity = np.full((count, 3), np.nan, dtype=np.float64)
    acceleration = np.full((count, 3), np.nan, dtype=np.float64)
    if count > 1:
        delta[1:] = np.diff(trajectory, axis=0)
        velocity[1:] = delta[1:] * fps
    if count > 2:
        acceleration[2:] = np.diff(velocity[1:], axis=0) * fps

    fields = [
        "frame", "time_sec", "state",
        "roll_rad", "pitch_rad", "yaw_rad",
        "roll_deg", "pitch_deg", "yaw_deg",
        "delta_roll_rad", "delta_pitch_rad", "delta_yaw_rad",
        "velocity_roll_rad_s", "velocity_pitch_rad_s", "velocity_yaw_rad_s",
        "acceleration_roll_rad_s2", "acceleration_pitch_rad_s2", "acceleration_yaw_rad_s2",
    ]

    def optional(value: float) -> float | str:
        return float(value) if math.isfinite(float(value)) else ""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(count):
            writer.writerow({
                "frame": frame,
                "time_sec": frame / fps,
                "state": states[frame] if states is not None else "",
                "roll_rad": trajectory[frame, 0],
                "pitch_rad": trajectory[frame, 1],
                "yaw_rad": trajectory[frame, 2],
                "roll_deg": trajectory[frame, 0] * RAD2DEG,
                "pitch_deg": trajectory[frame, 1] * RAD2DEG,
                "yaw_deg": trajectory[frame, 2] * RAD2DEG,
                "delta_roll_rad": optional(delta[frame, 0]),
                "delta_pitch_rad": optional(delta[frame, 1]),
                "delta_yaw_rad": optional(delta[frame, 2]),
                "velocity_roll_rad_s": optional(velocity[frame, 0]),
                "velocity_pitch_rad_s": optional(velocity[frame, 1]),
                "velocity_yaw_rad_s": optional(velocity[frame, 2]),
                "acceleration_roll_rad_s2": optional(acceleration[frame, 0]),
                "acceleration_pitch_rad_s2": optional(acceleration[frame, 1]),
                "acceleration_yaw_rad_s2": optional(acceleration[frame, 2]),
            })


def explicit_checkpoint(document: dict) -> str | None:
    value = document.get("checkpoint")
    if isinstance(value, str) and value.strip():
        return value
    meta = document.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("checkpoint"), str):
        return meta["checkpoint"] or None
    return None


def print_report(summary: dict, trajectory: np.ndarray, velocity: np.ndarray) -> None:
    print("=== Trajectory ===")
    print(f"name        : {summary['name']}")
    print(f"frame count : {summary['frame_count']}")
    print(f"duration    : {summary['duration_sec']:.6f} s")
    print(f"fps         : {summary['fps']:.6f}")
    for name in AXES:
        stats = summary["axes"][name]
        position = stats["position_radian"]
        delta = stats["frame_delta_radian"]
        vel = stats["velocity"]
        acc = stats["acceleration"]
        print(f"\n[{name}]")
        print(f"  min/max/mean                 : {position['min']:+.8f} / {position['max']:+.8f} / {position['mean']:+.8f} rad")
        print(f"  max absolute amplitude       : {position['max_absolute_amplitude']:.8f} rad")
        print(f"  max frame delta              : {delta['max_absolute']:.8f} rad @ frame {delta['frame_index']}")
        print(f"  max / mean absolute velocity : {vel['max_absolute_rad_s']:.8f} / {vel['mean_absolute_rad_s']:.8f} rad/s")
        print(f"                                {vel['max_absolute_deg_s']:.4f} / {vel['mean_absolute_deg_s']:.4f} deg/s")
        print(f"  max / mean abs acceleration  : {acc['max_absolute_rad_s2']:.8f} / {acc['mean_absolute_rad_s2']:.8f} rad/s^2")
        print(f"                                {acc['max_absolute_deg_s2']:.4f} / {acc['mean_absolute_deg_s2']:.4f} deg/s^2")

    print("\n=== First 10 frames ===")
    print("frame |       roll       pitch         yaw | delta [roll,pitch,yaw] | velocity rad/s [roll,pitch,yaw]")
    for index in range(min(10, len(trajectory))):
        position = trajectory[index]
        if index == 0:
            delta_text = "[       -,        -,        -]"
            velocity_text = "[       -,        -,        -]"
        else:
            frame_delta = trajectory[index] - trajectory[index - 1]
            frame_velocity = velocity[index - 1]
            delta_text = "[" + ", ".join(f"{value:+.6f}" for value in frame_delta) + "]"
            velocity_text = "[" + ", ".join(f"{value:+.6f}" for value in frame_velocity) + "]"
        print(f"{index:5d} | {position[0]:+11.7f} {position[1]:+11.7f} {position[2]:+11.7f} | {delta_text} | {velocity_text}")

    anomaly = summary["maximum_anomalies"]["velocity"]
    frame = anomaly["frame_index"]
    print("\n=== Maximum absolute velocity ===")
    print(f"axis={anomaly['axis']}, frame={frame}, value={anomaly['value']:.8f} rad/s ({anomaly['value'] * RAD2DEG:.4f} deg/s)")
    if 0 <= frame < len(trajectory):
        print(f"RPY at frame {frame}: {trajectory[frame].tolist()}")
        print(f"delta from frame {frame - 1}: {(trajectory[frame] - trajectory[frame - 1]).tolist() if frame else [0.0, 0.0, 0.0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and record a machine-head trajectory segment")
    parser.add_argument("trajectory_json", type=Path, nargs="?",
                        help="predicted RPY trajectory JSON")
    parser.add_argument("--measured-rpy", type=Path, default=None,
                        help="measured/real RPY trajectory JSON")
    parser.add_argument("--predicted-start-sec", type=float, default=None,
                        help="predicted trajectory start on the Turn-relative timeline")
    parser.add_argument("--measured-start-sec", type=float, default=None,
                        help="measured trajectory start on the Turn-relative timeline")
    parser.add_argument("--run", default=None, help="existing/new run ID: run_YYYYMMDD_XXX")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), default=None)
    parser.add_argument("--turn-id", default=None)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="record an explicit checkpoint; otherwise use JSON metadata or null")
    parser.add_argument("--description", default="Machine-head raw trajectory experiment")
    parser.add_argument("--notes", default="", help="free-form run/segment note")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trajectory_json is None and args.measured_rpy is None:
        raise ValueError("provide a predicted trajectory JSON or --measured-rpy")

    predicted = None
    if args.trajectory_json is not None:
        predicted_path = args.trajectory_json.resolve()
        predicted_document, predicted_trajectory, predicted_states = load_trajectory(predicted_path)
        predicted = (predicted_path, predicted_document, predicted_trajectory, predicted_states)
    measured = None
    if args.measured_rpy is not None:
        measured_path = args.measured_rpy.resolve()
        measured_document, measured_trajectory, measured_states = load_trajectory(measured_path)
        measured = (measured_path, measured_document, measured_trajectory, measured_states)

    primary = predicted or measured
    assert primary is not None
    input_path, document, trajectory, states = primary
    summary, velocity, acceleration = analyze(document, trajectory, states)
    role = "measured" if predicted is None and args.role is None else infer_role(args.role, states)

    project_root = find_project_root()
    timestamp_value = datetime.now().astimezone()
    timestamp = timestamp_value.isoformat(timespec="seconds")
    runs_dir = project_root / "experiments" / EXPERIMENT_TYPE / "runs"
    run_dir, run_id, run_created = resolve_run(runs_dir, args.run, timestamp_value)
    segment_dir, segment_id, segment_index = create_segment(run_dir, role)

    net_commit = git_commit(project_root / "modules/algorithm")
    motor_commit = git_commit(project_root / "modules/motor")
    checkpoint = args.checkpoint or explicit_checkpoint(document)
    fps = float(document["fps"])
    run_config_path = run_dir / "run_config.json"
    if run_created or not run_config_path.exists():
        run_config = {
            "run_id": run_id,
            "start_time": timestamp,
            "experiment_type": EXPERIMENT_TYPE,
            "description": args.description,
            "project_net_git_commit": net_commit,
            "project_motor_git_commit": motor_commit,
            "checkpoint": checkpoint,
            "fps": fps,
            "segment_count": 0,
            "notes": args.notes,
        }
    else:
        with run_config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        if run_config.get("run_id") != run_id:
            raise ValueError(f"run_config.json run_id mismatch in {run_dir}")

    summary["run_id"] = run_id
    summary["segment_id"] = segment_id
    summary["segment_index"] = segment_index
    summary["role"] = role
    summary["timestamp"] = timestamp
    summary["max_delta_frame"] = summary["maximum_anomalies"]["frame_delta"]["frame_index"]
    summary["max_velocity_frame"] = summary["maximum_anomalies"]["velocity"]["frame_index"]
    summary["max_acceleration_frame"] = summary["maximum_anomalies"]["acceleration"]["frame_index"]

    shutil.copyfile(input_path, segment_dir / "raw_trajectory.json")
    write_raw_csv(segment_dir / "raw_rpy.csv", trajectory, states, fps)

    rpy_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    primary_start_sec = 0.0
    if measured is not None:
        _, measured_document, measured_trajectory, _ = measured
        measured_times, measured_start_sec = trajectory_times(
            measured_document, measured_trajectory, args.measured_start_sec
        )
        rpy_series["measured"] = (measured_times, measured_trajectory)
        if predicted is None:
            primary_start_sec = measured_start_sec
    if predicted is not None:
        _, predicted_document, predicted_trajectory, _ = predicted
        predicted_times, predicted_start_sec = trajectory_times(
            predicted_document, predicted_trajectory, args.predicted_start_sec
        )
        rpy_series["predicted"] = (predicted_times, predicted_trajectory)
        primary_start_sec = predicted_start_sec
    save_rpy_visualizations(segment_dir, rpy_series)

    save_plot(segment_dir / "rpy_velocity.png",
              primary_start_sec + np.arange(1, len(trajectory)) / fps,
              velocity, "angular velocity (rad/s)", "RPY Angular Velocity", states, fps,
              primary_start_sec)
    save_plot(segment_dir / "rpy_acceleration.png",
              primary_start_sec + np.arange(2, len(trajectory)) / fps,
              acceleration, "angular acceleration (rad/s²)", "RPY Angular Acceleration", states, fps,
              primary_start_sec)
    write_json_atomic(segment_dir / "summary.json", summary)

    segment_config = {
        "run_id": run_id,
        "segment_id": segment_id,
        "segment_index": segment_index,
        "role": role,
        "turn_id": args.turn_id,
        "stream_id": args.stream_id,
        "timestamp": timestamp,
        "fps": fps,
        "source": "algorithm",
        "checkpoint": checkpoint,
        "project_net_git_commit": net_commit,
        "project_motor_git_commit": motor_commit,
        "input_file": str(input_path),
        "command": shlex.join([sys.executable, *sys.argv]),
        "notes": args.notes,
    }
    write_json_atomic(segment_dir / "segment_config.json", segment_config)
    run_config["segment_count"] = segment_count(run_dir)
    write_json_atomic(run_config_path, run_config)

    print_report(summary, trajectory, velocity)
    print(f"\nRun ID: {run_id}")
    print(f"Segment ID: {segment_id}")
    print(f"Saved diagnostics to: {segment_dir}")


if __name__ == "__main__":
    main()
