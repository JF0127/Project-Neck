#!/usr/bin/env python3
"""Read-only diagnostics for Algorithm -> Neck Interface V1 trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
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
TOOL_VERSION = "3.0.0"
TURN_PATTERN = re.compile(r"^turn_(\d+)$")


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
    project_root = Path(__file__).resolve().parents[1]
    if not (project_root / "modules/algorithm").is_dir():
        raise RuntimeError("cannot locate Project-Neck root from trajectory_visualizer.py")
    return project_root


def resolve_session(value: str, project_root: Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend([
            project_root / requested,
            project_root / "experiments/v0_trajectory/sessions" / requested,
        ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"session directory does not exist: {value}")


def discover_turns(session_dir: Path) -> list[Path]:
    turns_dir = session_dir / "turns"
    if not turns_dir.is_dir():
        raise ValueError(f"session has no turns directory: {turns_dir}")
    turns = []
    for child in turns_dir.iterdir():
        match = TURN_PATTERN.fullmatch(child.name) if child.is_dir() else None
        if match:
            turns.append((int(match.group(1)), child))
    return [path for _, path in sorted(turns)]


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


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
    parser = argparse.ArgumentParser(
        description="Visualize every available turn in a Project-Neck experiment session",
        epilog=(
            "examples:\n"
            "  python3 tools/trajectory_visualizer.py session_20260902_003\n"
            "  python3 tools/trajectory_visualizer.py "
            "experiments/v0_trajectory/sessions/session_20260902_003"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session",
        help="session ID or path to a session directory",
    )
    return parser.parse_args()


def process_turn(turn_dir: Path, output_dir: Path) -> None:
    predicted_path = turn_dir / "neck_rpy.json"
    measured_path = turn_dir / "measured_rpy.json"

    predicted = None
    if predicted_path.is_file():
        try:
            document, trajectory, states = load_trajectory(predicted_path)
            predicted = (predicted_path, document, trajectory, states)
        except Exception as exc:
            print(
                f"[visualizer][warning] {turn_dir.name}: ignore neck_rpy.json: "
                f"{type(exc).__name__}: {exc}"
            )

    measured = None
    if measured_path.is_file():
        try:
            document, trajectory, states = load_trajectory(measured_path)
            measured = (measured_path, document, trajectory, states)
        except Exception as exc:
            print(
                f"[visualizer][warning] {turn_dir.name}: ignore measured_rpy.json: "
                f"{type(exc).__name__}: {exc}"
            )

    primary = predicted or measured
    if primary is None:
        raise ValueError("neither neck_rpy.json nor measured_rpy.json exists")

    input_path, document, trajectory, states = primary
    summary, velocity, acceleration = analyze(document, trajectory, states)
    summary["turn_id"] = turn_dir.name
    summary["tool_version"] = TOOL_VERSION
    summary["input_file"] = str(input_path)
    summary["measured_file"] = str(measured_path) if measured is not None else None
    summary["checkpoint"] = explicit_checkpoint(document)
    summary["max_delta_frame"] = summary["maximum_anomalies"]["frame_delta"]["frame_index"]
    summary["max_velocity_frame"] = summary["maximum_anomalies"]["velocity"]["frame_index"]
    summary["max_acceleration_frame"] = summary["maximum_anomalies"]["acceleration"]["frame_index"]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copyfile(input_path, output_dir / "raw_trajectory.json")
    fps = float(document["fps"])
    write_raw_csv(output_dir / "raw_rpy.csv", trajectory, states, fps)

    rpy_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    primary_start_sec = 0.0
    if measured is not None:
        _, measured_document, measured_trajectory, _ = measured
        measured_times, measured_start_sec = trajectory_times(
            measured_document, measured_trajectory, None
        )
        rpy_series["measured"] = (measured_times, measured_trajectory)
        if predicted is None:
            primary_start_sec = measured_start_sec
    if predicted is not None:
        _, predicted_document, predicted_trajectory, _ = predicted
        predicted_times, predicted_start_sec = trajectory_times(
            predicted_document, predicted_trajectory, None
        )
        rpy_series["predicted"] = (predicted_times, predicted_trajectory)
        primary_start_sec = predicted_start_sec
    save_rpy_visualizations(output_dir, rpy_series)

    save_plot(
        output_dir / "rpy_velocity.png",
        primary_start_sec + np.arange(1, len(trajectory)) / fps,
        velocity,
        "angular velocity (rad/s)",
        "RPY Angular Velocity",
        states,
        fps,
        primary_start_sec,
    )
    save_plot(
        output_dir / "rpy_acceleration.png",
        primary_start_sec + np.arange(2, len(trajectory)) / fps,
        acceleration,
        "angular acceleration (rad/s²)",
        "RPY Angular Acceleration",
        states,
        fps,
        primary_start_sec,
    )
    write_json_atomic(output_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    session_dir = resolve_session(args.session, project_root)
    turns = discover_turns(session_dir)
    if not turns:
        raise ValueError(f"session has no valid turn directories: {session_dir / 'turns'}")

    visualizations_dir = session_dir / "visualizations"
    completed = 0
    skipped = 0
    print(f"[visualizer] session: {session_dir}")
    print(f"[visualizer] discovered {len(turns)} turn(s)")
    for turn_dir in turns:
        output_dir = visualizations_dir / turn_dir.name
        try:
            process_turn(turn_dir, output_dir)
        except Exception as exc:
            skipped += 1
            if output_dir.exists():
                shutil.rmtree(output_dir)
            print(f"[visualizer][warning] skip {turn_dir.name}: {type(exc).__name__}: {exc}")
            continue
        completed += 1
        print(f"[visualizer] {turn_dir.name}: saved to {output_dir}")

    print(f"[visualizer] complete: {completed} visualized, {skipped} skipped")
    if completed == 0:
        raise RuntimeError("session contains no visualizable turns")


if __name__ == "__main__":
    main()
