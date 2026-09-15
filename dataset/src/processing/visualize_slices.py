"""Visualize manual sentence audio and frame-aligned RPY curves."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import textwrap
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.processing.manual_annotations import ManualSegment, load_annotations

AXES = ("roll", "pitch", "yaw")
COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def _audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, rate


def _rpy(path: Path, duration: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        times = data["local_timestamps"].astype(np.float64)
        values = data["values"].astype(np.float32)
        valid = data["valid"].astype(bool)
    if values.shape != (len(times), 3) or valid.shape != times.shape:
        raise ValueError(f"invalid frame alignment: {path}")
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError(f"invalid RPY timestamps: {path}")
    values[~valid] = np.nan
    if times[0] < -1e-3 or times[-1] > duration + 0.05:
        raise ValueError(f"RPY timeline is outside annotation bounds: {path}")
    return times, np.rad2deg(values)


def _hold_move_axis(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("HOLD/MOVE input must be non-empty and finite")
    output = np.empty_like(values)
    target = values[0]
    move_indices = [0]
    output[0] = target
    for index in range(1, len(values)):
        if abs(values[index] - target) >= threshold:
            target = values[index]
            move_indices.append(index)
        output[index] = target
    return output, np.asarray(move_indices, dtype=np.int64)


def _motor_trajectory(values: np.ndarray, threshold: float) -> tuple[np.ndarray, list[np.ndarray]]:
    motor = np.empty_like(values)
    move_indices: list[np.ndarray] = []
    for column in range(values.shape[1]):
        motor[:, column], moves = _hold_move_axis(values[:, column], threshold)
        move_indices.append(moves)
    return motor, move_indices


def _plot(
    output: Path,
    segment: ManualSegment,
    audio: np.ndarray,
    sample_rate: int,
    rpy_times: np.ndarray,
    rpy_degrees: np.ndarray,
    hysteresis_degrees: float,
) -> None:
    motor_degrees, motor_keys = _motor_trajectory(rpy_degrees, hysteresis_degrees)
    figure = plt.figure(figsize=(14, 10), dpi=140, layout="constrained")
    grid = figure.add_gridspec(4, 1, height_ratios=(1.0, 1.3, 1.8, 1.8))

    text_axis = figure.add_subplot(grid[0])
    text_axis.axis("off")
    text_axis.text(
        0.01, 0.92, textwrap.fill(segment.text, width=58),
        ha="left", va="top", fontsize=13, linespacing=1.5,
        transform=text_axis.transAxes,
    )
    text_axis.set_title(
        f"{segment.id} | video {segment.start:.3f}s – {segment.end:.3f}s "
        f"| duration {segment.duration:.3f}s",
        loc="left", fontsize=12,
    )

    waveform_axis = figure.add_subplot(grid[1])
    step = max(1, len(audio) // 50_000)
    waveform_times = np.arange(len(audio), dtype=np.float64) / sample_rate
    waveform_axis.plot(waveform_times[::step], audio[::step], color="#444444", linewidth=0.5)
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.set_title("Manually segmented audio", loc="left")
    waveform_axis.set_xlim(0, segment.duration)
    waveform_axis.set_ylim(-1.0, 1.0)
    waveform_axis.grid(True, alpha=0.2)

    rpy_axis = figure.add_subplot(grid[2], sharex=waveform_axis)
    for column, (axis, color) in enumerate(zip(AXES, COLORS)):
        rpy_axis.plot(rpy_times, rpy_degrees[:, column], label=axis, color=color, linewidth=1.0)
    rpy_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    rpy_axis.set_ylabel("Angle / degree")
    rpy_axis.set_title("Raw RPY relative to source-video frame 0", loc="left")
    rpy_axis.set_xlim(0, segment.duration)
    rpy_axis.grid(True, alpha=0.25)
    rpy_axis.legend(loc="upper right", ncol=3)

    motor_axis = figure.add_subplot(grid[3], sharex=waveform_axis)
    for column, (axis, color) in enumerate(zip(AXES, COLORS)):
        motor_axis.plot(
            rpy_times, motor_degrees[:, column], label=axis,
            color=color, linewidth=1.4, drawstyle="steps-post",
        )
        keys = motor_keys[column]
        motor_axis.scatter(rpy_times[keys], motor_degrees[keys, column], color=color, s=16, zorder=3)
    motor_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    motor_axis.set_xlabel("Segment-local time / s")
    motor_axis.set_ylabel("Angle / degree")
    motor_axis.set_title(
        f"Causal motor HOLD/MOVE targets ({hysteresis_degrees:g}° threshold)", loc="left"
    )
    motor_axis.set_xlim(0, segment.duration)
    motor_axis.grid(True, alpha=0.25)
    motor_axis.legend(loc="upper right", ncol=3)

    figure.savefig(output)
    plt.close(figure)


def visualize(
    annotations_path: Path,
    rpy_root: Path,
    output: Path,
    hysteresis_degrees: float,
) -> int:
    segments = load_annotations(annotations_path, validate_audio=True)
    if not np.isfinite(hysteresis_degrees) or hysteresis_degrees <= 0:
        raise ValueError("hysteresis_degrees must be positive and finite")

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for segment in segments:
            audio, rate = _audio(annotations_path.parent / segment.file)
            times, values = _rpy(rpy_root / segment.id / "rpy.npz", segment.duration)
            _plot(
                staging / f"{segment.id}.png", segment, audio, rate,
                times, values, hysteresis_degrees,
            )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize manually annotated segments")
    parser.add_argument("--annotations", required=True, type=Path, help="manual metadata.jsonl")
    parser.add_argument("--rpy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hysteresis-degrees", type=float, default=5.0)
    args = parser.parse_args()

    count = visualize(
        args.annotations.resolve(), args.rpy_root.resolve(),
        args.output.resolve(), args.hysteresis_degrees,
    )
    print(f"Visualizations: {count}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
