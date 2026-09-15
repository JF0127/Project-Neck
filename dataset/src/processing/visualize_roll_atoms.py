"""Visualize threshold-confirmed Roll MOVE atoms and residual HOLD periods."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.processing.manual_annotations import ManualSegment, load_annotations


@dataclass(frozen=True)
class Move:
    start: int
    end: int
    direction: int


def confirmed_moves(values: np.ndarray, threshold: float) -> list[Move]:
    """Find directional moves whose ending extreme has a confirmed reversal."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Roll values must be non-empty and finite")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")

    moves: list[Move] = []
    direction = 0
    high = low = 0
    start = extreme = 0
    for index in range(1, len(values)):
        if direction == 0:
            if values[index] >= values[high]:
                high = index
            if values[index] <= values[low]:
                low = index
            if values[high] - values[low] >= threshold:
                if high > low:
                    direction, start, extreme = 1, low, high
                else:
                    direction, start, extreme = -1, high, low
        elif direction > 0:
            if values[index] >= values[extreme]:
                extreme = index
            elif values[extreme] - values[index] >= threshold:
                moves.append(Move(start, extreme, 1))
                direction, start, extreme = -1, extreme, index
        else:
            if values[index] <= values[extreme]:
                extreme = index
            elif values[index] - values[extreme] >= threshold:
                moves.append(Move(start, extreme, -1))
                direction, start, extreme = 1, extreme, index
    return moves


def _debounce(mask: np.ndarray, minimum_samples: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < len(result):
        end = index + 1
        while end < len(result) and result[end] == result[index]:
            end += 1
        if (
            not result[index]
            and index > 0
            and end < len(result)
            and end - index < minimum_samples
        ):
            result[index:end] = True
        index = end
    index = 0
    while index < len(result):
        end = index + 1
        while end < len(result) and result[end] == result[index]:
            end += 1
        if result[index] and end - index < minimum_samples:
            result[index:end] = False
        index = end
    return result


def actual_moves(
    times: np.ndarray,
    values: np.ndarray,
    coarse_moves: list[Move],
    min_speed: float,
    max_speed: float,
    minimum_duration: float,
    minimum_displacement: float,
) -> list[Move]:
    if not 0 < min_speed <= max_speed or minimum_duration <= 0 or minimum_displacement <= 0:
        raise ValueError("speed limits, duration, and displacement must be positive")
    dt = float(np.median(np.diff(times)))
    minimum_samples = max(2, int(np.ceil(minimum_duration / dt - 1e-9)) + 1)
    velocity = np.gradient(values, times)
    moves: list[Move] = []
    for coarse in coarse_moves:
        start, stop = coarse.start, coarse.end + 1
        speed = np.abs(velocity[start:stop])
        eligible = (speed >= min_speed) & (speed <= max_speed)
        eligible = _debounce(eligible, minimum_samples)
        index = 0
        while index < len(eligible):
            if not eligible[index]:
                index += 1
                continue
            end = index + 1
            while end < len(eligible) and eligible[end]:
                end += 1
            move_start, move_end = start + index, start + end - 1
            if abs(values[move_end] - values[move_start]) >= minimum_displacement:
                moves.append(Move(move_start, move_end, coarse.direction))
            index = end
    return moves


def _atoms(length: int, moves: list[Move]) -> list[tuple[str, int, int, int]]:
    atoms: list[tuple[str, int, int, int]] = []
    cursor = 0
    for move in moves:
        if move.start > cursor:
            atoms.append(("HOLD", cursor, move.start, 0))
        atoms.append(("MOVE", move.start, move.end, move.direction))
        cursor = move.end
    if cursor < length - 1:
        atoms.append(("HOLD", cursor, length - 1, 0))
    elif not atoms:
        atoms.append(("HOLD", 0, length - 1, 0))
    return atoms


def _load_roll(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        times = data["local_timestamps"].astype(np.float64)
        values = np.rad2deg(data["values"].astype(np.float64))
        valid = data["valid"].astype(bool)
    if values.shape != times.shape or valid.shape != times.shape or not valid.all():
        raise ValueError(f"Roll data must be frame-aligned and fully valid: {path}")
    return times, values


def _plot(
    output: Path,
    segment: ManualSegment,
    times: np.ndarray,
    roll: np.ndarray,
    threshold: float,
    min_speed: float,
    max_speed: float,
    minimum_duration: float,
) -> int:
    coarse_moves = confirmed_moves(roll, threshold)
    moves = actual_moves(
        times, roll, coarse_moves, min_speed, max_speed,
        minimum_duration, threshold,
    )
    atoms = _atoms(len(roll), moves)
    figure = plt.figure(figsize=(14, 7), dpi=140, layout="constrained")
    grid = figure.add_gridspec(3, 1, height_ratios=(0.8, 3.2, 1.1))

    text_axis = figure.add_subplot(grid[0])
    text_axis.axis("off")
    text_axis.set_title(
        f"{segment.id} | video {segment.start:.3f}s – {segment.end:.3f}s",
        loc="left",
    )
    text_axis.text(
        0.01, 0.8, textwrap.fill(segment.text, width=62),
        ha="left", va="top", fontsize=12, transform=text_axis.transAxes,
    )

    curve_axis = figure.add_subplot(grid[1])
    curve_axis.plot(times, roll, color="#d62728", linewidth=1.2, label="Roll")
    for number, move in enumerate(moves, 1):
        color = "#ff9896" if move.direction > 0 else "#6baed6"
        curve_axis.axvspan(times[move.start], times[move.end], color=color, alpha=0.16)
        curve_axis.scatter(times[move.start], roll[move.start], marker="o", s=42, color="#2ca02c", zorder=3)
        curve_axis.scatter(times[move.end], roll[move.end], marker="X", s=52, color="#111111", zorder=3)
        curve_axis.annotate(
            f"M{number} start", (times[move.start], roll[move.start]),
            xytext=(3, 8), textcoords="offset points", fontsize=8,
        )
        curve_axis.annotate(
            f"M{number} end", (times[move.end], roll[move.end]),
            xytext=(3, -14), textcoords="offset points", fontsize=8,
        )
    curve_axis.set_xlim(0, segment.duration)
    curve_axis.set_ylabel("Roll / degree")
    curve_axis.set_title(
        f"Roll MOVE start/end | {threshold:g}° confirmed | "
        f"{min_speed:g}–{max_speed:g}°/s for ≥{minimum_duration:g}s | Δ≥{threshold:g}°",
        loc="left",
    )
    curve_axis.grid(True, alpha=0.25)
    curve_axis.legend(loc="upper right")

    atom_axis = figure.add_subplot(grid[2], sharex=curve_axis)
    for kind, start, end, direction in atoms:
        left = times[start]
        width = max(times[end] - left, 1e-6)
        if kind == "HOLD":
            color, label = "#bdbdbd", "HOLD"
        else:
            color = "#ef3b2c" if direction > 0 else "#3182bd"
            label = "MOVE ↑" if direction > 0 else "MOVE ↓"
        atom_axis.broken_barh([(left, width)], (0.15, 0.7), facecolors=color, alpha=0.9)
        if width >= 0.35:
            atom_axis.text(left + width / 2, 0.5, label, ha="center", va="center", fontsize=8)
    atom_axis.set_xlim(0, segment.duration)
    atom_axis.set_ylim(0, 1)
    atom_axis.set_yticks([])
    atom_axis.set_xlabel("Slice-local time / s")
    atom_axis.set_title("HOLD/MOVE atomic sequence", loc="left")
    atom_axis.grid(True, axis="x", alpha=0.2)

    figure.savefig(output)
    plt.close(figure)
    return len(moves)


def visualize(
    annotations_path: Path,
    rpy_root: Path,
    output: Path,
    threshold: float,
    min_speed: float,
    max_speed: float,
    minimum_duration: float,
) -> list[int]:
    segments = load_annotations(annotations_path, validate_audio=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    counts: list[int] = []
    try:
        for segment in segments:
            times, roll = _load_roll(rpy_root / segment.id / "roll.npz")
            counts.append(
                _plot(
                    staging / f"{segment.id}.png", segment, times, roll,
                    threshold, min_speed, max_speed, minimum_duration,
                )
            )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize confirmed Roll HOLD/MOVE atoms")
    parser.add_argument("--annotations", required=True, type=Path, help="manual metadata.jsonl")
    parser.add_argument("--rpy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-degrees", type=float, default=5.0)
    parser.add_argument("--min-speed", type=float, default=3.0)
    parser.add_argument("--max-speed", type=float, default=10.0)
    parser.add_argument("--minimum-duration", type=float, default=0.12)
    args = parser.parse_args()

    counts = visualize(
        args.annotations.resolve(), args.rpy_root.resolve(),
        args.output.resolve(), args.threshold_degrees,
        args.min_speed, args.max_speed, args.minimum_duration,
    )
    print(f"Visualizations: {len(counts)}")
    print(f"Confirmed MOVEs: {sum(counts)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
