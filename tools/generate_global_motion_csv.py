#!/usr/bin/env python3
"""Generate three standalone periodic Global Motion CSVs (degrees, 30 fps)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

FPS = 30
KEYFRAMES = {
    "global_01": [
        (0.0, 0.0, 0.0, 0.0),
        (2.5, 0.5, -0.2, 2.0),
        (5.0, 1.0, 0.3, 4.0),
        (8.0, 0.2, 0.8, 0.8),
        (11.0, -0.8, 0.2, -3.2),
        (13.5, -0.4, -0.4, -1.4),
        (16.0, 0.0, 0.0, 0.0),
    ],
    "global_02": [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, -0.3, 0.5, 1.0),
        (4.5, -0.8, 1.0, 2.6),
        (7.0, 0.1, -0.2, 0.5),
        (9.5, 0.8, -0.8, -2.5),
        (12.0, 0.4, 0.2, -1.2),
        (14.0, 0.0, 0.0, 0.0),
    ],
    "global_03": [
        (0.0, 0.0, 0.0, 0.0),
        (3.0, 0.6, 0.2, -1.5),
        (6.0, 1.2, -0.4, -3.8),
        (9.0, 0.4, 0.7, -1.0),
        (12.0, -0.6, 0.9, 2.2),
        (15.0, -1.0, -0.3, 3.6),
        (18.0, 0.0, 0.0, 0.0),
    ],
}


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, keyframes in KEYFRAMES.items():
        values = np.asarray(keyframes, dtype=float)
        knots = values[:, 0]
        duration = knots[-1]
        if not np.all(np.diff(knots) > 0) or not np.array_equal(values[0, 1:], values[-1, 1:]):
            raise ValueError(f"{name}: invalid periodic keyframes")
        splines = [CubicSpline(knots, values[:, axis], bc_type="periodic") for axis in (1, 2, 3)]
        # Integer sample indices avoid floating drift and exclude the duplicate endpoint.
        count = int(duration * FPS)
        timestamps = np.arange(count) / FPS
        frames = np.column_stack((timestamps, *(spline(timestamps) for spline in splines)))
        position_error = max(abs(float(s(0) - s(duration))) for s in splines)
        velocity_error = max(abs(float(s(0, 1) - s(duration, 1))) for s in splines)
        if position_error > 1e-10 or velocity_error > 1e-10:
            raise RuntimeError(f"{name}: periodic seam is discontinuous")
        output = output_dir / f"{name}.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time", "roll", "pitch", "yaw"))
            writer.writerows(frames)
        print(f"{output}: {count} frames, fps={FPS}, duration={duration:g}s, "
              f"seam_position_error={position_error:.3g} deg, "
              f"seam_velocity_error={velocity_error:.3g} deg/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    generate(parser.parse_args().output_dir)
