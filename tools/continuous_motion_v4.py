"""Reproducible pure-software V4 layer and V3 comparison experiment.

Run from repository root with an environment containing numpy and matplotlib:
    dataset/.venv/bin/python tools/continuous_motion_v4.py
No audio device, API or motor socket is opened.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runtime.contracts import MotionOutput
from runtime.inference.motion_plan import MotionPlan, MotionSegment
from runtime.inference.processor import MotionProcessor
from runtime.inference.prosody import ProsodyAnalysis, SegmentProsody
from runtime.inference.trajectory_generator import (
    TrajectoryGenerator, _compose_with_safety, _generate_base_flow,
    _generate_gesture_modulation, _POSTURAL_INTERVALS,
)
from runtime.inference.trajectory_optimizer import trajectory_metrics

OUT = Path("runtime/experiments/continuous_motion_v4")
FPS = 30
AXES = ("roll", "pitch", "yaw")


def prosody() -> ProsodyAnalysis:
    segments = []
    for start, end, energy, level, peak in (
        (0., 2., .8, "low", 1.),
        (2., 4., 1.7, "high", 2.9),
        (4., 6., .9, "medium", 5.),
    ):
        segments.append(SegmentProsody("合成语句", start, end, end - start, 0., 0.,
                                       .1, .2, energy, level, 5., "normal", peak))
    return ProsodyAnalysis(6., 16000, tuple(segments), 0.)


def save(name: str, document: object) -> None:
    (OUT / name).write_text(json.dumps(document, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n", encoding="utf-8")


def trajectory(name: str, values: np.ndarray, *, states=None) -> None:
    document = {"fps": FPS, "unit": "radian", "order": AXES,
                "trajectory": values.tolist()}
    if states is not None:
        document["states"] = list(states)
    save(name, document)


def plot(name: str, curves: list[tuple[str, np.ndarray]], title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for label, values in curves:
        degrees = np.degrees(values)
        t = np.arange(len(values)) / FPS
        for axis in range(3):
            axes[axis].plot(t, degrees[:, axis], label=label, linewidth=1.5)
    for axis, ax in enumerate(axes):
        ax.set_ylabel(f"{AXES[axis]} (deg)")
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def metrics(values: np.ndarray) -> dict:
    velocity = np.abs(np.diff(np.degrees(values), axis=0) * FPS)
    result = trajectory_metrics(values, FPS)
    for axis, name in enumerate(AXES):
        result[name]["mean_absolute_velocity_deg_s"] = float(np.mean(velocity[:, axis]))
        result[name]["median_absolute_velocity_deg_s"] = float(np.median(velocity[:, axis]))
        result[name]["soft_hold_fraction_v_lt_0.03_deg_s"] = float(np.mean(velocity[:, axis] < .03))
    result["overall"] = {
        "mean_absolute_velocity_deg_s": float(np.mean(velocity)),
        "median_absolute_velocity_deg_s": float(np.median(velocity)),
        "peak_velocity_deg_s": float(np.max(velocity)),
        "peak_acceleration_deg_s2": max(result[a]["peak_acceleration_deg_s2"] for a in AXES),
        "peak_jerk_deg_s3": max(result[a]["peak_jerk_deg_s3"] for a in AXES),
        "soft_hold_fraction_all_axes_v_lt_0.03_deg_s": float(np.mean(np.all(velocity < .03, axis=1))),
        "exact_zero_frame_ratio": float(np.mean(np.all(np.abs(values) < 1e-12, axis=1))),
        "one_axis_active_fraction_v_gt_0.1_deg_s": float(np.mean(np.sum(velocity > .1, axis=1) == 1)),
        "all_axes_active_fraction_v_gt_0.1_deg_s": float(np.mean(np.all(velocity > .1, axis=1))),
    }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generator = TrajectoryGenerator()
    p = prosody()
    plan = MotionPlan("speaking", 6., (
        MotionSegment(2.4, 3.2, "nod", "pitch", 2., "emphasis"),
        MotionSegment(4.5, 5.3, "turn", "yaw", 2., "attention"),
    ))
    layers = generator.generate_layers(plan, p)
    post, accent, semantic, raw = (np.asarray(getattr(layers, field)) for field in
                                   ("postural_flow", "prosodic_accent", "semantic_gesture", "composed_raw"))
    both = np.asarray(generator.generate_layers(plan, p, include_semantic=False).composed_raw)
    final = MotionProcessor().process(MotionOutput(layers.composed_raw), (0., 0., 0.))
    optimized = np.asarray(final.rpy)
    save("motion_plan.json", plan.to_dict())
    save("prosody.json", p.to_dict())
    for name, values in (("postural_flow", post), ("prosodic_accent", accent),
                         ("semantic_gesture", semantic), ("raw_v4_trajectory", raw)):
        trajectory(name + ".json", values)
    trajectory("optimized_v4_trajectory.json", optimized, states=final.states)
    plot("01_postural_flow.png", [("postural", post)], "Postural flow (MOVE / HOLD)")
    plot("02_prosodic_accent.png", [("pitch energy accent", accent)], "Prosodic accent")
    plot("03_semantic_gesture.png", [("semantic", semantic)], "Semantic gestures")
    plot("04_layer_composition.png", [("postural", post), ("postural + prosodic", both),
                                       ("full", raw)], "V4 layers")
    plot("05_raw_v4_rpy.png", [("V4 raw", raw)], "V4 raw relative RPY")
    plot("06_optimized_v4_rpy.png", [("V4 optimized + neutral", optimized)],
         "MotionProcessor + unchanged optimizer")

    # Exactly the same V3 implementation and MotionPlan, without modifying its parameters.
    comparison_plan = MotionPlan("speaking", 5., (
        MotionSegment(.8, 1.4, "nod", "pitch", 2., "emphasis"),
        MotionSegment(2.2, 3., "turn", "yaw", 2.5, "attention"),
        MotionSegment(3.6, 4.3, "tilt", "roll", -1.8, "question"),
    ))
    n = round(comparison_plan.duration_sec * FPS)
    v3_carrier = _generate_base_flow(comparison_plan.duration_sec, n, generator.config)
    gesture = _generate_gesture_modulation(comparison_plan, n, generator.config)
    v3 = np.asarray(_compose_with_safety(v3_carrier, gesture, generator.config.max_composed_offset_deg))
    v4_layers = generator.generate_layers(comparison_plan)  # no fabricated prosody
    v4 = np.asarray(v4_layers.composed_raw)
    plot("07_v3_v4_comparison.png", [("V3", v3), ("V4", v4)], "Same plan: V3 vs V4 (no prosody)")
    save("comparison_motion_plan.json", comparison_plan.to_dict())
    results = {
        "units": "degree, second; derivatives: absolute per-axis frame differences at 30 fps",
        "synthetic_6s": {
            "postural_only": metrics(post), "postural_plus_prosodic": metrics(both),
            "full_raw": metrics(raw), "optimized_with_neutral_return": metrics(optimized),
            "semantic_visibility_peak_deg": float(np.max(np.abs(np.degrees(semantic)))),
        },
        "comparison_5s_same_plan_without_prosody": {
            "v3_carrier": metrics(np.asarray(v3_carrier)),
            "v4_postural": metrics(np.asarray(v4_layers.postural_flow)),
            "v3_full_raw": metrics(v3), "v4_full_raw": metrics(v4),
            "v3_optimized_with_neutral_return": metrics(np.asarray(MotionProcessor().process(MotionOutput(tuple(map(tuple, v3))), (0., 0., 0.)).rpy)),
            "v4_optimized_with_neutral_return": metrics(np.asarray(MotionProcessor().process(MotionOutput(v4_layers.composed_raw), (0., 0., 0.)).rpy)),
            "postural_target_intervals_sec": {axis: list(_POSTURAL_INTERVALS[i]) for i, axis in enumerate(AXES)},
            "semantic_visibility_peak_deg": float(np.max(np.abs(np.degrees(gesture)))),
        },
    }
    save("metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("Plots:", *(str(path) for path in sorted(OUT.glob("*.png"))), sep="\n")


if __name__ == "__main__":
    main()
