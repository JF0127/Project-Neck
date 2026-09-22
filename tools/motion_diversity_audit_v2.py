#!/usr/bin/env python3
"""Collect MotionPlan V2 diversity samples and compare them with the V1 audit."""
from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.motion_diversity_audit as v1  # noqa: E402
from runtime.__main__ import load_config  # noqa: E402
from runtime.contracts import MotionOutput  # noqa: E402
from runtime.doubao_tts import DoubaoTTS  # noqa: E402
from runtime.inference.artifacts import write_wav_atomic  # noqa: E402
from runtime.inference.deepseek_motion import (  # noqa: E402
    DeepSeekMotionBackend,
    MOTION_SYSTEM_PROMPT,
)
from runtime.inference.motion_plan import MotionPlan  # noqa: E402
from runtime.inference.motion_plan_validator import validate_motion_plan  # noqa: E402
from runtime.inference.processor import MotionProcessor  # noqa: E402
from runtime.inference.prosody import extract_prosody  # noqa: E402
from runtime.inference.speech_alignment import align_speech  # noqa: E402
from runtime.inference.trajectory_generator import TrajectoryGenerator  # noqa: E402
from runtime.inference.trajectory_optimizer import AXES, trajectory_metrics  # noqa: E402

FPS = 30.0
RUNS_PER_TEXT = 3
NORMALIZED_SAMPLES = 120
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime/experiments/motion_diversity_audit_v2"
V1_OUTPUT = PROJECT_ROOT / "runtime/experiments/motion_diversity_audit"
DEFAULT_CONFIG = PROJECT_ROOT / "runtime/config.yaml"
SENTENCES = v1.SENTENCES
MINIMAL_PAIRS = v1.MINIMAL_PAIRS
PLOT_SELECTION = v1.PLOT_SELECTION
ACTION_TYPES = ("nod", "turn", "tilt", "shake")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


async def collect(output: Path, config_path: Path, *, resume: bool) -> None:
    config, resolved_config = load_config(config_path)
    dialogue = _section(config, "dialogue")
    motion = _section(config, "motion")
    model = str(motion.get("deepseek_model", dialogue.get("model", "deepseek-flash")))
    base_url = str(
        motion.get("deepseek_base_url", dialogue.get("base_url", "https://api.deepseek.com"))
    )
    timeout_sec = float(motion.get("deepseek_timeout_sec", 15.0))
    temperature = float(motion.get("deepseek_temperature", 0.2))
    max_output_tokens = int(motion.get("deepseek_max_output_tokens", 1024))
    if output.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume or --reset")
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    _write_json(
        output / "request_config.json",
        {
            "planner_version": "motion_plan_v2",
            "runtime_config_path": str(resolved_config),
            "model": model,
            "temperature": temperature,
            "top_p": "not explicitly configured",
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "none"},
            "timeout_sec": timeout_sec,
            "system_prompt": MOTION_SYSTEM_PROMPT,
        },
    )
    _write_json(
        output / "sentences.json",
        [{"id": identifier, "text": text} for identifier, text in SENTENCES],
    )

    tts = DoubaoTTS()
    backend = DeepSeekMotionBackend(
        model=model,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        output_dir=output / "_unused_backend_output",
    )
    generator = TrajectoryGenerator()
    processor = MotionProcessor()
    tts_calls = planner_calls = planner_success = 0
    failures: list[dict[str, str]] = []

    for sentence_id, text in SENTENCES:
        sample_dir = output / "samples" / sentence_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
        audio_path = sample_dir / "audio.wav"
        alignment_path = sample_dir / "alignment.json"
        prosody_path = sample_dir / "prosody.json"
        try:
            if resume and audio_path.is_file() and alignment_path.is_file() and prosody_path.is_file():
                import wave
                with wave.open(str(audio_path), "rb") as handle:
                    pcm = handle.readframes(handle.getnframes())
                duration_sec = len(pcm) / 2 / 16_000
                payload_segments = _read_json(prosody_path)["segments"]
            else:
                tts_calls += 1
                speech = await tts.synthesize(text)
                pcm = speech.pcm_s16le
                duration_sec = float(speech.duration_sec)
                write_wav_atomic(audio_path, pcm)
                alignment = align_speech(text, pcm, duration_sec, sample_rate=16_000)
                _write_json(
                    alignment_path,
                    {
                        "text": text,
                        "audio_duration_sec": duration_sec,
                        "segments": alignment.segments_as_dicts(),
                        "alignment": alignment.metadata(),
                    },
                )
                prosody = extract_prosody(pcm, 16_000, alignment)
                payload_segments = prosody.payload_segments()
                _write_json(
                    prosody_path,
                    {
                        "robot_text": text,
                        "alignment": alignment.metadata(),
                        **prosody.to_dict(),
                    },
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _write_json(sample_dir / "tts_or_prosody_error.json", {"error": error})
            failures.append({"id": sentence_id, "stage": "tts_prosody", "error": error})
            continue

        for run_number in range(1, RUNS_PER_TEXT + 1):
            prefix = f"run_{run_number:02d}"
            paths = {
                "raw": sample_dir / f"{prefix}_deepseek_motion_plan_raw.json",
                "plan": sample_dir / f"{prefix}_motion_plan.json",
                "relative": sample_dir / f"{prefix}_raw_relative_trajectory.json",
                "final": sample_dir / f"{prefix}_final_trajectory.json",
            }
            if resume and all(path.is_file() for path in paths.values()):
                planner_success += 1
                continue
            planner_calls += 1
            started = time.perf_counter()
            raw_response: str | None = None
            payload = backend._payload(text, duration_sec, payload_segments)
            try:
                raw_response = backend._request_plan(text, duration_sec, payload_segments)
                raw_document = json.loads(raw_response)
                _write_json(
                    paths["raw"],
                    {
                        "request_payload": payload,
                        "raw_response": raw_response,
                        "parsed_json": raw_document,
                        "latency_sec": time.perf_counter() - started,
                    },
                )
                plan = MotionPlan.from_dict(raw_document)
                validate_motion_plan(plan, expected_duration_sec=duration_sec)
                _write_json(paths["plan"], plan.to_dict())
                relative = generator.generate(plan)
                relative_frames = [list(frame) for frame in relative.rpy_offset]
                _write_json(
                    paths["relative"],
                    {
                        "fps": relative.fps,
                        "unit": relative.unit,
                        "order": list(relative.order),
                        "representation": relative.representation,
                        "trajectory": relative_frames,
                        "metrics": trajectory_metrics(relative_frames, relative.fps),
                    },
                )
                final = processor.process(relative, (0.0, 0.0, 0.0))
                final_frames = [list(frame) for frame in final.rpy]
                _write_json(
                    paths["final"],
                    {
                        "fps": final.fps,
                        "unit": final.unit,
                        "order": list(final.order),
                        "representation": "absolute_rpy",
                        "start_pose": [0.0, 0.0, 0.0],
                        "duration_sec": final.duration_sec,
                        "states": list(final.states),
                        "trajectory": final_frames,
                        "metrics": trajectory_metrics(final_frames, final.fps),
                    },
                )
                planner_success += 1
                print(f"[audit-v2] {sentence_id} run {run_number}: ok")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _write_json(
                    paths["raw"],
                    {
                        "request_payload": payload,
                        "raw_response": raw_response,
                        "error": error,
                        "latency_sec": time.perf_counter() - started,
                    },
                )
                _write_json(sample_dir / f"{prefix}_error.json", {"error": error})
                failures.append(
                    {"id": sentence_id, "stage": f"planner_{run_number}", "error": error}
                )
                print(f"[audit-v2] {sentence_id} run {run_number}: {error}")

    manifest = {
        "complete": planner_success == len(SENTENCES) * RUNS_PER_TEXT,
        "sentence_count": len(SENTENCES),
        "runs_per_sentence": RUNS_PER_TEXT,
        "expected_planner_samples": len(SENTENCES) * RUNS_PER_TEXT,
        "tts_calls_this_invocation": tts_calls,
        "planner_calls_this_invocation": planner_calls,
        "successful_planner_samples": planner_success,
        "failures": failures,
        "finished_unix_sec": time.time(),
    }
    _write_json(output / "collection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class V2Run:
    sentence_id: str
    text: str
    run: int
    duration_sec: float
    plan: MotionPlan
    raw: np.ndarray
    final: np.ndarray
    raw_metrics: dict[str, dict[str, float]]
    final_metrics: dict[str, dict[str, float]]

    @property
    def key(self) -> str:
        return f"{self.sentence_id}_run_{self.run:02d}"


def _load_runs(output: Path) -> tuple[list[V2Run], list[str]]:
    runs: list[V2Run] = []
    failures: list[str] = []
    for sentence_id, text in SENTENCES:
        sample_dir = output / "samples" / sentence_id
        alignment_path = sample_dir / "alignment.json"
        if not alignment_path.is_file():
            failures.append(f"{sentence_id}: missing alignment")
            continue
        duration = float(_read_json(alignment_path)["audio_duration_sec"])
        for number in range(1, RUNS_PER_TEXT + 1):
            prefix = f"run_{number:02d}"
            plan_path = sample_dir / f"{prefix}_motion_plan.json"
            raw_path = sample_dir / f"{prefix}_raw_relative_trajectory.json"
            final_path = sample_dir / f"{prefix}_final_trajectory.json"
            if not all(path.is_file() for path in (plan_path, raw_path, final_path)):
                failures.append(f"{sentence_id}/{prefix}: incomplete")
                continue
            plan = MotionPlan.from_dict(_read_json(plan_path))
            raw = _read_json(raw_path)
            final = _read_json(final_path)
            runs.append(
                V2Run(
                    sentence_id,
                    text,
                    number,
                    duration,
                    plan,
                    np.asarray(raw["trajectory"], dtype=np.float64),
                    np.asarray(final["trajectory"], dtype=np.float64),
                    raw["metrics"],
                    final["metrics"],
                )
            )
    return runs, failures


def _timing_bin(start: float, end: float, duration: float) -> str:
    midpoint = (start + end) / (2.0 * duration)
    return "early" if midpoint < 1 / 3 else "mid" if midpoint < 2 / 3 else "late"


def _amp_bin(value: float) -> str:
    return f"{round(value / 0.5) * 0.5:+g}"


def _signature(run: V2Run) -> str:
    if not run.plan.segments:
        return "0 | HOLD"
    actions = "-".join(segment.action for segment in run.plan.segments)
    axes = "-".join(segment.primary_axis for segment in run.plan.segments)
    amplitudes = ",".join(_amp_bin(segment.amplitude_deg) for segment in run.plan.segments)
    timing = "-".join(
        _timing_bin(segment.start_sec, segment.end_sec, run.duration_sec)
        for segment in run.plan.segments
    )
    return f"{len(run.plan.segments)} | {actions} | {axes} | {amplitudes} | {timing}"


def _normalize(values: np.ndarray, final: bool) -> np.ndarray:
    trajectory = values - values[0] if final else values
    old_time = np.linspace(0.0, 1.0, len(trajectory))
    new_time = np.linspace(0.0, 1.0, NORMALIZED_SAMPLES)
    return np.stack(
        [np.interp(new_time, old_time, trajectory[:, axis]) for axis in range(3)],
        axis=1,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1)
    b = right.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _similarities(
    runs: Sequence[V2Run], final: bool
) -> tuple[np.ndarray, dict[str, np.ndarray], list[float]]:
    ids = [identifier for identifier, _ in SENTENCES]
    normalized = {run.key: _normalize(run.final if final else run.raw, final) for run in runs}
    by_id = {identifier: [run for run in runs if run.sentence_id == identifier] for identifier in ids}
    matrix = np.full((len(ids), len(ids)), np.nan)
    cross: list[float] = []
    for i, left_id in enumerate(ids):
        for j, right_id in enumerate(ids):
            values = [
                _cosine(normalized[left.key], normalized[right.key])
                for left in by_id[left_id]
                for right in by_id[right_id]
            ]
            if values:
                matrix[i, j] = float(np.mean(values))
        for right_id in ids[i + 1 :]:
            cross.extend(
                _cosine(normalized[left.key], normalized[right.key])
                for left in by_id[left_id]
                for right in by_id[right_id]
            )
    return matrix, normalized, cross


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    ids = [identifier for identifier, _ in SENTENCES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence_id", *ids])
        for identifier, row in zip(ids, matrix):
            writer.writerow([identifier, *[f"{value:.9f}" for value in row]])


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {name: math.nan for name in ("mean", "std", "min", "median", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _bucket_counts(counts: Sequence[int]) -> dict[str, int]:
    return {
        "empty": sum(value == 0 for value in counts),
        "one": sum(value == 1 for value in counts),
        "two": sum(value == 2 for value in counts),
        "three_plus": sum(value >= 3 for value in counts),
    }


def _write_motion_metrics(path: Path, runs: Sequence[V2Run]) -> None:
    fields = [
        "sentence_id",
        "run",
        "layer",
        "axis",
        "peak_velocity_deg_s",
        "peak_acceleration_deg_s2",
        "peak_jerk_deg_s3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for layer, metrics in (
                ("raw", run.raw_metrics),
                ("optimized", run.final_metrics),
            ):
                for axis in AXES:
                    writer.writerow(
                        {
                            "sentence_id": run.sentence_id,
                            "run": run.run,
                            "layer": layer,
                            "axis": axis,
                            **metrics[axis],
                        }
                    )


def _metrics_summary(runs: Sequence[Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer, attribute in (("raw", "raw_metrics"), ("optimized", "final_metrics")):
        output[layer] = {}
        for axis in AXES:
            output[layer][axis] = {
                metric: _summary(
                    [float(getattr(run, attribute)[axis][metric]) for run in runs]
                )
                for metric in (
                    "peak_velocity_deg_s",
                    "peak_acceleration_deg_s2",
                    "peak_jerk_deg_s3",
                )
            }
    return output


def _plot(
    output: Path,
    runs: Sequence[V2Run],
    raw_matrix: np.ndarray,
    final_matrix: np.ndarray,
    v1_axis: Counter[str],
    v2_axis: Counter[str],
    v1_action_total: int,
    v2_actions: Counter[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    ids = [identifier for identifier, _ in SENTENCES]
    by_id = {identifier: [run for run in runs if run.sentence_id == identifier] for identifier in ids}

    means = [np.mean([len(run.plan.segments) for run in by_id[item]]) for item in ids]
    stds = [np.std([len(run.plan.segments) for run in by_id[item]]) for item in ids]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ids, means, yerr=stds, capsize=3)
    ax.set_title("V2 action count by sentence")
    ax.set_ylabel("Actions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(plots / "01_action_count.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar((*ACTION_TYPES, "HOLD"), [v2_actions[x] for x in ACTION_TYPES] + [sum(not r.plan.segments for r in runs)])
    ax.set_title("V2 action type distribution")
    fig.tight_layout(); fig.savefig(plots / "02_action_type_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar((*AXES, "none"), [v2_axis[x] for x in (*AXES, "none")])
    ax.set_title("V2 primary axis distribution")
    fig.tight_layout(); fig.savefig(plots / "03_primary_axis_distribution.png", dpi=160); plt.close(fig)

    def heatmap(matrix: np.ndarray, name: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
        ax.set_yticks(range(len(ids)), ids)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Mean cosine similarity")
        fig.tight_layout(); fig.savefig(plots / name, dpi=160); plt.close(fig)
    heatmap(raw_matrix, "04_raw_similarity_heatmap.png", "V2 raw RPY similarity")
    heatmap(final_matrix, "05_optimized_similarity_heatmap.png", "V2 optimized RPY similarity")

    def selected(final: bool, name: str, title: str) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for ax, identifier in zip(axes, PLOT_SELECTION):
            run = by_id[identifier][0]
            values = run.final - run.final[0] if final else run.raw
            times = np.arange(len(values)) / FPS
            for axis_index, axis_name in enumerate(AXES):
                ax.plot(times, np.degrees(values[:, axis_index]), label=axis_name)
            ax.set_title(identifier); ax.set_ylabel("degree"); ax.grid(alpha=0.25); ax.legend()
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(plots / name, dpi=160); plt.close(fig)
    selected(False, "06_selected_raw_rpy.png", "V2 selected raw trajectories")
    selected(True, "07_selected_optimized_rpy.png", "V2 selected optimized trajectories")

    x = np.arange(3); width = 0.36
    v1_total = sum(v1_axis.values()) or 1; v2_total = sum(v2_axis.values()) or 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [100 * v1_axis[a] / v1_total for a in AXES], width, label="V1")
    ax.bar(x + width / 2, [100 * v2_axis[a] / v2_total for a in AXES], width, label="V2")
    ax.set_xticks(x, AXES); ax.set_ylabel("Percent"); ax.set_title("V1 vs V2 primary axis"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "08_v1_v2_primary_axis.png", dpi=160); plt.close(fig)

    labels = ("V1 untyped", *ACTION_TYPES)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, [v1_action_total, *[v2_actions[a] for a in ACTION_TYPES]])
    ax.set_title("V1 untyped sparse actions vs V2 action vocabulary")
    fig.tight_layout(); fig.savefig(plots / "09_v1_v2_action_type.png", dpi=160); plt.close(fig)


def analyze(output: Path) -> dict[str, Any]:
    runs, failures = _load_runs(output)
    manifest_path = output / "collection_manifest.json"
    collection_failures = (
        _read_json(manifest_path).get("failures", [])
        if manifest_path.is_file()
        else []
    )
    if not runs:
        raise RuntimeError("no complete V2 samples")
    v1_runs, v1_failures = v1._load_runs(V1_OUTPUT)
    if not v1_runs:
        raise RuntimeError("V1 audit is unavailable")

    _write_motion_metrics(output / "motion_metrics.csv", runs)
    signatures = Counter(_signature(run) for run in runs)
    v1_signatures = Counter(v1._signature(run) for run in v1_runs)
    counts = [len(run.plan.segments) for run in runs]
    v1_counts = [len(run.actions) for run in v1_runs]
    buckets = _bucket_counts(counts)
    v1_buckets = _bucket_counts(v1_counts)
    action_counts = Counter(segment.action for run in runs for segment in run.plan.segments)
    axis_counts = Counter(segment.primary_axis for run in runs for segment in run.plan.segments)
    v1_axis_counts = Counter(
        v1._primary_axis(action)[0] for run in v1_runs for action in run.actions
    )
    reasons = Counter(segment.reason for run in runs for segment in run.plan.segments)
    amplitudes = [abs(segment.amplitude_deg) for run in runs for segment in run.plan.segments]
    v1_amplitudes = [
        v1._primary_axis(action)[1]
        for run in v1_runs
        for action in run.actions
    ]
    starts = [segment.start_sec / run.duration_sec for run in runs for segment in run.plan.segments]
    ends = [segment.end_sec / run.duration_sec for run in runs for segment in run.plan.segments]
    durations = [
        (segment.end_sec - segment.start_sec) / run.duration_sec
        for run in runs for segment in run.plan.segments
    ]
    timing_bins = Counter(
        _timing_bin(segment.start_sec, segment.end_sec, run.duration_sec)
        for run in runs for segment in run.plan.segments
    )
    v1_starts = [
        float(action["start"]) / run.duration_sec
        for run in v1_runs
        for action in run.actions
    ]
    v1_ends = [
        float(action["end"]) / run.duration_sec
        for run in v1_runs
        for action in run.actions
    ]
    v1_durations = [
        (float(action["end"]) - float(action["start"])) / run.duration_sec
        for run in v1_runs
        for action in run.actions
    ]

    raw_matrix, raw_normalized, raw_cross = _similarities(runs, False)
    final_matrix, final_normalized, final_cross = _similarities(runs, True)
    _write_matrix(output / "trajectory_similarity_raw.csv", raw_matrix)
    _write_matrix(output / "trajectory_similarity_optimized.csv", final_matrix)
    raw_stats = _summary(raw_cross)
    final_stats = _summary(final_cross)
    v1_raw_matrix, v1_raw_normalized, v1_raw_pairs = v1._similarity_data(
        v1_runs, final=False
    )
    v1_final_matrix, v1_final_normalized, v1_final_pairs = v1._similarity_data(
        v1_runs, final=True
    )
    v1_raw_mean = float(np.mean([value for value, _, _ in v1_raw_pairs]))
    v1_final_mean = float(np.mean([value for value, _, _ in v1_final_pairs]))

    by_id = {identifier: [run for run in runs if run.sentence_id == identifier] for identifier, _ in SENTENCES}
    v1_by_id = {
        identifier: [run for run in v1_runs if run.sentence_id == identifier]
        for identifier, _ in SENTENCES
    }
    minimal_rows = []
    for left_id, right_id in MINIMAL_PAIRS:
        left = by_id[left_id]; right = by_id[right_id]
        raw_values = [_cosine(raw_normalized[a.key], raw_normalized[b.key]) for a in left for b in right]
        final_values = [_cosine(final_normalized[a.key], final_normalized[b.key]) for a in left for b in right]
        v1_left = v1_by_id[left_id]
        v1_right = v1_by_id[right_id]
        v1_raw_values = [
            v1._cosine(v1_raw_normalized[a.key], v1_raw_normalized[b.key])
            for a in v1_left
            for b in v1_right
        ]
        v1_final_values = [
            v1._cosine(v1_final_normalized[a.key], v1_final_normalized[b.key])
            for a in v1_left
            for b in v1_right
        ]
        minimal_rows.append(
            {
                "pair": f"{left_id} vs {right_id}",
                "left_plans": [_signature(run) for run in left],
                "right_plans": [_signature(run) for run in right],
                "v1_raw_similarity": float(np.mean(v1_raw_values)),
                "v2_raw_similarity": float(np.mean(raw_values)),
                "v1_optimized_similarity": float(np.mean(v1_final_values)),
                "v2_optimized_similarity": float(np.mean(final_values)),
            }
        )
    _write_json(output / "minimal_pairs.json", minimal_rows)

    within = []
    for identifier, _ in SENTENCES:
        values = [_signature(run) for run in by_id[identifier]]
        within.append({"sentence_id": identifier, "signatures": values, "unique": len(set(values))})
    _write_json(output / "within_text_consistency.json", within)

    with (output / "plan_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sentence_id", "run", "text", "action_count", "signature", "actions", "axes", "amplitudes_deg", "reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "sentence_id": run.sentence_id,
                    "run": run.run,
                    "text": run.text,
                    "action_count": len(run.plan.segments),
                    "signature": _signature(run),
                    "actions": ";".join(s.action for s in run.plan.segments),
                    "axes": ";".join(s.primary_axis for s in run.plan.segments),
                    "amplitudes_deg": ";".join(str(s.amplitude_deg) for s in run.plan.segments),
                    "reasons": ";".join(s.reason for s in run.plan.segments),
                }
            )

    _plot(output, runs, raw_matrix, final_matrix, v1_axis_counts, axis_counts, sum(v1_axis_counts.values()), action_counts)

    metrics_summary = {
        "v1": _metrics_summary(v1_runs),
        "v2": _metrics_summary(runs),
    }
    _write_json(output / "trajectory_metrics_summary.json", metrics_summary)
    v1_axis_total = sum(v1_axis_counts.values())
    v2_axis_total = sum(axis_counts.values())
    comparison = {
        "v1": {
            "axis_percent": {axis: 100 * v1_axis_counts[axis] / v1_axis_total for axis in AXES},
            "plan_buckets": v1_buckets,
            "unique_signatures": len(v1_signatures),
            "most_common_signature": v1_signatures.most_common(1)[0],
            "raw_mean_similarity": v1_raw_mean,
            "optimized_mean_similarity": v1_final_mean,
            "action_counts": {"untyped_sparse_action": sum(v1_counts)},
            "primary_amplitude": _summary(v1_amplitudes),
            "normalized_start": _summary(v1_starts),
            "normalized_end": _summary(v1_ends),
            "normalized_duration": _summary(v1_durations),
        },
        "v2": {
            "axis_percent": {axis: 100 * axis_counts[axis] / v2_axis_total if v2_axis_total else 0.0 for axis in AXES},
            "plan_buckets": buckets,
            "unique_signatures": len(signatures),
            "most_common_signature": signatures.most_common(1)[0],
            "raw_mean_similarity": raw_stats["mean"],
            "optimized_mean_similarity": final_stats["mean"],
            "action_counts": dict(action_counts),
            "reason_counts": dict(reasons),
            "primary_amplitude": _summary(amplitudes),
            "normalized_start": _summary(starts),
            "normalized_end": _summary(ends),
            "normalized_duration": _summary(durations),
            "timing_bins": dict(timing_bins),
        },
        "failures": failures,
        "collection_failures": collection_failures,
        "trajectory_metrics": metrics_summary,
        "v1_load_failures": v1_failures,
    }
    _write_json(output / "v1_v2_comparison.json", comparison)

    def percent(value: int, total: int) -> float:
        return 100.0 * value / total if total else 0.0
    report = [
        "# Motion Diversity Audit V2",
        "",
        "## Completeness",
        f"- V2 successful samples: {len(runs)} / 48",
        f"- V2 failures: {len(failures)}",
        f"- V1 preserved samples: {len(v1_runs)} / 48",
        *[
            f"- Failed sample {item['id']} {item['stage']}: `{item['error']}`"
            for item in collection_failures
        ],
        "",
        "## Production V2 chain",
        "```text",
        "Text + TTS PCM → Speech Alignment → Prosody Extractor V1 → DeepSeek MotionPlan V2",
        "→ Plan Validator V2 → Action Primitive Generator V2 → raw relative RPY",
        "→ MotionProcessor → unchanged TrajectoryOptimizer → Final RPY",
        "```",
        "",
        "## V1 vs V2",
        "| Metric | V1 | V2 |",
        "|---|---:|---:|",
        *[
            f"| {axis} primary | {100*v1_axis_counts[axis]/v1_axis_total:.2f}% | {100*axis_counts[axis]/v2_axis_total if v2_axis_total else 0:.2f}% |"
            for axis in AXES
        ],
        f"| empty/HOLD plans | {percent(v1_buckets['empty'], len(v1_runs)):.2f}% | {percent(buckets['empty'], len(runs)):.2f}% |",
        f"| 1-action plans | {percent(v1_buckets['one'], len(v1_runs)):.2f}% | {percent(buckets['one'], len(runs)):.2f}% |",
        f"| 2-action plans | {percent(v1_buckets['two'], len(v1_runs)):.2f}% | {percent(buckets['two'], len(runs)):.2f}% |",
        f"| 3+-action plans | {percent(v1_buckets['three_plus'], len(v1_runs)):.2f}% | {percent(buckets['three_plus'], len(runs)):.2f}% |",
        f"| unique signatures | {len(v1_signatures)}/{len(v1_runs)} successful plans | {len(signatures)}/{len(runs)} successful plans |",
        f"| most common signature frequency | {v1_signatures.most_common(1)[0][1]} ({100*v1_signatures.most_common(1)[0][1]/len(v1_runs):.2f}%) | {signatures.most_common(1)[0][1]} ({100*signatures.most_common(1)[0][1]/len(runs):.2f}%) |",
        f"| primary amplitude mean | {_summary(v1_amplitudes)['mean']:.3f}° | {_summary(amplitudes)['mean']:.3f}° |",
        f"| normalized start mean | {_summary(v1_starts)['mean']:.3f} | {_summary(starts)['mean']:.3f} |",
        f"| normalized end mean | {_summary(v1_ends)['mean']:.3f} | {_summary(ends)['mean']:.3f} |",
        f"| normalized duration mean | {_summary(v1_durations)['mean']:.3f} | {_summary(durations)['mean']:.3f} |",
        f"| raw mean similarity | {v1_raw_mean:.6f} | {raw_stats['mean']:.6f} |",
        f"| optimized mean similarity | {v1_final_mean:.6f} | {final_stats['mean']:.6f} |",
        "",
        "V1 most common signature:",
        f"`{v1_signatures.most_common(1)[0][0]}`",
        "",
        "V2 most common signature:",
        f"`{signatures.most_common(1)[0][0]}`",
        "",
        "## V2 action diversity",
        *[f"- {action}: {action_counts[action]}" for action in ACTION_TYPES],
        f"- empty/HOLD plans: {buckets['empty']}",
        "",
        "## V2 reason vocabulary",
        *[f"- `{reason}`: {count}" for reason, count in reasons.most_common()],
        "",
        "## V2 plan distributions",
        f"- action count buckets: {buckets}",
        f"- primary amplitude: {_summary(amplitudes)}",
        f"- normalized start: {_summary(starts)}",
        f"- normalized end: {_summary(ends)}",
        f"- normalized duration: {_summary(durations)}",
        f"- timing bins: {dict(timing_bins)}",
        "",
        "## Trajectory metrics",
        "Per-run and per-axis raw/optimized peak velocity, acceleration and jerk are in `motion_metrics.csv`; V1/V2 aggregate mean/std/min/median/max values are in `trajectory_metrics_summary.json`.",
        *[
            f"- {version} {layer} {axis}: velocity mean/max="
            f"{metrics_summary[version][layer][axis]['peak_velocity_deg_s']['mean']:.3f}/"
            f"{metrics_summary[version][layer][axis]['peak_velocity_deg_s']['max']:.3f} deg/s; "
            f"acceleration mean/max="
            f"{metrics_summary[version][layer][axis]['peak_acceleration_deg_s2']['mean']:.3f}/"
            f"{metrics_summary[version][layer][axis]['peak_acceleration_deg_s2']['max']:.3f} deg/s²"
            for version in ("v1", "v2")
            for layer in ("raw", "optimized")
            for axis in AXES
        ],
        "",
        "## V2 signatures",
        *[
            f"- `{signature}`: {count} ({100*count/len(runs):.2f}% of successful plans), "
            + ", ".join(sorted({run.sentence_id for run in runs if _signature(run) == signature}))
            for signature, count in signatures.most_common()
        ],
        "",
        "## Minimal pairs",
        *[
            f"### {row['pair']}\n- V2 left: {row['left_plans']}\n- V2 right: {row['right_plans']}\n- raw similarity V1/V2: {row['v1_raw_similarity']:.6f} / {row['v2_raw_similarity']:.6f}\n- optimized similarity V1/V2: {row['v1_optimized_similarity']:.6f} / {row['v2_optimized_similarity']:.6f}"
            for row in minimal_rows
        ],
        "",
        "## Same-text stability",
        *[f"- {row['sentence_id']}: unique={row['unique']}; {row['signatures']}" for row in within],
        "",
        "## Notes",
        "- Similarity uses the same 120-sample linear normalization and cosine method as V1.",
        "- Lower similarity is not treated as automatically better.",
        "- V1 actions have no action vocabulary; the comparison plot labels them as untyped.",
        "- F0 was not extracted because no reliable installed F0 dependency was used.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("all", "collect", "analyze"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if args.reset and output.exists():
        if output == PROJECT_ROOT or PROJECT_ROOT not in output.parents:
            raise SystemExit("refusing to reset output outside project")
        shutil.rmtree(output)
    try:
        if args.phase in {"all", "collect"}:
            asyncio.run(collect(output, args.config.expanduser().resolve(), resume=args.resume))
        if args.phase in {"all", "analyze"}:
            comparison = analyze(output)
            print(json.dumps(comparison, ensure_ascii=False, indent=2))
            print("Report:", output / "report.md")
            print("Plots:", output / "plots")
    except Exception as exc:
        raise SystemExit(f"motion diversity audit V2 failed: {type(exc).__name__}: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
