#!/usr/bin/env python3
"""Run and analyze the real DeepSeek Motion path without changing production code."""
from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import wave
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.__main__ import load_config  # noqa: E402
from runtime.contracts import MotionOutput  # noqa: E402
from runtime.doubao_tts import (  # noqa: E402
    DOUBAO_RESOURCE_ID,
    DoubaoTTS,
)
from runtime.inference.artifacts import write_wav_atomic  # noqa: E402
from runtime.inference.deepseek_motion import (  # noqa: E402
    DeepSeekMotionBackend,
    MOTION_SYSTEM_PROMPT,
)
from runtime.inference.motion_compiler import (  # noqa: E402
    MotionPlanError,
    SparseMotionAction,
    compile_motion_plan,
    parse_motion_plan_with_rejections,
)
from runtime.inference.processor import MotionProcessor  # noqa: E402
from runtime.inference.speech_alignment import (  # noqa: E402
    SpeechAlignment,
    align_speech,
)
from runtime.inference.trajectory_optimizer import (  # noqa: E402
    AXES,
    trajectory_metrics,
)

FPS = 30.0
RUNS_PER_TEXT = 3
NORMALIZED_SAMPLES = 120
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime/experiments/motion_diversity_audit"
DEFAULT_CONFIG = PROJECT_ROOT / "runtime/config.yaml"
PRIMARY_AXIS_EPSILON_DEG = 1e-9
AMPLITUDE_BIN_DEG = 0.5

SENTENCES: tuple[tuple[str, str], ...] = (
    ("A1", "这个方案可行。"),
    ("A2", "这个方案可行吗？"),
    ("B1", "我同意这个方案。"),
    ("B2", "我不同意这个方案。"),
    ("C1", "这个结果已经确定了。"),
    ("C2", "这个结果可能还不确定。"),
    ("D1", "这个结果需要记录。"),
    ("D2", "这个结果非常重要，一定要记住。"),
    ("E1", "好的，我们继续下一步。"),
    ("E2", "太好了，我们马上继续下一步！"),
    ("F1", "我认为这里还有问题。"),
    ("F2", "嗯……我想了一下，这里可能还有问题。"),
    ("G1", "什么？这个结果居然已经出来了？"),
    ("G2", "前半部分没有问题，但是后半部分需要重新设计。"),
    ("G3", "第一，我们检查输入；第二，我们检查输出；最后再看整体效果。"),
    ("G4", "没关系，我们慢慢来，这个问题可以解决。"),
)
MINIMAL_PAIRS = (("A1", "A2"), ("B1", "B2"), ("C1", "C2"),
                 ("D1", "D2"), ("E1", "E2"), ("F1", "F2"))
PLOT_SELECTION = ("A2", "B2", "E2")


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


def _read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16_000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getcomptype() != "NONE"
        ):
            raise ValueError(f"audit WAV has an unexpected format: {path}")
        return handle.readframes(handle.getnframes())


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _request_record(
    model: str,
    base_url: str,
    timeout_sec: float,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "base_url": base_url,
        "timeout_sec": timeout_sec,
        "temperature": temperature,
        "top_p": "not explicitly configured",
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "none"},
        "system_prompt": MOTION_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            MOTION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "user_prompt_construction": (
            "compact JSON object with robot_text, duration_sec, and segments "
            "when speech alignment succeeds"
        ),
    }


async def collect(
    output: Path,
    config_path: Path,
    *,
    resume: bool,
) -> None:
    config, resolved_config = load_config(config_path)
    dialogue = _config_section(config, "dialogue")
    motion = _config_section(config, "motion")
    model = str(motion.get("deepseek_model", dialogue.get("model", "deepseek-flash")))
    base_url = str(
        motion.get(
            "deepseek_base_url",
            dialogue.get("base_url", "https://api.deepseek.com"),
        )
    )
    timeout_sec = float(motion.get("deepseek_timeout_sec", 15.0))
    temperature = float(motion.get("deepseek_temperature", 0.2))
    max_output_tokens = int(motion.get("deepseek_max_output_tokens", 1024))

    if output.exists() and not resume:
        raise FileExistsError(
            f"audit output already exists: {output}; use --resume or --reset"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)

    request_config = _request_record(
        model, base_url, timeout_sec, temperature, max_output_tokens
    )
    request_config["runtime_config_path"] = str(resolved_config)
    _write_json(output / "request_config.json", request_config)
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
    processor = MotionProcessor()
    planner_calls = 0
    tts_calls = 0
    tts_success = 0
    planner_success = 0
    failures: list[dict[str, str]] = []

    for sentence_id, text in SENTENCES:
        sample_dir = output / "samples" / sentence_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
        audio_path = sample_dir / "audio.wav"
        alignment_path = sample_dir / "alignment.json"
        try:
            if resume and audio_path.is_file() and alignment_path.is_file():
                pcm = _read_wav(audio_path)
                duration_sec = len(pcm) / 2 / 16_000
                alignment_document = _read_json(alignment_path)
                segments = alignment_document["segments"]
            else:
                tts_calls += 1
                speech = await tts.synthesize(text)
                pcm = speech.pcm_s16le
                duration_sec = float(speech.duration_sec)
                write_wav_atomic(audio_path, pcm)
                alignment = align_speech(text, pcm, duration_sec, sample_rate=16_000)
                segments = alignment.segments_as_dicts()
                alignment_document = {
                    "text": text,
                    "audio_duration_sec": duration_sec,
                    "sample_rate": 16_000,
                    "segments": segments,
                    "alignment": alignment.metadata(),
                }
                _write_json(alignment_path, alignment_document)
            tts_success += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _write_json(sample_dir / "tts_error.json", {"error": error})
            failures.append({"id": sentence_id, "stage": "tts_alignment", "error": error})
            continue

        for run_number in range(1, RUNS_PER_TEXT + 1):
            prefix = f"run_{run_number:02d}"
            raw_plan_path = sample_dir / f"{prefix}_raw_plan.json"
            actions_path = sample_dir / f"{prefix}_actions.json"
            raw_rpy_path = sample_dir / f"{prefix}_raw_rpy.json"
            final_rpy_path = sample_dir / f"{prefix}_final_rpy.json"
            if (
                resume
                and raw_plan_path.is_file()
                and actions_path.is_file()
                and raw_rpy_path.is_file()
                and final_rpy_path.is_file()
            ):
                planner_success += 1
                continue

            planner_calls += 1
            started = time.perf_counter()
            raw_response: str | None = None
            try:
                raw_response = backend._request_plan(text, duration_sec, segments)
                latency = time.perf_counter() - started
                raw_document = json.loads(raw_response)
                _write_json(
                    raw_plan_path,
                    {
                        "sentence_id": sentence_id,
                        "run": run_number,
                        "raw_response": raw_response,
                        "parsed_json": raw_document,
                        "latency_sec": latency,
                        "user_payload": {
                            "robot_text": text,
                            "duration_sec": duration_sec,
                            "segments": segments,
                        },
                    },
                )
                accepted, rejected = parse_motion_plan_with_rejections(
                    raw_document, duration_sec
                )
                _write_json(
                    actions_path,
                    {
                        "accepted_actions": [action.as_dict() for action in accepted],
                        "rejected_actions": [action.as_dict() for action in rejected],
                    },
                )
                if not accepted:
                    raise MotionPlanError("plan contains no accepted actions")
                raw_output = compile_motion_plan(accepted, duration_sec)
                raw_frames = [list(frame) for frame in raw_output.rpy_offset]
                _write_json(
                    raw_rpy_path,
                    {
                        "fps": raw_output.fps,
                        "unit": raw_output.unit,
                        "order": list(raw_output.order),
                        "representation": raw_output.representation,
                        "trajectory": raw_frames,
                        "metrics": trajectory_metrics(raw_frames, raw_output.fps),
                    },
                )
                final = processor.process(raw_output, (0.0, 0.0, 0.0))
                final_frames = [list(frame) for frame in final.rpy]
                _write_json(
                    final_rpy_path,
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
                print(f"[audit] {sentence_id} run {run_number}: ok")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if not raw_plan_path.exists():
                    _write_json(
                        raw_plan_path,
                        {
                            "sentence_id": sentence_id,
                            "run": run_number,
                            "raw_response": raw_response,
                            "error": error,
                            "latency_sec": time.perf_counter() - started,
                        },
                    )
                _write_json(sample_dir / f"{prefix}_error.json", {"error": error})
                failures.append(
                    {"id": sentence_id, "stage": f"planner_{run_number}", "error": error}
                )
                print(f"[audit] {sentence_id} run {run_number}: {error}")

    available_raw = sum(
        1
        for sentence_id, _ in SENTENCES
        for run_number in range(1, RUNS_PER_TEXT + 1)
        if (output / "samples" / sentence_id / f"run_{run_number:02d}_raw_plan.json").is_file()
    )
    manifest = {
        "complete": planner_success == len(SENTENCES) * RUNS_PER_TEXT,
        "sentence_count": len(SENTENCES),
        "runs_per_sentence": RUNS_PER_TEXT,
        "expected_planner_samples": len(SENTENCES) * RUNS_PER_TEXT,
        "tts_calls_this_invocation": tts_calls,
        "tts_successful_sentences": tts_success,
        "planner_calls_this_invocation": planner_calls,
        "successful_planner_samples": planner_success,
        "available_raw_response_files": available_raw,
        "failures": failures,
        "finished_unix_sec": time.time(),
    }
    _write_json(output / "collection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class AuditRun:
    sentence_id: str
    text: str
    run: int
    duration_sec: float
    actions: tuple[dict[str, float], ...]
    rejected_actions: tuple[dict[str, Any], ...]
    raw: np.ndarray
    final: np.ndarray
    raw_metrics: dict[str, dict[str, float]]
    final_metrics: dict[str, dict[str, float]]

    @property
    def key(self) -> str:
        return f"{self.sentence_id}_run_{self.run:02d}"


def _load_runs(output: Path) -> tuple[list[AuditRun], list[dict[str, str]]]:
    text_by_id = dict(SENTENCES)
    runs: list[AuditRun] = []
    failures: list[dict[str, str]] = []
    for sentence_id, text in SENTENCES:
        sample_dir = output / "samples" / sentence_id
        alignment_path = sample_dir / "alignment.json"
        if not alignment_path.is_file():
            failures.append({"id": sentence_id, "error": "missing alignment.json"})
            continue
        duration = float(_read_json(alignment_path)["audio_duration_sec"])
        for run_number in range(1, RUNS_PER_TEXT + 1):
            prefix = f"run_{run_number:02d}"
            paths = {
                "actions": sample_dir / f"{prefix}_actions.json",
                "raw": sample_dir / f"{prefix}_raw_rpy.json",
                "final": sample_dir / f"{prefix}_final_rpy.json",
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                failures.append(
                    {"id": f"{sentence_id}/{prefix}", "error": f"missing {missing}"}
                )
                continue
            action_doc = _read_json(paths["actions"])
            raw_doc = _read_json(paths["raw"])
            final_doc = _read_json(paths["final"])
            runs.append(
                AuditRun(
                    sentence_id=sentence_id,
                    text=text_by_id[sentence_id],
                    run=run_number,
                    duration_sec=duration,
                    actions=tuple(action_doc["accepted_actions"]),
                    rejected_actions=tuple(action_doc["rejected_actions"]),
                    raw=np.asarray(raw_doc["trajectory"], dtype=np.float64),
                    final=np.asarray(final_doc["trajectory"], dtype=np.float64),
                    raw_metrics=raw_doc["metrics"],
                    final_metrics=final_doc["metrics"],
                )
            )
    return runs, failures


def _primary_axis(action: dict[str, float]) -> tuple[str, float]:
    amplitudes = [abs(float(action[axis])) for axis in AXES]
    peak = max(amplitudes)
    if peak <= PRIMARY_AXIS_EPSILON_DEG:
        return "none", 0.0
    return AXES[int(np.argmax(amplitudes))], peak


def _signed_primary_amplitude(action: dict[str, float]) -> float:
    axis, _ = _primary_axis(action)
    return 0.0 if axis == "none" else float(action[axis])


def _primary_direction(action: dict[str, float]) -> str:
    axis, _ = _primary_axis(action)
    if axis == "none":
        return "none"
    return axis + ("+" if float(action[axis]) >= 0.0 else "-")


def _timing_label(action: dict[str, float], duration: float) -> str:
    midpoint = (float(action["start"]) + float(action["end"])) / (2.0 * duration)
    if midpoint < 1.0 / 3.0:
        return "early"
    if midpoint < 2.0 / 3.0:
        return "mid"
    return "late"


def _amplitude_bin(value: float, *, signed: bool = False) -> str:
    rounded = round(value / AMPLITUDE_BIN_DEG) * AMPLITUDE_BIN_DEG
    return f"{rounded:+g}" if signed else f"{rounded:g}"


def _signature(run: AuditRun) -> str:
    axes = [_primary_axis(action)[0] for action in run.actions]
    amplitudes = [
        _amplitude_bin(_signed_primary_amplitude(action), signed=True)
        for action in run.actions
    ]
    timing = [_timing_label(action, run.duration_sec) for action in run.actions]
    vectors = [
        "(" + "/".join(
            _amplitude_bin(float(action[axis]), signed=True) for axis in AXES
        ) + ")"
        for action in run.actions
    ]
    return (
        f"{len(run.actions)} | {'-'.join(axes) or 'none'} | "
        f"{','.join(amplitudes) or 'none'} | {'-'.join(timing) or 'none'} | "
        f"{';'.join(vectors) or 'none'}"
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {name: math.nan for name in ("mean", "std", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _normalize_trajectory(values: np.ndarray, *, relative_to_first: bool) -> np.ndarray:
    trajectory = np.asarray(values, dtype=np.float64)
    if relative_to_first:
        trajectory = trajectory - trajectory[0]
    old_time = np.linspace(0.0, 1.0, len(trajectory))
    new_time = np.linspace(0.0, 1.0, NORMALIZED_SAMPLES)
    return np.stack(
        [np.interp(new_time, old_time, trajectory[:, axis]) for axis in range(3)],
        axis=1,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1)
    b = right.reshape(-1)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 and b_norm == 0.0:
        return 1.0
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _similarity_data(
    runs: Sequence[AuditRun],
    *,
    final: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[tuple[float, AuditRun, AuditRun]]]:
    normalized = {
        run.key: _normalize_trajectory(
            run.final if final else run.raw,
            relative_to_first=final,
        )
        for run in runs
    }
    sentence_ids = [identifier for identifier, _ in SENTENCES]
    matrix = np.full((len(sentence_ids), len(sentence_ids)), np.nan, dtype=np.float64)
    by_sentence = {
        sentence_id: [run for run in runs if run.sentence_id == sentence_id]
        for sentence_id in sentence_ids
    }
    cross_pairs: list[tuple[float, AuditRun, AuditRun]] = []
    for left_index, left_id in enumerate(sentence_ids):
        for right_index, right_id in enumerate(sentence_ids):
            values = [
                _cosine(normalized[left.key], normalized[right.key])
                for left in by_sentence[left_id]
                for right in by_sentence[right_id]
            ]
            if values:
                matrix[left_index, right_index] = float(np.mean(values))
        for right_id in sentence_ids[left_index + 1 :]:
            for left in by_sentence[left_id]:
                for right in by_sentence[right_id]:
                    cross_pairs.append(
                        (_cosine(normalized[left.key], normalized[right.key]), left, right)
                    )
    return matrix, normalized, cross_pairs


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    identifiers = [identifier for identifier, _ in SENTENCES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence_id", *identifiers])
        for identifier, row in zip(identifiers, matrix):
            writer.writerow(
                [identifier, *["" if not math.isfinite(value) else f"{value:.9f}" for value in row]]
            )


def _pair_average(
    left_id: str,
    right_id: str,
    normalized: dict[str, np.ndarray],
    runs: Sequence[AuditRun],
) -> float:
    left_runs = [run for run in runs if run.sentence_id == left_id]
    right_runs = [run for run in runs if run.sentence_id == right_id]
    values = [
        _cosine(normalized[left.key], normalized[right.key])
        for left in left_runs
        for right in right_runs
    ]
    return float(np.mean(values)) if values else math.nan


def _write_plan_summary(path: Path, runs: Sequence[AuditRun]) -> None:
    fields = [
        "sentence_id", "run", "text", "audio_duration_sec", "action_count",
        "axis_sequence", "primary_direction_sequence", "primary_amplitudes_deg",
        "signed_primary_amplitudes_deg", "normalized_starts",
        "normalized_ends", "normalized_durations", "rejected_count", "signature",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "sentence_id": run.sentence_id,
                    "run": run.run,
                    "text": run.text,
                    "audio_duration_sec": f"{run.duration_sec:.6f}",
                    "action_count": len(run.actions),
                    "axis_sequence": "-".join(_primary_axis(action)[0] for action in run.actions),
                    "primary_direction_sequence": "-".join(
                        _primary_direction(action) for action in run.actions
                    ),
                    "primary_amplitudes_deg": ";".join(
                        f"{_primary_axis(action)[1]:.6f}" for action in run.actions
                    ),
                    "signed_primary_amplitudes_deg": ";".join(
                        f"{_signed_primary_amplitude(action):.6f}"
                        for action in run.actions
                    ),
                    "normalized_starts": ";".join(
                        f"{float(action['start']) / run.duration_sec:.6f}"
                        for action in run.actions
                    ),
                    "normalized_ends": ";".join(
                        f"{float(action['end']) / run.duration_sec:.6f}"
                        for action in run.actions
                    ),
                    "normalized_durations": ";".join(
                        f"{(float(action['end']) - float(action['start'])) / run.duration_sec:.6f}"
                        for action in run.actions
                    ),
                    "rejected_count": len(run.rejected_actions),
                    "signature": _signature(run),
                }
            )


def _write_motion_metrics(path: Path, runs: Sequence[AuditRun]) -> None:
    fields = [
        "sentence_id", "run", "layer", "axis", "peak_velocity_deg_s",
        "peak_acceleration_deg_s2", "peak_jerk_deg_s3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for layer, metrics in (("raw", run.raw_metrics), ("optimized", run.final_metrics)):
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


def _plot_results(
    output: Path,
    runs: Sequence[AuditRun],
    raw_matrix: np.ndarray,
    final_matrix: np.ndarray,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for audit plots") from exc

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    identifiers = [identifier for identifier, _ in SENTENCES]
    by_sentence = {
        identifier: [run for run in runs if run.sentence_id == identifier]
        for identifier in identifiers
    }

    means = [np.mean([len(run.actions) for run in by_sentence[item]]) for item in identifiers]
    stds = [np.std([len(run.actions) for run in by_sentence[item]]) for item in identifiers]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(identifiers, means, yerr=stds, capsize=3)
    axis.set_ylabel("Accepted action count")
    axis.set_xlabel("Sentence ID")
    axis.set_title("Plan action count by sentence")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "01_action_count_by_sentence.png", dpi=160)
    plt.close(figure)

    axis_counts = Counter(
        _primary_axis(action)[0] for run in runs for action in run.actions
    )
    axis_names = ("roll", "pitch", "yaw", "none")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(axis_names, [axis_counts[name] for name in axis_names])
    axis.set_ylabel("Accepted actions")
    axis.set_title("Primary axis distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "02_primary_axis_distribution.png", dpi=160)
    plt.close(figure)

    def heatmap(matrix: np.ndarray, filename: str, title: str) -> None:
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(len(identifiers)), identifiers, rotation=45, ha="right")
        axis.set_yticks(range(len(identifiers)), identifiers)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Mean cosine similarity")
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=160)
        plt.close(figure)

    heatmap(raw_matrix, "03_raw_similarity_heatmap.png", "Raw relative RPY similarity")
    heatmap(
        final_matrix,
        "04_optimized_similarity_heatmap.png",
        "Final optimized RPY similarity",
    )

    def trajectory_plot(final: bool, filename: str, title: str) -> None:
        figure, axes = plt.subplots(len(PLOT_SELECTION), 1, figsize=(12, 9), sharex=False)
        for plot_axis, sentence_id in zip(axes, PLOT_SELECTION):
            run = by_sentence[sentence_id][0]
            trajectory = run.final if final else run.raw
            if final:
                trajectory = trajectory - trajectory[0]
            times = np.arange(len(trajectory)) / FPS
            values = np.degrees(trajectory)
            for axis_index, axis_name in enumerate(AXES):
                plot_axis.plot(times, values[:, axis_index], label=axis_name)
            plot_axis.set_title(sentence_id)
            plot_axis.set_ylabel("degree")
            plot_axis.grid(alpha=0.25)
            plot_axis.legend(loc="best")
        axes[-1].set_xlabel("time (s)")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=160)
        plt.close(figure)

    trajectory_plot(False, "05_selected_raw_rpy.png", "Selected raw relative trajectories")
    trajectory_plot(True, "06_selected_optimized_rpy.png", "Selected optimized trajectories")


def _format_stats(stats: dict[str, float], digits: int = 4) -> str:
    return ", ".join(f"{key}={value:.{digits}f}" for key, value in stats.items())


def analyze(output: Path) -> dict[str, Any]:
    runs, load_failures = _load_runs(output)
    if not runs:
        raise RuntimeError("no complete audit samples are available")
    _write_plan_summary(output / "plan_summary.csv", runs)
    _write_motion_metrics(output / "motion_metrics.csv", runs)

    action_counts = [len(run.actions) for run in runs]
    count_distribution = Counter(action_counts)
    actions = [(run, action) for run in runs for action in run.actions]
    axis_counts = Counter(_primary_axis(action)[0] for _, action in actions)
    total_actions = len(actions)
    axis_percentages = {
        axis: 100.0 * axis_counts[axis] / total_actions if total_actions else 0.0
        for axis in (*AXES, "none")
    }
    direction_counts = Counter(_primary_direction(action) for _, action in actions)
    amplitude_stats = {
        axis: _summary([abs(float(action[axis])) for _, action in actions])
        for axis in AXES
    }
    primary_amplitude_stats = _summary(
        [_primary_axis(action)[1] for _, action in actions]
    )
    timing_stats = {
        "start_sec": _summary([float(action["start"]) for _, action in actions]),
        "end_sec": _summary([float(action["end"]) for _, action in actions]),
        "duration_sec": _summary(
            [float(action["end"]) - float(action["start"]) for _, action in actions]
        ),
        "normalized_start": _summary(
            [float(action["start"]) / run.duration_sec for run, action in actions]
        ),
        "normalized_end": _summary(
            [float(action["end"]) / run.duration_sec for run, action in actions]
        ),
        "normalized_duration": _summary(
            [
                (float(action["end"]) - float(action["start"])) / run.duration_sec
                for run, action in actions
            ]
        ),
    }

    signatures = Counter(_signature(run) for run in runs)
    duplicate_pairs = sum(count * (count - 1) // 2 for count in signatures.values())
    all_pairs = len(runs) * (len(runs) - 1) // 2
    duplicate_rate = duplicate_pairs / all_pairs if all_pairs else 0.0
    most_common = signatures.most_common(10)

    within_rows: list[dict[str, Any]] = []
    for sentence_id, _ in SENTENCES:
        sentence_runs = [run for run in runs if run.sentence_id == sentence_id]
        sentence_signatures = [_signature(run) for run in sentence_runs]
        count_sequences = [len(run.actions) for run in sentence_runs]
        axis_sequences = [
            tuple(_primary_axis(action)[0] for action in run.actions)
            for run in sentence_runs
        ]
        direction_sequences = [
            tuple(_primary_direction(action) for action in run.actions)
            for run in sentence_runs
        ]
        exact = len(set(sentence_signatures)) == 1 and len(sentence_runs) == RUNS_PER_TEXT
        structural = (
            len(set(count_sequences)) == 1
            and len(set(axis_sequences)) == 1
            and len(set(direction_sequences)) == 1
            and len(sentence_runs) == RUNS_PER_TEXT
        )
        within_rows.append(
            {
                "sentence_id": sentence_id,
                "available_runs": len(sentence_runs),
                "classification": (
                    "exact_signature_consistent"
                    if exact
                    else "structurally_consistent"
                    if structural
                    else "variable"
                ),
                "unique_signatures": len(set(sentence_signatures)),
                "signatures": sentence_signatures,
            }
        )
    _write_json(output / "within_text_consistency.json", within_rows)

    raw_matrix, raw_normalized, raw_pairs = _similarity_data(runs, final=False)
    final_matrix, final_normalized, final_pairs = _similarity_data(runs, final=True)
    _write_matrix(output / "trajectory_similarity_raw.csv", raw_matrix)
    _write_matrix(output / "trajectory_similarity_optimized.csv", final_matrix)

    raw_values = [value for value, _, _ in raw_pairs]
    final_values = [value for value, _, _ in final_pairs]
    raw_similarity_stats = {
        "mean": float(np.mean(raw_values)),
        "median": float(np.median(raw_values)),
        "min": float(np.min(raw_values)),
        "max": float(np.max(raw_values)),
        "mean_absolute": float(np.mean(np.abs(raw_values))),
        "median_absolute": float(np.median(np.abs(raw_values))),
    }
    final_similarity_stats = {
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "mean_absolute": float(np.mean(np.abs(final_values))),
        "median_absolute": float(np.median(np.abs(final_values))),
    }
    top_raw = sorted(raw_pairs, key=lambda item: item[0], reverse=True)[:10]
    top_final = sorted(final_pairs, key=lambda item: item[0], reverse=True)[:10]

    minimal_pair_rows: list[dict[str, Any]] = []
    for left_id, right_id in MINIMAL_PAIRS:
        left_runs = [run for run in runs if run.sentence_id == left_id]
        right_runs = [run for run in runs if run.sentence_id == right_id]
        left_counts = sorted({len(run.actions) for run in left_runs})
        right_counts = sorted({len(run.actions) for run in right_runs})
        left_axes = sorted({
            "-".join(_primary_axis(action)[0] for action in run.actions)
            for run in left_runs
        })
        right_axes = sorted({
            "-".join(_primary_axis(action)[0] for action in run.actions)
            for run in right_runs
        })
        left_directions = sorted({
            "/".join(_primary_direction(action) for action in run.actions)
            for run in left_runs
        })
        right_directions = sorted({
            "/".join(_primary_direction(action) for action in run.actions)
            for run in right_runs
        })
        left_amp = np.mean([
            _primary_axis(action)[1] for run in left_runs for action in run.actions
        ])
        right_amp = np.mean([
            _primary_axis(action)[1] for run in right_runs for action in run.actions
        ])
        left_timing = np.mean([
            float(action["start"]) / run.duration_sec
            for run in left_runs for action in run.actions
        ])
        right_timing = np.mean([
            float(action["start"]) / run.duration_sec
            for run in right_runs for action in run.actions
        ])
        shared_signatures = set(map(_signature, left_runs)) & set(map(_signature, right_runs))
        clearly_different = bool(
            left_counts != right_counts
            or left_axes != right_axes
            or left_directions != right_directions
        )
        nearly_identical = bool(
            not clearly_different
            and (
                bool(shared_signatures)
                or (
                    abs(left_amp - right_amp) <= AMPLITUDE_BIN_DEG
                    and abs(left_timing - right_timing) <= 0.05
                )
            )
        )
        minimal_pair_rows.append(
            {
                "pair": f"{left_id} vs {right_id}",
                "action_counts": f"{left_counts} vs {right_counts}",
                "axis_sequences": f"{left_axes} vs {right_axes}",
                "primary_directions": f"{left_directions} vs {right_directions}",
                "mean_primary_amplitude_deg": f"{left_amp:.3f} vs {right_amp:.3f}",
                "mean_normalized_start": f"{left_timing:.3f} vs {right_timing:.3f}",
                "raw_similarity": _pair_average(
                    left_id, right_id, raw_normalized, runs
                ),
                "optimized_similarity": _pair_average(
                    left_id, right_id, final_normalized, runs
                ),
                "shared_signature_count": len(shared_signatures),
                "clearly_different": clearly_different,
                "nearly_identical": nearly_identical,
            }
        )
    _write_json(output / "minimal_pairs.json", minimal_pair_rows)

    _plot_results(output, runs, raw_matrix, final_matrix)

    per_text_axis_sequences = {
        sentence_id: [
            "-".join(_primary_axis(action)[0] for action in run.actions)
            for run in runs if run.sentence_id == sentence_id
        ]
        for sentence_id, _ in SENTENCES
    }
    per_text_direction_sequences = {
        sentence_id: [
            "/".join(_primary_direction(action) for action in run.actions)
            for run in runs if run.sentence_id == sentence_id
        ]
        for sentence_id, _ in SENTENCES
    }
    exact_within = sum(
        row["classification"] == "exact_signature_consistent" for row in within_rows
    )
    variable_within = sum(row["classification"] == "variable" for row in within_rows)
    mean_similarity_change = final_similarity_stats["mean"] - raw_similarity_stats["mean"]

    planner_highly_similar = (
        len(signatures) <= max(8, len(runs) // 4)
        or signatures.most_common(1)[0][1] >= len(runs) * 0.25
    )
    compiler_high_similarity = raw_similarity_stats["mean"] >= 0.75
    optimizer_increases = mean_similarity_change >= 0.05
    if planner_highly_similar and compiler_high_similarity:
        classification = "C. Planner + Generator 都存在明显问题"
    elif planner_highly_similar:
        classification = "A. Planner diversity 不足"
    elif compiler_high_similarity:
        classification = "B. Generator primitive 太单一"
    else:
        classification = "D. 暂时无法判断"

    request_config = _read_json(output / "request_config.json")
    report_lines = [
        "# Motion Diversity Audit",
        "",
        "## 实验范围与真实调用链",
        "",
        "本实验使用当前 production 组件，唯一审计差异是每句话的 TTS PCM 和 alignment 只计算一次并供三次 Planner 调用复用：",
        "",
        "```text",
        "text → DoubaoTTS → align_speech → DeepSeekMotionBackend._request_plan()",
        "→ parse_motion_plan_with_rejections() → SparseMotionAction[]",
        "→ compile_motion_plan() → raw relative RPY",
        "→ MotionProcessor(start pose = [0,0,0]) → TrajectoryOptimizer → FinalTrajectory",
        "```",
        "",
        "`MotionPlan` / `MotionSegment` 未接入此路径。Final absolute trajectory 比较前减去首帧。所有 similarity 只在离线分析时线性归一化到 120 帧，production trajectory 未被重采样。",
        "",
        "## DeepSeek Motion 实际请求参数",
        "",
        f"- model: `{request_config['model']}`",
        f"- temperature: `{request_config['temperature']}`",
        "- top_p: not explicitly configured",
        f"- max_output_tokens: `{request_config['max_output_tokens']}`",
        f"- timeout_sec: `{request_config['timeout_sec']}`",
        "- reasoning: `effort=none`",
        "- user prompt: compact JSON `robot_text + duration_sec + segments`；segments 来自真实 TTS PCM alignment。",
        f"- system prompt SHA-256: `{request_config['system_prompt_sha256']}`",
        "",
        "<details><summary>System prompt</summary>",
        "",
        "```text",
        request_config["system_prompt"].rstrip(),
        "```",
        "</details>",
        "",
        "## 样本完整性",
        "",
        f"- Complete samples: **{len(runs)} / {len(SENTENCES) * RUNS_PER_TEXT}**",
        f"- Load failures: **{len(load_failures)}**",
        f"- Accepted actions: **{total_actions}**",
        f"- Rejected actions: **{sum(len(run.rejected_actions) for run in runs)}**",
        "",
        "## Plan-level diversity",
        "",
        f"Action count: {_format_stats(_summary(action_counts))}; distribution={dict(sorted(count_distribution.items()))}.",
        "",
        "Primary axis distribution: " + ", ".join(
            f"{axis}={axis_percentages[axis]:.2f}%" for axis in (*AXES, "none")
        ) + ".",
        "",
        "Signed primary-direction counts: " + str(dict(sorted(direction_counts.items()))) + ".",
        "",
        "Per-text axis and signed-direction sequences:",
        "",
        *[
            f"- {item}: axis=`{per_text_axis_sequences[item]}`; "
            f"direction=`{per_text_direction_sequences[item]}`"
            for item, _ in SENTENCES
        ],
        "",
        "Amplitude statistics in degree:",
        "",
        *[f"- abs({axis}): {_format_stats(amplitude_stats[axis])}" for axis in AXES],
        f"- primary amplitude: {_format_stats(primary_amplitude_stats)}",
        "",
        "Timing statistics:",
        "",
        *[f"- {name}: {_format_stats(stats)}" for name, stats in timing_stats.items()],
        "",
        "Signature includes action count, primary-axis sequence, signed primary-amplitude bins, normalized timing bins, and signed three-axis amplitude-vector bins.",
        "",
        f"Unique signatures: **{len(signatures)} / {len(runs)}**.",
        f"Exact duplicate pair rate: **{duplicate_rate:.2%}**.",
        "",
        "Most common signatures:",
        "",
        *[f"- `{signature}`: {count}" for signature, count in most_common],
        "",
        "## 同一句话内部稳定性",
        "",
        f"- Exact-signature consistent texts: **{exact_within} / {len(SENTENCES)}**",
        f"- Variable texts: **{variable_within} / {len(SENTENCES)}**",
        "",
        *[
            f"- {row['sentence_id']}: {row['classification']}; "
            f"unique signatures={row['unique_signatures']}"
            for row in within_rows
        ],
        "",
        "## Trajectory similarity",
        "",
        f"Raw cross-text run-pair cosine: {_format_stats(raw_similarity_stats)}.",
        f"Optimized cross-text run-pair cosine: {_format_stats(final_similarity_stats)}.",
        "Absolute-cosine values are included as a direction-insensitive secondary diagnostic; signed cosine remains the primary metric.",
        f"Mean change (optimized - raw): **{mean_similarity_change:+.4f}**.",
        "",
        "Top 10 most similar different-text raw trajectories:",
        "",
        *[
            f"- {left.key} vs {right.key}: {value:.6f}"
            for value, left, right in top_raw
        ],
        "",
        "Top 10 most similar different-text optimized trajectories:",
        "",
        *[
            f"- {left.key} vs {right.key}: {value:.6f}"
            for value, left, right in top_final
        ],
        "",
        "## Minimal pairs",
        "",
        "| Pair | Count | Axis sequence | Primary direction | Primary amplitude | Normalized start | Raw sim | Optimized sim | Assessment |",
        "|---|---|---|---|---|---|---:|---:|---|",
        *[
            "| {pair} | {action_counts} | {axis_sequences} | {primary_directions} | "
            "{mean_primary_amplitude_deg} | {mean_normalized_start} | {raw_similarity:.4f} | "
            "{optimized_similarity:.4f} | {assessment} |".format(
                **row,
                assessment=(
                    "clearly different"
                    if row["clearly_different"]
                    else "nearly identical"
                    if row["nearly_identical"]
                    else "partially different"
                ),
            )
            for row in minimal_pair_rows
        ],
        "",
        "## Motion metrics",
        "",
        "每个 sample 的 raw/final JSON 已按 roll/pitch/yaw 保存 peak velocity、peak acceleration 和 peak jerk。Optimizer 未执行 jerk limit；这些指标用于确认运动学变化，不参与 signature。",
        "",
        "## 四个结论",
        "",
        "### Q1：不同文本的 sparse plan 是否高度相似？",
        "",
        (
            "是。"
            if planner_highly_similar
            else "不是高度同质；当前 plan 显示出可观差异。"
        ) + f" 证据：48 个目标样本得到 {len(signatures)} 个 signature，最常见 signature 出现 {most_common[0][1]} 次，primary axis 分布见上文。",
        "",
        "### Q2：Compiler 是否把差异变成高度相似的 shape？",
        "",
        (
            "是，raw relative trajectory 的跨文本平均 cosine 已处于高相似区间。"
            if compiler_high_similarity
            else "没有足够证据认为 compiler 单独造成高度相似。"
        ) + f" Raw signed mean={raw_similarity_stats['mean']:.4f}, median={raw_similarity_stats['median']:.4f}, mean absolute={raw_similarity_stats['mean_absolute']:.4f}。",
        "",
        "### Q3：TrajectoryOptimizer 是否明显增加 similarity？",
        "",
        (
            "是。"
            if optimizer_increases
            else "否，未观察到至少 +0.05 的平均 cosine 增幅。"
        ) + f" Raw mean={raw_similarity_stats['mean']:.4f}，optimized mean={final_similarity_stats['mean']:.4f}，变化={mean_similarity_change:+.4f}。",
        "",
        "### Q4：主要问题判断",
        "",
        f"**{classification}**",
        "",
        "该判断只基于本次固定 16 句、每句 3 次的 production-path audit；阈值和 signature 仅用于诊断，不进入 runtime。",
        "",
        "## 产物索引",
        "",
        "- `plan_summary.csv`",
        "- `motion_metrics.csv`",
        "- `trajectory_similarity_raw.csv`",
        "- `trajectory_similarity_optimized.csv`",
        "- `within_text_consistency.json`",
        "- `minimal_pairs.json`",
        "- `samples/<ID>/`",
        "- `plots/`",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    analysis = {
        "successful_samples": len(runs),
        "planner_calls": len(runs),
        "unique_plan_signatures": len(signatures),
        "most_common_signature": most_common[0][0],
        "most_common_signature_frequency": most_common[0][1],
        "raw_similarity": raw_similarity_stats,
        "optimized_similarity": final_similarity_stats,
        "minimal_pairs_clearly_different": [
            row["pair"] for row in minimal_pair_rows if row["clearly_different"]
        ],
        "minimal_pairs_nearly_identical": [
            row["pair"] for row in minimal_pair_rows if row["nearly_identical"]
        ],
        "main_finding": classification,
        "load_failures": load_failures,
    }
    _write_json(output / "analysis_summary.json", analysis)
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=("all", "collect", "analyze"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="remove only the selected audit output directory before collection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if args.reset and output.exists():
        if output == PROJECT_ROOT or PROJECT_ROOT not in output.parents:
            raise SystemExit("refusing to reset an output outside the project")
        shutil.rmtree(output)
    try:
        if args.phase in {"all", "collect"}:
            asyncio.run(
                collect(
                    output,
                    args.config.expanduser().resolve(),
                    resume=args.resume,
                )
            )
        if args.phase in {"all", "analyze"}:
            summary = analyze(output)
            print("\nSamples:", summary["successful_samples"])
            print("Planner calls:", summary["planner_calls"])
            print("\nUnique plan signatures:", summary["unique_plan_signatures"])
            print(
                "Most common signature frequency:",
                summary["most_common_signature_frequency"],
            )
            print(
                "\nMean raw trajectory similarity:",
                f"{summary['raw_similarity']['mean']:.6f}",
            )
            print(
                "Mean optimized trajectory similarity:",
                f"{summary['optimized_similarity']['mean']:.6f}",
            )
            print(
                "\nMinimal pairs with clearly different plans:",
                summary["minimal_pairs_clearly_different"],
            )
            print(
                "Minimal pairs with nearly identical plans:",
                summary["minimal_pairs_nearly_identical"],
            )
            print("\nMain finding:", summary["main_finding"])
            print("Report:", output / "report.md")
            print("Plots:", output / "plots")
    except Exception as exc:
        raise SystemExit(f"motion diversity audit failed: {type(exc).__name__}: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
