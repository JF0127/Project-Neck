"""Diagnose high-energy, low-speech, text-free audio without modifying videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "videos"
DEFAULT_TRANSCRIPTS = DEFAULT_INPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "audio_corruption_debug"
)
DEFAULT_VAD_MODEL = PROJECT_ROOT / "models" / "silero_vad" / "silero_vad.jit"
QWEN_SUFFIX = ".qwen_asr_v1.json"
SAMPLE_RATE = 16_000
VAD_WINDOW_SAMPLES = 512
ANALYSIS_WINDOW_SEC = 1.0
HIGH_ENERGY_DBFS = -30.0
VAD_THRESHOLD = 0.5
LOW_SPEECH_RATIO = 0.10
MIN_CONTIGUOUS_ANOMALY_SEC = 10.0
MIN_ANOMALOUS_RATIO = 0.15
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class AudioCorruptionDebugError(RuntimeError):
    """Raised when read-only audio diagnosis cannot run or publish safely."""


@dataclass(frozen=True)
class AnalysisWindow:
    start: float
    end: float
    rms: float
    rms_dbfs: float
    vad_speech_ratio: float
    has_text: bool
    anomalous: bool


class SileroScorer:
    """Score consecutive 512-sample windows with a local Silero JIT model."""

    def __init__(self, model_path: Path) -> None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise AudioCorruptionDebugError(f"Silero VAD model not found: {path}")
        try:
            import torch
        except ImportError as exc:
            raise AudioCorruptionDebugError("PyTorch is required for Silero VAD") from exc
        self._torch = torch
        try:
            self._model = torch.jit.load(str(path), map_location="cpu")
            self._model.eval()
        except Exception as exc:
            raise AudioCorruptionDebugError(
                f"could not load Silero VAD model {path}: {exc}"
            ) from exc
        self.model_path = path

    def probabilities(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()
        probabilities: list[float] = []
        durations: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(waveform), VAD_WINDOW_SAMPLES):
                chunk = waveform[start : start + VAD_WINDOW_SAMPLES]
                actual_samples = len(chunk)
                if actual_samples == 0:
                    continue
                if actual_samples < VAD_WINDOW_SAMPLES:
                    chunk = np.pad(chunk, (0, VAD_WINDOW_SAMPLES - actual_samples))
                value = float(
                    self._model(
                        self._torch.from_numpy(chunk.astype(np.float32, copy=False)),
                        SAMPLE_RATE,
                    ).item()
                )
                if not math.isfinite(value):
                    raise AudioCorruptionDebugError("Silero VAD returned NaN or Inf")
                probabilities.append(value)
                durations.append(actual_samples / SAMPLE_RATE)
        return (
            np.asarray(probabilities, dtype=np.float32),
            np.asarray(durations, dtype=np.float64),
        )


def _decode_audio(video: Path, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise AudioCorruptionDebugError(
            f"ffmpeg audio decode failed for {video.name}: "
            f"{details[-1] if details else 'unknown error'}"
        )
    if not result.stdout or len(result.stdout) % 2:
        raise AudioCorruptionDebugError(f"decoded audio is empty or invalid: {video.name}")
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _transcript_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise AudioCorruptionDebugError(f"transcript directory not found: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{QWEN_SUFFIX}")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            audio_name = document["audio"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AudioCorruptionDebugError(f"invalid Qwen transcript: {path}") from exc
        if not isinstance(audio_name, str) or Path(audio_name).name != audio_name:
            raise AudioCorruptionDebugError(f"invalid audio name in Qwen transcript: {path}")
        stem = Path(audio_name).stem
        if stem in index:
            raise AudioCorruptionDebugError(
                f"multiple Qwen transcripts resolve to video stem {stem}: "
                f"{index[stem]}, {path}"
            )
        index[stem] = path
    return index


def _load_text_intervals(path: Path, duration: float) -> list[tuple[float, float]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioCorruptionDebugError(f"could not read Qwen transcript: {path}") from exc
    if document.get("version") != "qwen_asr_v1":
        raise AudioCorruptionDebugError(f"unsupported Qwen transcript version: {path}")
    timestamps = document.get("timestamps")
    if not isinstance(timestamps, list):
        raise AudioCorruptionDebugError(f"Qwen timestamps must be a list: {path}")

    intervals: list[tuple[float, float]] = []
    previous_start = 0.0
    for index, item in enumerate(timestamps, 1):
        try:
            text = str(item["text"]).strip()
            start = float(item["start_time"])
            end = float(item["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioCorruptionDebugError(
                f"invalid Qwen timestamp at {path}, index {index}"
            ) from exc
        if (
            not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_start - 1e-6
            or end < start
        ):
            raise AudioCorruptionDebugError(
                f"invalid Qwen timestamp values at {path}, index {index}"
            )
        clipped_start = min(duration, max(0.0, start))
        clipped_end = min(duration, max(0.0, end))
        if clipped_end > clipped_start:
            intervals.append((clipped_start, clipped_end))
        previous_start = start
    return _merge_intervals(intervals)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _longest_no_text(duration: float, text_intervals: list[tuple[float, float]]) -> float:
    cursor = 0.0
    longest = 0.0
    for start, end in text_intervals:
        longest = max(longest, start - cursor)
        cursor = max(cursor, end)
    return max(longest, duration - cursor)


def _overlaps_text(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(text_start < end and text_end > start for text_start, text_end in intervals)


def _speech_ratio(
    start: float,
    end: float,
    probabilities: np.ndarray,
    durations: np.ndarray,
) -> float:
    weighted_speech = 0.0
    covered = 0.0
    cursor = 0.0
    for probability, duration in zip(probabilities, durations):
        frame_start = cursor
        frame_end = cursor + float(duration)
        cursor = frame_end
        overlap = max(0.0, min(end, frame_end) - max(start, frame_start))
        if overlap <= 0.0:
            continue
        covered += overlap
        if float(probability) >= VAD_THRESHOLD:
            weighted_speech += overlap
        if frame_start >= end:
            break
    return weighted_speech / covered if covered > 0.0 else 0.0


def _analysis_windows(
    waveform: np.ndarray,
    text_intervals: list[tuple[float, float]],
    probabilities: np.ndarray,
    vad_durations: np.ndarray,
) -> list[AnalysisWindow]:
    duration = len(waveform) / SAMPLE_RATE
    windows: list[AnalysisWindow] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + ANALYSIS_WINDOW_SEC)
        first_sample = round(start * SAMPLE_RATE)
        last_sample = round(end * SAMPLE_RATE)
        samples = waveform[first_sample:last_sample]
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
        speech_ratio = _speech_ratio(
            start, end, probabilities, vad_durations
        )
        has_text = _overlaps_text(start, end, text_intervals)
        anomalous = bool(
            rms_dbfs >= HIGH_ENERGY_DBFS
            and speech_ratio <= LOW_SPEECH_RATIO
            and not has_text
        )
        windows.append(
            AnalysisWindow(
                start=start,
                end=end,
                rms=rms,
                rms_dbfs=rms_dbfs,
                vad_speech_ratio=speech_ratio,
                has_text=has_text,
                anomalous=anomalous,
            )
        )
        start = end
    return windows


def _anomalous_intervals(windows: list[AnalysisWindow]) -> list[dict[str, Any]]:
    groups: list[list[AnalysisWindow]] = []
    for window in windows:
        if not window.anomalous:
            continue
        if groups and abs(groups[-1][-1].end - window.start) <= 1e-9:
            groups[-1].append(window)
        else:
            groups.append([window])

    results: list[dict[str, Any]] = []
    for group in groups:
        duration = sum(window.end - window.start for window in group)
        rms_square = sum(
            window.rms * window.rms * (window.end - window.start) for window in group
        ) / duration
        speech = sum(
            window.vad_speech_ratio * (window.end - window.start) for window in group
        ) / duration
        results.append(
            {
                "start_sec": round(group[0].start, 6),
                "end_sec": round(group[-1].end, 6),
                "duration_sec": round(duration, 6),
                "mean_rms_dbfs": round(20.0 * math.log10(max(math.sqrt(rms_square), 1e-12)), 6),
                "vad_speech_ratio": round(speech, 6),
                "reason": "high_energy_low_speech_no_text",
            }
        )
    return results


def _diagnose_video(
    video: Path,
    transcript: Path,
    scorer: SileroScorer,
    ffmpeg: str,
) -> dict[str, Any]:
    waveform = _decode_audio(video, ffmpeg)
    duration = len(waveform) / SAMPLE_RATE
    text_intervals = _load_text_intervals(transcript, duration)
    probabilities, vad_durations = scorer.probabilities(waveform)
    windows = _analysis_windows(
        waveform, text_intervals, probabilities, vad_durations
    )
    intervals = _anomalous_intervals(windows)
    anomalous_total = sum(item["duration_sec"] for item in intervals)
    anomalous_ratio = anomalous_total / duration
    longest_anomaly = max(
        (item["duration_sec"] for item in intervals), default=0.0
    )
    suspected = bool(
        longest_anomaly >= MIN_CONTIGUOUS_ANOMALY_SEC
        or anomalous_ratio >= MIN_ANOMALOUS_RATIO
    )
    return {
        "video": video.name,
        "status": "suspected_corrupt_audio" if suspected else "ok",
        "audio_duration_sec": round(duration, 6),
        "longest_no_text_sec": round(_longest_no_text(duration, text_intervals), 6),
        "anomalous_total_sec": round(anomalous_total, 6),
        "anomalous_ratio": round(anomalous_ratio, 6),
        "anomalous_intervals": intervals,
        "transcript": str(transcript),
    }


def _atomic_publish(staging: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise AudioCorruptionDebugError(
            f"output already exists (use --force to replace it): {output}"
        )
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def diagnose_directory(
    input_root: Path,
    transcript_root: Path,
    output: Path,
    vad_model: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    transcript_root = transcript_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not input_root.is_dir():
        raise AudioCorruptionDebugError(f"input directory not found: {input_root}")
    if output.exists() and not force:
        raise AudioCorruptionDebugError(
            f"output already exists (use --force to replace it): {output}"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioCorruptionDebugError("ffmpeg is required for read-only audio decoding")
    videos = sorted(
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not videos:
        raise AudioCorruptionDebugError(f"no supported videos found in: {input_root}")
    transcripts = _transcript_index(transcript_root)
    missing = [video.name for video in videos if video.stem not in transcripts]
    if missing:
        raise AudioCorruptionDebugError(
            "matching Qwen transcripts are missing for: " + ", ".join(missing)
        )

    scorer = SileroScorer(vad_model)
    records: list[dict[str, Any]] = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        records.append(
            _diagnose_video(
                video, transcripts[video.stem], scorer, ffmpeg
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        with (staging / "results.jsonl").open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        status_counts = {
            status: sum(record["status"] == status for record in records)
            for status in ("ok", "suspected_corrupt_audio")
        }
        manifest = {
            "version": "audio_corruption_debug_v1",
            "complete": True,
            "input_root": str(input_root),
            "transcript_root": str(transcript_root),
            "video_count": len(videos),
            "status_counts": status_counts,
            "audio_decode": "ffmpeg pipe to 16000 Hz mono pcm_s16le; source videos unchanged",
            "qwen_transcript_suffix": QWEN_SUFFIX,
            "silero_vad_model": str(scorer.model_path),
            "thresholds": {
                "analysis_window_sec": ANALYSIS_WINDOW_SEC,
                "high_energy_rms_dbfs": HIGH_ENERGY_DBFS,
                "vad_probability": VAD_THRESHOLD,
                "low_speech_ratio_max": LOW_SPEECH_RATIO,
                "minimum_contiguous_anomaly_sec": MIN_CONTIGUOUS_ANOMALY_SEC,
                "minimum_anomalous_audio_ratio": MIN_ANOMALOUS_RATIO,
                "low_text": "no Qwen timestamp overlap in the analysis window",
            },
            "status_rule": (
                "suspected_corrupt_audio when the longest contiguous anomalous interval "
                ">= 10 seconds OR anomalous_total_sec/audio_duration_sec >= 0.15"
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish(staging, output, force)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose high-energy, low-speech, text-free intervals in video audio"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=DEFAULT_TRANSCRIPTS,
        help="directory recursively containing *.qwen_asr_v1.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD_MODEL)
    parser.add_argument("--force", action="store_true", help="replace existing output")
    args = parser.parse_args(argv)
    try:
        manifest = diagnose_directory(
            args.input,
            args.transcripts,
            args.output,
            args.vad_model,
            force=args.force,
        )
    except (OSError, ValueError, AudioCorruptionDebugError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Videos diagnosed: {manifest['video_count']}")
    print(f"OK: {manifest['status_counts']['ok']}")
    print(
        "Suspected corrupt audio: "
        f"{manifest['status_counts']['suspected_corrupt_audio']}"
    )
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
