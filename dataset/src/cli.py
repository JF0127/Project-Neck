"""Command-line interface for candidate discovery."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.crawler.discover import DiscoveryError, discover_videos_with_stats
from src.crawler.downloader import DownloadError, download_videos
from src.crawler.storage import load_jsonl, save_jsonl, save_preview, save_validation_preview
from src.crawler.validation import stderr_progress, validate_candidates
from src.features.asr import ASRError, DEFAULT_MODEL, extract_speech_clips
from src.features.mediapipe import MediaPipeFeatureError, extract_mediapipe_clips
from src.features.neck_pose import NeckPoseError, extract_neck_pose_clips
from src.features.neck_pose_v0 import NeckPoseV0Error, validate_neck_pose_clips
from src.features.neck_smoothing_v0 import NeckSmoothingV0Error, compare_smoothing_clips
from src.fragments.fragment_v1 import (
    DEFAULT_LONG_SPLIT_PAUSE,
    DEFAULT_ABSOLUTE_MAX_DURATION,
    DEFAULT_MAX_CONTEXT_PADDING,
    DEFAULT_MIN_DURATION,
    DEFAULT_SOFT_MAX_DURATION,
    DEFAULT_STRONG_PAUSE,
    FragmentConfig,
    build_fragment_dataset,
)
from src.splits.split_v1 import build_split_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "youtube.yaml"
DEFAULT_MEDIAPIPE_CONFIG = PROJECT_ROOT / "configs" / "mediapipe.yaml"
DEFAULT_NECK_FEATURES = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/features/mediapipe_v1/clips"
DEFAULT_NECK_VIDEOS = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/clean_v1/clips"
DEFAULT_NECK_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/analysis/neck_pose_v0"
DEFAULT_SMOOTHING_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/analysis/neck_smoothing_v0"
DEFAULT_NECK_V1_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/features/neck_pose_v1"
DEFAULT_SPEECH_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/features/speech_v1"
DEFAULT_FRAGMENT_SPEECH_INPUT = DEFAULT_SPEECH_OUTPUT / "clips"
DEFAULT_FRAGMENT_NECK_INPUT = DEFAULT_NECK_V1_OUTPUT / "clips"
DEFAULT_FRAGMENT_CLEAN_INPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/clean_v1"
DEFAULT_FRAGMENT_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/fragments/fragment_v1"
DEFAULT_SPLIT_INPUT = DEFAULT_FRAGMENT_OUTPUT
DEFAULT_SPLIT_OUTPUT = PROJECT_ROOT / "datasets/zhubo_shuo_lianbo/splits/split_v1"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover YouTube candidate videos")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover", help="search and save candidate metadata")
    discover.add_argument("--query", help="override the configured search query")
    discover.add_argument("--limit", type=int, help="override the configured result limit")
    subparsers.add_parser("validate", help="validate discovered candidates using full metadata")
    download = subparsers.add_parser("download", help="download validated videos")
    download.add_argument("--limit", type=int, help="download only the first N validated videos")
    clean = subparsers.add_parser("clean", help="extract stable upper-body presenter clips")
    clean.add_argument("--input", required=True, help="directory containing source MP4 videos")
    clean.add_argument("--limit", type=int, help="process only the first N videos")
    clean.add_argument("--force", action="store_true", help="reprocess sources already in Clean V1 JSONL")
    mediapipe = subparsers.add_parser(
        "extract-mediapipe", help="extract raw MediaPipe Face Landmarker features"
    )
    mediapipe.add_argument("--input", help="Clean V1 clips directory")
    mediapipe.add_argument("--output", help="MediaPipe V1 output directory")
    mediapipe.add_argument("--model", help="Face Landmarker .task model path")
    mediapipe.add_argument("--limit", type=int, help="process only the first N clips")
    mediapipe.add_argument("--force", action="store_true", help="reprocess completed clips")
    neck = subparsers.add_parser(
        "validate-neck-pose", help="visualize the experimental candidate RPY convention"
    )
    neck.add_argument("--input", help="MediaPipe V1 clip feature directory")
    neck.add_argument("--video-input", help="corresponding Clean V1 clips directory")
    neck.add_argument("--output", help="Neck Pose V0 analysis output directory")
    neck.add_argument("--limit", type=int, help="process only the first N clips")
    neck.add_argument("--force", action="store_true", help="regenerate completed diagnostics")
    smoothing = subparsers.add_parser(
        "compare-neck-smoothing", help="compare candidate Euler low-pass cutoffs"
    )
    smoothing.add_argument("--input", help="Neck Pose V0 analysis directory")
    smoothing.add_argument("--video-input", help="corresponding Clean V1 clips directory")
    smoothing.add_argument("--mediapipe-input", help="MediaPipe V1 clip feature directory")
    smoothing.add_argument("--output", help="smoothing comparison output directory")
    smoothing.add_argument("--limit", type=int, help="process only the first N clips")
    smoothing.add_argument("--force", action="store_true", help="regenerate completed comparisons")
    neck_v1 = subparsers.add_parser(
        "extract-neck-pose", help="generate formal Neck Pose V1 ground truth"
    )
    neck_v1.add_argument("--input", help="MediaPipe V1 clip feature directory")
    neck_v1.add_argument("--output", help="Neck Pose V1 output directory")
    neck_v1.add_argument("--limit", type=int, help="process only the first N clips")
    neck_v1.add_argument("--force", action="store_true", help="reprocess completed clips")
    speech = subparsers.add_parser(
        "extract-speech", help="extract standardized audio and Chinese word-timestamp ASR"
    )
    speech.add_argument("--input", help="Clean V1 clips directory")
    speech.add_argument("--output", help="Speech V1 output directory")
    speech.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model name or path")
    speech.add_argument("--device", help="ASR device: auto, cuda, or cpu")
    speech.add_argument("--compute-type", help="CTranslate2 compute type")
    speech.add_argument("--limit", type=int, help="process only the first N clips")
    speech.add_argument("--force", action="store_true", help="reprocess completed clips")
    fragments = subparsers.add_parser(
        "build-fragments", help="build lookahead spoken-utterance Fragment V1.2.1 samples"
    )
    fragments.add_argument("--speech-input", help="Speech V1 clips directory")
    fragments.add_argument("--neck-input", help="Neck Pose V1 clips directory")
    fragments.add_argument("--clean-input", help="Clean V1 root containing metadata.jsonl")
    fragments.add_argument("--output", help="Fragment V1 output root")
    fragments.add_argument("--limit", type=int, help="process only the first N Speech V1 clips")
    fragments.add_argument("--force", action="store_true", help="rebuild completed clips")
    fragments.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    fragments.add_argument(
        "--soft-max-duration",
        "--hard-max-duration",
        "--max-duration",
        dest="soft_max_duration",
        type=float,
        default=DEFAULT_SOFT_MAX_DURATION,
        help=(
            "lookahead start (default: 20s; --hard-max-duration and "
            "--max-duration are deprecated aliases, not hard cutoffs)"
        ),
    )
    fragments.add_argument(
        "--absolute-max-duration",
        type=float,
        default=DEFAULT_ABSOLUTE_MAX_DURATION,
        help="absolute complete-segment safety limit (default: 30s)",
    )
    fragments.add_argument(
        "--strong-pause",
        "--sentence-pause",
        dest="strong_pause",
        type=float,
        default=DEFAULT_STRONG_PAUSE,
        help="strong segment-gap boundary (default: 0.45s; --sentence-pause is an alias)",
    )
    fragments.add_argument("--long-split-pause", type=float, default=DEFAULT_LONG_SPLIT_PAUSE)
    fragments.add_argument(
        "--max-context-padding", type=float, default=DEFAULT_MAX_CONTEXT_PADDING
    )
    split = subparsers.add_parser(
        "build-split", help="build deterministic source-video-grouped split manifests"
    )
    split.add_argument("--input", default=DEFAULT_SPLIT_INPUT, help="Fragment V1 root")
    split.add_argument("--output", default=DEFAULT_SPLIT_OUTPUT, help="Split V1 output root")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--test-ratio", type=float, default=0.2)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--force", action="store_true", help="replace an existing split")
    return parser


def _print_channels(records: list[dict[str, Any]]) -> None:
    print("Channels:")
    channel_counts = Counter(item.get("channel") or "Unknown" for item in records)
    for channel, count in sorted(channel_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {channel}: {count}")


def _run_discover(args: argparse.Namespace, config: dict[str, Any]) -> int:
    search = config["search"]
    filter_config = config["filter"]
    output = config["output"]
    query = args.query if args.query is not None else search["query"]
    limit = args.limit if args.limit is not None else int(search["limit"])
    title_keyword = filter_config["title_keyword"]
    output_path = _project_path(output["candidates"])

    candidates, stats = discover_videos_with_stats(query, limit, title_keyword)
    preview_path = output_path.with_name("candidates_preview.txt")
    save_jsonl(candidates, output_path)
    save_preview(candidates, preview_path)

    print(f"Search query: {query}")
    print(f"Raw results: {stats['raw_results']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Output: {output_path}")
    _print_channels(candidates)
    return 0


def _run_validate(config: dict[str, Any]) -> int:
    validation = config["validation"]
    candidates_path = _project_path(validation["candidates"])
    validated_path = _project_path(validation["validated"])
    rejected_path = _project_path(validation["rejected"])
    preview_path = _project_path(validation["preview"])
    candidates = load_jsonl(candidates_path)
    validated, rejected = validate_candidates(
        candidates,
        str(validation["title_keyword"]),
        set(validation["allowed_channels"]),
        stderr_progress,
    )
    save_jsonl(validated, validated_path)
    save_jsonl(rejected, rejected_path)
    save_validation_preview(validated, rejected, preview_path)

    print(f"Candidates: {len(candidates)}")
    print(f"Validated: {len(validated)}")
    print(f"Rejected: {len(rejected)}")
    print()
    _print_channels(validated)
    durations = [float(item["duration"]) for item in validated]
    print()
    print("Duration:")
    if durations:
        print(f"  min: {min(durations):g}s")
        print(f"  max: {max(durations):g}s")
        print(f"  mean: {statistics.mean(durations):.2f}s")
        print(f"  median: {statistics.median(durations):g}s")
    else:
        for label in ("min", "max", "mean", "median"):
            print(f"  {label}: n/a")
    if rejected:
        print()
        print("Rejected videos:")
        for item in rejected:
            print(
                f"{item.get('id') or 'Unknown'} | "
                f"{item.get('title') or 'Untitled'} | {item['reject_reason']}"
            )
    print()
    print(f"Validated output: {validated_path}")
    print(f"Rejected output: {rejected_path}")
    print(f"Preview output: {preview_path}")
    return 0


def _run_download(args: argparse.Namespace, config: dict[str, Any]) -> int:
    download_config = config["download"]
    validated_path = _project_path(download_config["validated"])
    if not validated_path.is_file():
        discovery_path = _project_path(config["validation"]["candidates"])
        if discovery_path.is_file():
            raise DownloadError(
                f"validated input not found: {validated_path}. "
                "Run `python -m src.cli validate` first."
            )
        raise DownloadError(f"validated input not found: {validated_path}")

    records = load_jsonl(validated_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("download limit must be greater than zero")
        records = records[: args.limit]
    result = download_videos(records, _project_path(download_config["dataset_root"]))
    print(f"Total: {result['total']}")
    print(f"Downloaded: {result['downloaded']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print(f"Output: {result['videos_dir']}")
    return 0 if result["failed"] == 0 else 2


def _run_clean(args: argparse.Namespace) -> int:
    # Keep heavyweight Torch/Ultralytics imports out of unrelated CLI commands.
    from src.cleaning.clean_v1 import CleanError, clean_videos

    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise CleanError(f"input video directory not found: {input_dir}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("clean limit must be greater than zero")
    video_paths = sorted(input_dir.glob("*.mp4"))
    if args.limit is not None:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        raise CleanError(f"no MP4 videos found in: {input_dir}")
    output_root = input_dir.parent / "clean_v1"
    result = clean_videos(video_paths, output_root, force=args.force)
    accepted = result["accepted"]
    rejected = result["rejected"]
    durations = [float(item["duration"]) for item in accepted]

    accepted_total = sum(durations)
    rejected_total = sum(float(item["duration"]) for item in rejected)
    source_total = float(result["source_total_duration"])
    accepted_sources = {str(item["source_video_id"]) for item in accepted}
    all_sources = {path.stem for path in video_paths}
    empty_sources = sorted(all_sources - accepted_sources)

    print()
    print(f"Input videos: {result['input_videos']}")
    print(f"Processed source videos: {result['processed_sources']}")
    print(f"Skipped source videos: {result['skipped_sources']}")
    print(f"Failed source videos: {result['failed_sources']}")
    print(f"Total source duration: {source_total:.2f}s")
    print()
    print(f"Detected shots: {result['detected_shots']}")
    print(f"Accepted shots: {len(accepted)}")
    print(f"Rejected shots: {len(rejected)}")
    print()
    print(f"Accepted total duration: {accepted_total:.2f}s")
    print(f"Rejected total duration: {rejected_total:.2f}s")
    ratio = accepted_total / source_total if source_total else 0.0
    print(f"Acceptance duration ratio: {ratio:.2%}")
    if durations:
        print(f"Average accepted clip duration: {statistics.mean(durations):.2f}s")
        print(f"Median accepted clip duration: {statistics.median(durations):.2f}s")
        print(f"Min accepted clip duration: {min(durations):.2f}s")
        print(f"Max accepted clip duration: {max(durations):.2f}s")
    else:
        for label in ("Average", "Median", "Min", "Max"):
            print(f"{label} accepted clip duration: n/a")
    print()
    print("Rejected:")
    reject_reasons = (
        "too_short", "no_person", "multiple_people", "no_face",
        "multiple_faces", "face_too_small", "shoulders_not_visible",
        "unstable_detection", "crop_invalid", "export_failed",
    )
    for reason in reject_reasons:
        print(f"  {reason}: {result['reject_counts'].get(reason, 0)}")
    print()
    print(f"Source videos with accepted clips: {len(accepted_sources)}")
    print(f"Source videos without accepted clips: {len(empty_sources)}")
    if empty_sources:
        print("Videos without accepted clips:")
        for video_id in empty_sources:
            print(f"  {video_id}")
    print(f"Clips: {result['clips_dir']}")
    print(f"Debug: {result['debug_dir']}")
    return 0


def _run_extract_mediapipe(args: argparse.Namespace) -> int:
    config = _load_config(DEFAULT_MEDIAPIPE_CONFIG)
    input_dir = _project_path(args.input or config["input"])
    output_root = _project_path(args.output or config["output"])
    model_path = _project_path(args.model or config["model"])
    if not input_dir.is_dir():
        raise MediaPipeFeatureError(f"input clips directory not found: {input_dir}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("MediaPipe limit must be greater than zero")
    clip_paths = sorted(input_dir.glob("*.mp4"))
    if args.limit is not None:
        clip_paths = clip_paths[: args.limit]
    if not clip_paths:
        raise MediaPipeFeatureError(f"no MP4 clips found in: {input_dir}")
    result = extract_mediapipe_clips(clip_paths, output_root, model_path, force=args.force)
    print()
    print(f"Input clips: {result['input_clips']}")
    print(f"Processed: {result['processed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print()
    print(f"Total frames: {result['total_frames']}")
    print(f"Valid frames: {result['valid_frames']}")
    print(f"Invalid frames: {result['invalid_frames']}")
    print(f"Overall valid ratio: {result['valid_ratio']:.2%}")
    return 0 if result["failed"] == 0 else 2


def _run_validate_neck_pose(args: argparse.Namespace) -> int:
    feature_root = _project_path(args.input) if args.input else DEFAULT_NECK_FEATURES
    video_root = _project_path(args.video_input) if args.video_input else DEFAULT_NECK_VIDEOS
    output_root = _project_path(args.output) if args.output else DEFAULT_NECK_OUTPUT
    if not feature_root.is_dir():
        raise NeckPoseV0Error(f"MediaPipe V1 clips directory not found: {feature_root}")
    if not video_root.is_dir():
        raise NeckPoseV0Error(f"Clean V1 clips directory not found: {video_root}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("neck pose validation limit must be greater than zero")
    feature_dirs = sorted(path for path in feature_root.iterdir() if path.is_dir())
    if args.limit is not None:
        feature_dirs = feature_dirs[: args.limit]
    if not feature_dirs:
        raise NeckPoseV0Error(f"no MediaPipe V1 clip directories found in: {feature_root}")
    result = validate_neck_pose_clips(feature_dirs, video_root, output_root, force=args.force)
    print()
    print(f"Input clips: {result['input_clips']}")
    print(f"Processed: {result['processed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    return 0 if result["failed"] == 0 else 2


def _run_compare_neck_smoothing(args: argparse.Namespace) -> int:
    pose_root = _project_path(args.input) if args.input else DEFAULT_NECK_OUTPUT
    video_root = _project_path(args.video_input) if args.video_input else DEFAULT_NECK_VIDEOS
    mediapipe_root = (
        _project_path(args.mediapipe_input) if args.mediapipe_input else DEFAULT_NECK_FEATURES
    )
    output_root = _project_path(args.output) if args.output else DEFAULT_SMOOTHING_OUTPUT
    for label, path in (
        ("Neck Pose V0", pose_root),
        ("Clean V1 clips", video_root),
        ("MediaPipe V1 clips", mediapipe_root),
    ):
        if not path.is_dir():
            raise NeckSmoothingV0Error(f"{label} directory not found: {path}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("smoothing comparison limit must be greater than zero")
    pose_dirs = sorted(path for path in pose_root.iterdir() if path.is_dir())
    if args.limit is not None:
        pose_dirs = pose_dirs[: args.limit]
    if not pose_dirs:
        raise NeckSmoothingV0Error(f"no Neck Pose V0 clip directories found in: {pose_root}")
    result = compare_smoothing_clips(
        pose_dirs, mediapipe_root, video_root, output_root, force=args.force
    )
    print()
    print(f"Input clips: {result['input_clips']}")
    print(f"Processed: {result['processed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    return 0 if result["failed"] == 0 else 2


def _run_extract_neck_pose(args: argparse.Namespace) -> int:
    input_root = _project_path(args.input) if args.input else DEFAULT_NECK_FEATURES
    output_root = _project_path(args.output) if args.output else DEFAULT_NECK_V1_OUTPUT
    if not input_root.is_dir():
        raise NeckPoseError(f"MediaPipe V1 clips directory not found: {input_root}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Neck Pose V1 limit must be greater than zero")
    feature_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if args.limit is not None:
        feature_dirs = feature_dirs[: args.limit]
    if not feature_dirs:
        raise NeckPoseError(f"no MediaPipe V1 clip directories found in: {input_root}")
    result = extract_neck_pose_clips(feature_dirs, output_root, force=args.force)
    print()
    print(f"Input clips: {result['input_clips']}")
    print(f"Processed: {result['processed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print()
    print(f"Total frames: {result['total_frames']}")
    print(f"Source valid frames: {result['source_valid_frames']}")
    print(f"Interpolated frames: {result['interpolated_frames']}")
    print(f"Remaining invalid frames: {result['remaining_invalid_frames']}")
    print(f"Final valid ratio: {result['final_valid_ratio']:.2%}")
    return 0 if result["failed"] == 0 else 2


def _run_extract_speech(args: argparse.Namespace) -> int:
    input_root = _project_path(args.input) if args.input else DEFAULT_NECK_VIDEOS
    output_root = _project_path(args.output) if args.output else DEFAULT_SPEECH_OUTPUT
    if not input_root.is_dir():
        raise ASRError(f"Clean V1 clips directory not found: {input_root}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Speech V1 limit must be greater than zero")
    video_paths = sorted(input_root.glob("*.mp4"))
    if args.limit is not None:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        raise ASRError(f"no Clean V1 MP4 clips found in: {input_root}")
    result = extract_speech_clips(
        video_paths,
        output_root,
        model_name=args.model,
        requested_device=args.device,
        requested_compute_type=args.compute_type,
        force=args.force,
        neck_root=DEFAULT_NECK_V1_OUTPUT / "clips",
    )
    print()
    print(f"Input clips: {result['input_clips']}")
    print(f"Processed: {result['processed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print()
    print(f"Total audio duration: {result['total_audio_duration']:.2f}s")
    print(f"Total segments: {result['total_segments']}")
    print(f"Total words: {result['total_words']}")
    return 0 if result["failed"] == 0 else 2


def _run_build_fragments(args: argparse.Namespace) -> int:
    config = FragmentConfig(
        min_duration=args.min_duration,
        soft_max_duration=args.soft_max_duration,
        absolute_max_duration=args.absolute_max_duration,
        strong_pause=args.strong_pause,
        long_split_pause=args.long_split_pause,
        max_context_padding=args.max_context_padding,
    )
    summary = build_fragment_dataset(
        _project_path(args.speech_input) if args.speech_input else DEFAULT_FRAGMENT_SPEECH_INPUT,
        _project_path(args.neck_input) if args.neck_input else DEFAULT_FRAGMENT_NECK_INPUT,
        _project_path(args.clean_input) if args.clean_input else DEFAULT_FRAGMENT_CLEAN_INPUT,
        _project_path(args.output) if args.output else DEFAULT_FRAGMENT_OUTPUT,
        config=config,
        limit=args.limit,
        force=args.force,
    )
    return 0 if summary["failed_clips"] == 0 else 2


def _run_build_split(args: argparse.Namespace) -> int:
    summary = build_split_dataset(
        _project_path(args.input),
        _project_path(args.output),
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        force=args.force,
    )
    actual = summary["actual_ratios"]
    print(f"Output: {_project_path(args.output)}")
    print(
        "Source videos: "
        f"train={actual['source_train']:.2%}, "
        f"val={actual['source_val']:.2%}, test={actual['source_test']:.2%}"
    )
    print(
        "Fragments: "
        f"train={actual['fragment_train']:.2%}, "
        f"val={actual['fragment_val']:.2%}, test={actual['fragment_test']:.2%}"
    )
    print(
        "Duration: "
        f"train={actual['duration_train']:.2%}, "
        f"val={actual['duration_val']:.2%}, test={actual['duration_test']:.2%}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "clean":
            return _run_clean(args)
        if args.command == "extract-mediapipe":
            return _run_extract_mediapipe(args)
        if args.command == "validate-neck-pose":
            return _run_validate_neck_pose(args)
        if args.command == "compare-neck-smoothing":
            return _run_compare_neck_smoothing(args)
        if args.command == "extract-neck-pose":
            return _run_extract_neck_pose(args)
        if args.command == "extract-speech":
            return _run_extract_speech(args)
        if args.command == "build-fragments":
            return _run_build_fragments(args)
        if args.command == "build-split":
            return _run_build_split(args)
        config = _load_config(_project_path(args.config))
        if args.command == "discover":
            return _run_discover(args, config)
        if args.command == "validate":
            return _run_validate(config)
        return _run_download(args, config)
    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
