"""Command-line interface for YouTube dataset collection."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.crawler.discover import discover_videos_with_stats
from src.crawler.downloader import DownloadError, download_videos
from src.crawler.storage import load_jsonl, save_jsonl, save_preview, save_validation_preview
from src.crawler.validation import stderr_progress, validate_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "youtube.yaml"


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
    parser = argparse.ArgumentParser(description="Collect YouTube dataset videos")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="search and save candidate metadata")
    discover.add_argument("--query", help="override the configured search query")
    discover.add_argument("--limit", type=int, help="override the configured result limit")

    subparsers.add_parser("validate", help="validate discovered candidates using full metadata")

    download = subparsers.add_parser("download", help="download validated videos")
    download.add_argument("--limit", type=int, help="download only the first N validated videos")

    clean = subparsers.add_parser(
        "clean", help="strictly filter whole videos and group presenters"
    )
    clean.add_argument("--input", required=True, help="source video directory")
    clean.add_argument("--output", required=True, help="new cleaned dataset directory")
    clean.add_argument("--cleaning-config", help="cleaning YAML configuration")
    clean.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    clean.add_argument("--limit", type=int, help="process only the first N videos")
    clean.add_argument("--force", action="store_true", help="atomically replace existing output")
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
    # Keep Torch, Ultralytics, ONNX Runtime, and InsightFace out of crawler commands.
    from src.processing.clean_videos import DEFAULT_CONFIG as DEFAULT_CLEANING_CONFIG
    from src.processing.clean_videos import clean_videos, load_config

    cleaning_config = load_config(
        _project_path(args.cleaning_config)
        if args.cleaning_config
        else DEFAULT_CLEANING_CONFIG
    )
    result = clean_videos(
        _project_path(args.input),
        _project_path(args.output),
        cleaning_config,
        device=args.device,
        limit=args.limit,
        force=args.force,
    )
    print(f"Input videos: {result['input_videos']}")
    print(f"Accepted videos: {result['accepted_videos']}")
    print(f"Truncated videos: {result['truncated_videos']}")
    print(f"Rejected without usable prefix: {result['rejected_without_usable_prefix']}")
    print(f"Anchors: {result['anchors']}")
    print(f"Output: {_project_path(args.output)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "clean":
            return _run_clean(args)
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
