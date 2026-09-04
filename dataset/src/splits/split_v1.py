"""Deterministic source-video-grouped Train/Val/Test manifest splitting."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FEATURE_VERSION = "split_v1"
DEFAULT_SEED = 42
DEFAULT_RATIOS = {"train": 0.7, "test": 0.2, "val": 0.1}
_SPLIT_NAMES = ("train", "test", "val")
_OUTPUT_NAMES = ("train", "val", "test")
_RATIO_TOLERANCE = 1e-9


class SplitError(ValueError):
    """Raised when split input or output invariants are violated."""


def validate_ratios(train_ratio: float, test_ratio: float, val_ratio: float) -> dict[str, float]:
    """Validate and return ratios in the canonical train/test/val order."""
    ratios = {
        "train": float(train_ratio),
        "test": float(test_ratio),
        "val": float(val_ratio),
    }
    for name, ratio in ratios.items():
        if not math.isfinite(ratio) or ratio <= 0:
            raise SplitError(f"{name}_ratio must be a positive finite number")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=_RATIO_TOLERANCE):
        raise SplitError("train_ratio + test_ratio + val_ratio must equal 1.0")
    return ratios


def allocate_source_counts(
    source_count: int,
    train_ratio: float = DEFAULT_RATIOS["train"],
    test_ratio: float = DEFAULT_RATIOS["test"],
    val_ratio: float = DEFAULT_RATIOS["val"],
) -> dict[str, int]:
    """Allocate sources with the deterministic largest-remainder method.

    Equal fractional remainders are resolved in train, test, val order. Empty
    splits are allowed when there are too few source videos to represent every
    positive ratio.
    """
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
        raise SplitError("source_count must be a non-negative integer")
    ratios = validate_ratios(train_ratio, test_ratio, val_ratio)
    decimal_ratios = {name: Decimal(str(ratios[name])) for name in _SPLIT_NAMES}
    ratio_total = sum(decimal_ratios.values())
    quotas = {
        name: Decimal(source_count) * decimal_ratios[name] / ratio_total
        for name in _SPLIT_NAMES
    }
    counts = {name: int(quotas[name]) for name in _SPLIT_NAMES}
    remaining = source_count - sum(counts.values())
    priority = {name: index for index, name in enumerate(_SPLIT_NAMES)}
    ranked = sorted(
        _SPLIT_NAMES,
        key=lambda name: (-(quotas[name] - counts[name]), priority[name]),
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != source_count:
        raise SplitError("internal source allocation error")
    return counts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SplitError(f"input manifest not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"invalid JSON in {path} line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise SplitError(f"expected JSON object in {path} line {line_number}")
            records.append(record)
    return records


def _required_id(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"{context} has missing or invalid {field}")
    return value


def _duration(record: Mapping[str, Any], context: str) -> float:
    try:
        value = float(record["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"{context} has missing or invalid duration") from exc
    if not math.isfinite(value) or value < 0:
        raise SplitError(f"{context} has invalid duration")
    return value


def _sort_key(record: Mapping[str, Any]) -> tuple[str, str, float, str]:
    try:
        source_start = float(record.get("source_start", 0.0))
    except (TypeError, ValueError) as exc:
        raise SplitError(f"fragment {record.get('fragment_id')!r} has invalid source_start") from exc
    if not math.isfinite(source_start):
        raise SplitError(f"fragment {record.get('fragment_id')!r} has invalid source_start")
    return (
        str(record["source_video_id"]),
        str(record["clip_id"]),
        source_start,
        str(record["fragment_id"]),
    )


def _validate_manifests(
    fragments: Sequence[dict[str, Any]], clips: Sequence[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if not fragments:
        raise SplitError("fragments.jsonl contains no fragments")

    source_fragments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fragment_ids: set[str] = set()
    fragment_clip_sources: dict[str, str] = {}
    for index, record in enumerate(fragments, 1):
        context = f"fragment record {index}"
        fragment_id = _required_id(record, "fragment_id", context)
        source_id = _required_id(record, "source_video_id", f"fragment {fragment_id}")
        clip_id = _required_id(record, "clip_id", f"fragment {fragment_id}")
        _duration(record, f"fragment {fragment_id}")
        _sort_key(record)
        if fragment_id in fragment_ids:
            raise SplitError(f"duplicate fragment_id: {fragment_id}")
        fragment_ids.add(fragment_id)
        previous_source = fragment_clip_sources.setdefault(clip_id, source_id)
        if previous_source != source_id:
            raise SplitError(
                f"fragment clip/source mapping conflict for clip_id {clip_id}: "
                f"{previous_source!r} vs {source_id!r}"
            )
        source_fragments[source_id].append(record)

    clip_sources: dict[str, str] = {}
    for index, record in enumerate(clips, 1):
        context = f"clip record {index}"
        clip_id = _required_id(record, "clip_id", context)
        source_id = _required_id(record, "source_video_id", f"clip {clip_id}")
        if clip_id in clip_sources:
            raise SplitError(f"duplicate clip_id in clips.jsonl: {clip_id}")
        clip_sources[clip_id] = source_id
        if source_id not in source_fragments:
            raise SplitError(
                f"clip {clip_id} references source_video_id {source_id!r} with no fragments"
            )

    for clip_id, source_id in fragment_clip_sources.items():
        clip_source = clip_sources.get(clip_id)
        if clip_source is None:
            raise SplitError(f"fragment clip_id {clip_id} is missing from clips.jsonl")
        if clip_source != source_id:
            raise SplitError(
                f"clips.jsonl conflicts with fragments for clip_id {clip_id}: "
                f"{clip_source!r} vs {source_id!r}"
            )
    return dict(source_fragments), clip_sources


def _source_mapping(
    source_ids: Iterable[str], counts: Mapping[str, int], seed: int
) -> dict[str, str]:
    shuffled = sorted(source_ids)
    random.Random(seed).shuffle(shuffled)
    mapping: dict[str, str] = {}
    offset = 0
    for name in _SPLIT_NAMES:
        next_offset = offset + counts[name]
        for source_id in shuffled[offset:next_offset]:
            mapping[source_id] = name
        offset = next_offset
    return {source_id: mapping[source_id] for source_id in sorted(mapping)}


def _ratio(part: float | int, total: float | int) -> float:
    return float(part) / float(total) if total else 0.0


def _validate_assignment(
    fragments: Sequence[dict[str, Any]],
    split_records: Mapping[str, Sequence[dict[str, Any]]],
    source_to_split: Mapping[str, str],
    clip_sources: Mapping[str, str],
) -> None:
    source_sets = {
        name: {str(record["source_video_id"]) for record in split_records[name]}
        for name in _OUTPUT_NAMES
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise SplitError(f"source leakage between {left} and {right}: {sorted(overlap)}")

    expected_sources = {str(record["source_video_id"]) for record in fragments}
    assigned_sources = set().union(*source_sets.values())
    if assigned_sources != expected_sources or set(source_to_split) != expected_sources:
        raise SplitError("not every source_video_id was assigned exactly once")

    expected_ids = [str(record["fragment_id"]) for record in fragments]
    output_ids = {
        name: [str(record["fragment_id"]) for record in split_records[name]]
        for name in _OUTPUT_NAMES
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(output_ids[left]) & set(output_ids[right]):
            raise SplitError(f"fragment leakage between {left} and {right}")
    flattened = [item for name in _OUTPUT_NAMES for item in output_ids[name]]
    if len(flattened) != len(set(flattened)):
        raise SplitError("fragment duplication detected in split output")
    if set(flattened) != set(expected_ids) or len(flattened) != len(expected_ids):
        raise SplitError("fragment omission detected in split output")

    clip_splits: dict[str, set[str]] = defaultdict(set)
    for name in _OUTPUT_NAMES:
        for record in split_records[name]:
            clip_splits[str(record["clip_id"])].add(name)
    if any(len(names) != 1 for names in clip_splits.values()):
        raise SplitError("a clip has fragments in multiple splits")
    for clip_id, source_id in clip_sources.items():
        expected_split = source_to_split[source_id]
        if clip_id in clip_splits and clip_splits[clip_id] != {expected_split}:
            raise SplitError(f"clip-level split mismatch for clip_id {clip_id}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    path.write_text(content, encoding="utf-8")


def build_split_dataset(
    input_root: Path,
    output_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_RATIOS["train"],
    test_ratio: float = DEFAULT_RATIOS["test"],
    val_ratio: float = DEFAULT_RATIOS["val"],
    force: bool = False,
) -> dict[str, Any]:
    """Build reference-only split manifests grouped by ``source_video_id``."""
    ratios = validate_ratios(train_ratio, test_ratio, val_ratio)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    input_root = Path(input_root)
    output_root = Path(output_root)
    if output_root.exists() and not force:
        raise SplitError(f"output already exists: {output_root}; use --force to overwrite")
    fragments_path = input_root / "fragments.jsonl"
    clips_path = input_root / "clips.jsonl"
    fragments = _load_jsonl(fragments_path)
    clips = _load_jsonl(clips_path)
    source_fragments, clip_sources = _validate_manifests(fragments, clips)

    counts = allocate_source_counts(
        len(source_fragments), ratios["train"], ratios["test"], ratios["val"]
    )
    source_to_split = _source_mapping(source_fragments, counts, seed)
    split_records: dict[str, list[dict[str, Any]]] = {name: [] for name in _OUTPUT_NAMES}
    for record in fragments:
        name = source_to_split[str(record["source_video_id"])]
        split_records[name].append({**record, "split": name})
    for records in split_records.values():
        records.sort(key=_sort_key)

    _validate_assignment(fragments, split_records, source_to_split, clip_sources)

    source_counts = {
        name: sum(split_name == name for split_name in source_to_split.values())
        for name in _OUTPUT_NAMES
    }
    clip_counts = {
        name: sum(source_to_split[source_id] == name for source_id in clip_sources.values())
        for name in _OUTPUT_NAMES
    }
    fragment_counts = {name: len(split_records[name]) for name in _OUTPUT_NAMES}
    durations = {
        name: sum(_duration(record, f"fragment {record['fragment_id']}") for record in split_records[name])
        for name in _OUTPUT_NAMES
    }
    total_duration = sum(durations.values())
    total_clips = len(clip_sources)
    total_fragments = len(fragments)
    total_sources = len(source_to_split)

    source_split: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "ratios": ratios,
        "input_manifest": str(fragments_path.resolve()),
        "input_fragment_count": total_fragments,
        "input_source_video_count": total_sources,
        "source_video_count": total_sources,
        "source_to_split": source_to_split,
    }
    for name in _OUTPUT_NAMES:
        source_split[f"{name}_sources"] = sorted(
            source_id for source_id, split_name in source_to_split.items() if split_name == name
        )

    summary: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "ratios": ratios,
        "input_manifest": str(fragments_path.resolve()),
        "input_clips_manifest": str(clips_path.resolve()),
        "input_fragment_count": total_fragments,
        "input_source_video_count": total_sources,
        "source_videos": {"total": total_sources, **source_counts},
        "clips": {"total": total_clips, **clip_counts},
        "fragments": {"total": total_fragments, **fragment_counts},
        "duration_seconds": {"total": total_duration, **durations},
        "duration_hours": {
            "total": total_duration / 3600.0,
            **{name: durations[name] / 3600.0 for name in _OUTPUT_NAMES},
        },
        "actual_ratios": {
            **{f"source_{name}": _ratio(source_counts[name], total_sources) for name in _OUTPUT_NAMES},
            **{f"fragment_{name}": _ratio(fragment_counts[name], total_fragments) for name in _OUTPUT_NAMES},
            **{f"duration_{name}": _ratio(durations[name], total_duration) for name in _OUTPUT_NAMES},
        },
        "leakage_check": {
            "source_overlap": False,
            "fragment_overlap": False,
            "missing_fragments": 0,
        },
    }

    if output_root.exists() and not force:
        raise SplitError(f"output already exists: {output_root}; use --force to overwrite")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        for name in _OUTPUT_NAMES:
            _write_jsonl(staging / f"{name}.jsonl", split_records[name])
        _write_json(staging / "source_split.json", source_split)
        _write_json(staging / "summary.json", summary)
        if output_root.exists():
            if output_root.is_dir():
                shutil.rmtree(output_root)
            else:
                output_root.unlink()
        staging.replace(output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return summary
