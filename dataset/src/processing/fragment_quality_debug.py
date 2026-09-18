"""Read-only quality diagnostics for generated fragment boundary documents."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "fragment_split_debug"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo" / "fragment_quality_debug"
)
EXPECTED_VERSION = "fragment_split_debug_v1"
EXTREMELY_SHORT_MAX_CHARS = 5
FAST_TEXT_MIN_CHARS = 15
FAST_TEXT_MIN_CHARS_PER_SEC = 7.0
SLOW_TEXT_MAX_CHARS = 12
SLOW_TEXT_MIN_DURATION_SEC = 6.0
VIDEO_FEW_FRAGMENTS = 5
VIDEO_MANY_FRAGMENTS = 30
UNFINISHED_PUNCTUATION = set("，,、；;：:")
INCOMPLETE_SUFFIXES = (
    "因为",
    "但是",
    "如果",
    "所以",
    "虽然",
    "由于",
    "尽管",
    "而且",
    "并且",
    "以及",
    "或者",
    "不仅",
    "只要",
    "即使",
    "为了",
    "那么",
)
STRIPPED_ENDINGS = " \t\r\n，,、；;：:。！？!?…—-\"'“”‘’（）()《》〈〉【】[]"
MOJIBAKE_MARKERS = ("�", "锟斤拷", "烫烫烫", "屯屯屯", "Ã", "Â", "â€")
REASON_ORDER = (
    "empty_text",
    "extremely_short_text",
    "obvious_repetition",
    "invalid_character_or_mojibake",
    "unfinished_punctuation",
    "unfinished_connector",
    "text_too_long_for_duration",
    "text_too_short_for_duration",
    "video_fragment_count_too_low",
    "video_fragment_count_too_high",
)


class FragmentQualityDebugError(RuntimeError):
    """Raised when the diagnostic input is invalid or cannot be published."""


def _lexical_text(text: str) -> str:
    return "".join(
        character
        for character in text
        if unicodedata.category(character).startswith(("L", "N"))
    )


def _has_obvious_repetition(text: str) -> bool:
    lexical = _lexical_text(text)
    if re.search(r"(.)\1{3,}", lexical):
        return True
    length = len(lexical)
    for unit_length in range(3, min(20, length // 2) + 1):
        repeated_length = unit_length * 2
        if repeated_length < max(8, math.ceil(length * 0.5)):
            continue
        for start in range(0, length - repeated_length + 1):
            first = lexical[start : start + unit_length]
            second = lexical[start + unit_length : start + repeated_length]
            if first == second:
                return True
    return False


def _has_invalid_character(text: str) -> bool:
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return True
    return any(
        unicodedata.category(character) in {"Cc", "Cs", "Co", "Cn"}
        and not character.isspace()
        for character in text
    )


def _fragment_reasons(text: str, duration: float) -> list[str]:
    stripped = text.strip()
    lexical_count = len(_lexical_text(stripped))
    reasons: list[str] = []
    if not stripped:
        reasons.append("empty_text")
    if 0 < lexical_count <= EXTREMELY_SHORT_MAX_CHARS:
        reasons.append("extremely_short_text")
    if stripped and _has_obvious_repetition(stripped):
        reasons.append("obvious_repetition")
    if stripped and _has_invalid_character(stripped):
        reasons.append("invalid_character_or_mojibake")
    if stripped and stripped[-1] in UNFINISHED_PUNCTUATION:
        reasons.append("unfinished_punctuation")
    core = stripped.rstrip(STRIPPED_ENDINGS)
    if core and any(core.endswith(suffix) for suffix in INCOMPLETE_SUFFIXES):
        reasons.append("unfinished_connector")
    if (
        duration > 0.0
        and lexical_count >= FAST_TEXT_MIN_CHARS
        and lexical_count / duration > FAST_TEXT_MIN_CHARS_PER_SEC
    ):
        reasons.append("text_too_long_for_duration")
    if lexical_count <= SLOW_TEXT_MAX_CHARS and duration >= SLOW_TEXT_MIN_DURATION_SEC:
        reasons.append("text_too_short_for_duration")
    return reasons


def _load_document(path: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentQualityDebugError(f"could not read fragment document: {path}") from exc
    if document.get("version") != EXPECTED_VERSION:
        raise FragmentQualityDebugError(f"unsupported fragment version: {path}")
    video = document.get("video")
    fragments = document.get("fragments")
    if not isinstance(video, str) or not video or Path(video).name != video:
        raise FragmentQualityDebugError(f"invalid video name: {path}")
    if not isinstance(fragments, list):
        raise FragmentQualityDebugError(f"fragments must be a list: {path}")
    if document.get("fragment_count") != len(fragments):
        raise FragmentQualityDebugError(f"fragment_count mismatch: {path}")

    validated: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    previous_end = 0.0
    for index, fragment in enumerate(fragments, 1):
        try:
            fragment_id = str(fragment["id"])
            text = fragment["text"]
            start = float(fragment["start_sec"])
            end = float(fragment["end_sec"])
            duration = float(fragment["duration_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FragmentQualityDebugError(
                f"invalid fragment {index} in {path}"
            ) from exc
        if not isinstance(text, str) or not fragment_id or fragment_id in identifiers:
            raise FragmentQualityDebugError(
                f"invalid id or text at fragment {index} in {path}"
            )
        if (
            not all(math.isfinite(value) for value in (start, end, duration))
            or start < previous_end - 1e-6
            or end < start
            or duration < 0.0
            or abs(duration - (end - start)) > 1e-5
        ):
            raise FragmentQualityDebugError(
                f"invalid timing at fragment {index} in {path}"
            )
        identifiers.add(fragment_id)
        previous_end = end
        validated.append(
            {
                "id": fragment_id,
                "text": text,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration,
            }
        )
    return video, validated


def _discover_documents(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FragmentQualityDebugError(f"input directory not found: {root}")
    paths = sorted(root.glob("*/fragments.json"), key=lambda path: path.parent.name)
    if not paths:
        raise FragmentQualityDebugError(f"no fragments.json documents found under: {root}")
    return paths


def analyze(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_videos: set[str] = set()
    video_flags: list[dict[str, Any]] = []

    for path in _discover_documents(root):
        video, fragments = _load_document(path)
        if video in seen_videos:
            raise FragmentQualityDebugError(f"duplicate video in fragment inputs: {video}")
        seen_videos.add(video)
        video_reasons: list[str] = []
        if len(fragments) < VIDEO_FEW_FRAGMENTS:
            video_reasons.append("video_fragment_count_too_low")
        if len(fragments) > VIDEO_MANY_FRAGMENTS:
            video_reasons.append("video_fragment_count_too_high")
        if video_reasons:
            video_flags.append(
                {
                    "video": video,
                    "fragment_count": len(fragments),
                    "reasons": video_reasons,
                }
            )

        for fragment in fragments:
            reasons = _fragment_reasons(fragment["text"], fragment["duration_sec"])
            reasons.extend(video_reasons)
            reasons = [reason for reason in REASON_ORDER if reason in reasons]
            reason_counts.update(reasons)
            records.append(
                {
                    "video": video,
                    "fragment_id": fragment["id"],
                    "text": fragment["text"],
                    "start_sec": fragment["start_sec"],
                    "end_sec": fragment["end_sec"],
                    "duration_sec": fragment["duration_sec"],
                    "status": "suspected" if reasons else "ok",
                    "reasons": reasons,
                }
            )

    suspected_count = sum(record["status"] == "suspected" for record in records)
    manifest = {
        "version": "fragment_quality_debug_v1",
        "complete": True,
        "input": str(root),
        "video_count": len(seen_videos),
        "fragment_count": len(records),
        "ok_count": len(records) - suspected_count,
        "suspected_count": suspected_count,
        "reason_counts": {reason: reason_counts.get(reason, 0) for reason in REASON_ORDER},
        "video_flags": video_flags,
        "thresholds": {
            "extremely_short_max_lexical_chars": EXTREMELY_SHORT_MAX_CHARS,
            "fast_text_min_lexical_chars": FAST_TEXT_MIN_CHARS,
            "fast_text_min_chars_per_sec_exclusive": FAST_TEXT_MIN_CHARS_PER_SEC,
            "slow_text_max_lexical_chars": SLOW_TEXT_MAX_CHARS,
            "slow_text_min_duration_sec": SLOW_TEXT_MIN_DURATION_SEC,
            "video_fragment_count_too_low_exclusive": VIDEO_FEW_FRAGMENTS,
            "video_fragment_count_too_high_exclusive": VIDEO_MANY_FRAGMENTS,
            "unfinished_punctuation": "".join(sorted(UNFINISHED_PUNCTUATION)),
            "incomplete_suffixes": list(INCOMPLETE_SUFFIXES),
        },
    }
    return records, manifest


def _publish(
    output: Path, records: list[dict[str, Any]], manifest: dict[str, Any], force: bool
) -> None:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise FragmentQualityDebugError(
            f"output already exists (use --force to replace it): {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    backup: Path | None = None
    try:
        with (staging / "results.jsonl").open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        with (staging / "manifest.json").open("w", encoding="utf-8") as target:
            json.dump(manifest, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if output.exists():
            backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
            os.replace(output, backup)
        os.replace(staging, output)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose text, semantic, duration, and video-count fragment anomalies"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    try:
        records, manifest = analyze(args.input.expanduser().resolve())
        _publish(args.output, records, manifest, args.force)
    except (OSError, ValueError, FragmentQualityDebugError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Videos: {manifest['video_count']}")
    print(f"Fragments: {manifest['fragment_count']}")
    print(f"OK: {manifest['ok_count']}")
    print(f"Suspected: {manifest['suspected_count']}")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
