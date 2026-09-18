"""Create diagnostic sentence-like fragments from one Qwen timestamp document."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IDEAL_MIN_SEC = 2.0
IDEAL_MAX_SEC = 8.0
NORMAL_MAX_SEC = 10.0
SEMANTIC_MAX_SEC = 12.0
MIN_FRAGMENT_SEC = 1.5
PAUSE_STRONG_SEC = 0.8
PAUSE_MEDIUM_SEC = 0.5
PAUSE_MIN_SEC = 0.3
_STRONG_PUNCTUATION = set("。！？!?")
_WEAK_PUNCTUATION = set("，,、；;：:")
_INCOMPLETE_SUFFIXES = (
    "因为",
    "但是",
    "如果",
    "所以",
    "虽然",
    "尽管",
    "由于",
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
_STRIPPED_ENDINGS = " \t\r\n，,、；;：:。！？!?…—-\"'“”‘’（）()《》〈〉【】[]"


class FragmentSplitDebugError(RuntimeError):
    """Raised when one diagnostic fragment document cannot be produced safely."""


@dataclass
class TimedUnit:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Span:
    first: int
    last: int


def _separator_only(value: str) -> bool:
    return all(
        character.isspace()
        or unicodedata.category(character).startswith(("P", "S", "Z"))
        for character in value
    )


def _load_units(transcript: Path, video: Path) -> tuple[dict[str, Any], list[TimedUnit]]:
    try:
        document = json.loads(transcript.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentSplitDebugError(f"could not read transcript: {transcript}") from exc
    if document.get("version") != "qwen_asr_v1":
        raise FragmentSplitDebugError(f"unsupported transcript version: {transcript}")
    if document.get("timeline") != "original_audio_video" or document.get("time_unit") != "second":
        raise FragmentSplitDebugError("transcript must use the original audio/video second timeline")
    audio_name = document.get("audio")
    if not isinstance(audio_name, str) or Path(audio_name).stem != video.stem:
        raise FragmentSplitDebugError("transcript audio stem does not match the video")
    full_text = document.get("text")
    timestamps = document.get("timestamps")
    if not isinstance(full_text, str) or not full_text.strip():
        raise FragmentSplitDebugError("transcript text is empty")
    if not isinstance(timestamps, list) or not timestamps:
        raise FragmentSplitDebugError("transcript timestamps are empty")

    units: list[TimedUnit] = []
    cursor = 0
    previous_end = 0.0
    for index, item in enumerate(timestamps, 1):
        try:
            token = str(item["text"])
            start = float(item["start_time"])
            end = float(item["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FragmentSplitDebugError(
                f"invalid timestamp at index {index}"
            ) from exc
        if (
            not token
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end - 1e-6
            or end < start
        ):
            raise FragmentSplitDebugError(
                f"invalid timestamp values at index {index}"
            )
        position = full_text.find(token, cursor)
        if position < 0:
            raise FragmentSplitDebugError(
                f"timestamp token cannot be aligned to transcript text at index {index}: {token!r}"
            )
        skipped = full_text[cursor:position]
        if not _separator_only(skipped):
            raise FragmentSplitDebugError(
                f"non-punctuation text is missing from timestamps at index {index}: {skipped!r}"
            )
        if units:
            units[-1].text += skipped
            display_text = token
        else:
            display_text = skipped + token
        units.append(TimedUnit(text=display_text, start=start, end=end))
        cursor = position + len(token)
        previous_end = end

    trailing = full_text[cursor:]
    if not _separator_only(trailing):
        raise FragmentSplitDebugError(
            f"non-punctuation transcript tail is missing from timestamps: {trailing!r}"
        )
    units[-1].text += trailing
    return document, units


def _span_text(units: list[TimedUnit], span: Span) -> str:
    return "".join(unit.text for unit in units[span.first : span.last + 1]).strip()


def _duration(units: list[TimedUnit], span: Span) -> float:
    return units[span.last].end - units[span.first].start


def _is_incomplete(text: str) -> bool:
    core = text.rstrip(_STRIPPED_ENDINGS)
    return any(core.endswith(suffix) for suffix in _INCOMPLETE_SUFFIXES)


def _boundary_quality(units: list[TimedUnit], index: int) -> float:
    text = units[index].text.rstrip()
    if text and text[-1] in _STRONG_PUNCTUATION:
        return 4.0
    pause = (
        units[index + 1].start - units[index].end
        if index + 1 < len(units)
        else 0.0
    )
    if pause >= PAUSE_STRONG_SEC:
        return 3.5
    if text and text[-1] in _WEAK_PUNCTUATION:
        return 3.0
    if pause >= PAUSE_MEDIUM_SEC:
        return 2.5
    if pause >= PAUSE_MIN_SEC:
        return 2.0
    return 0.0


def _candidates(
    units: list[TimedUnit], first: int, minimum: float, maximum: float
) -> list[tuple[int, float, float]]:
    values: list[tuple[int, float, float]] = []
    start = units[first].start
    for index in range(first, len(units)):
        duration = units[index].end - start
        if duration > maximum + 1e-9:
            break
        if duration < minimum - 1e-9:
            continue
        text = _span_text(units, Span(first, index))
        if _is_incomplete(text):
            continue
        quality = _boundary_quality(units, index)
        if quality > 0.0:
            values.append((index, quality, duration))
    return values


def _choose_last(units: list[TimedUnit], first: int) -> int:
    remaining_duration = units[-1].end - units[first].start
    if remaining_duration <= SEMANTIC_MAX_SEC:
        return len(units) - 1

    ideal = _candidates(units, first, IDEAL_MIN_SEC, IDEAL_MAX_SEC)
    ideal_strong = [candidate for candidate in ideal if candidate[1] >= 4.0]
    if ideal_strong:
        return min(ideal_strong, key=lambda candidate: candidate[0])[0]

    semantic_extension = _candidates(
        units, first, IDEAL_MAX_SEC, SEMANTIC_MAX_SEC
    )
    extended_strong = [
        candidate for candidate in semantic_extension if candidate[1] >= 4.0
    ]
    if extended_strong:
        return min(extended_strong, key=lambda candidate: candidate[0])[0]

    # A comma, enumeration comma, or semicolon is not enough reason to stop
    # near the ideal duration. Such weaker boundaries are considered only now,
    # after proving that no sentence-ending boundary exists within 12 seconds.
    natural = ideal + semantic_extension
    if natural:
        return max(natural, key=lambda candidate: (candidate[1], candidate[2]))[0]

    start = units[first].start
    fallback = first
    fallback_complete: int | None = None
    for index in range(first, len(units)):
        duration = units[index].end - start
        if duration > SEMANTIC_MAX_SEC + 1e-9:
            break
        fallback = index
        if not _is_incomplete(_span_text(units, Span(first, index))):
            fallback_complete = index
    return fallback_complete if fallback_complete is not None else fallback


def _initial_spans(units: list[TimedUnit]) -> list[Span]:
    spans: list[Span] = []
    first = 0
    while first < len(units):
        last = _choose_last(units, first)
        if last < first:
            raise AssertionError("fragment splitter did not advance")
        spans.append(Span(first, last))
        first = last + 1
    return spans


def _merge_short_spans(units: list[TimedUnit], spans: list[Span]) -> list[Span]:
    spans = list(spans)
    index = 0
    while index < len(spans) and len(spans) > 1:
        if _duration(units, spans[index]) >= MIN_FRAGMENT_SEC:
            index += 1
            continue
        if index == 0:
            spans[1] = Span(spans[0].first, spans[1].last)
            spans.pop(0)
            continue
        if index == len(spans) - 1:
            spans[index - 1] = Span(spans[index - 1].first, spans[index].last)
            spans.pop(index)
            index = max(0, index - 1)
            continue

        left_duration = units[spans[index].last].end - units[spans[index - 1].first].start
        right_duration = units[spans[index + 1].last].end - units[spans[index].first].start
        left_penalty = (
            left_duration > SEMANTIC_MAX_SEC,
            abs(left_duration - IDEAL_MAX_SEC),
        )
        right_penalty = (
            right_duration > SEMANTIC_MAX_SEC,
            abs(right_duration - IDEAL_MAX_SEC),
        )
        if left_penalty <= right_penalty:
            spans[index - 1] = Span(spans[index - 1].first, spans[index].last)
            spans.pop(index)
            index = max(0, index - 1)
        else:
            spans[index + 1] = Span(spans[index].first, spans[index + 1].last)
            spans.pop(index)
    return spans


def split_fragments(video: Path, transcript: Path) -> dict[str, Any]:
    if not video.is_file():
        raise FragmentSplitDebugError(f"video not found: {video}")
    if not transcript.is_file():
        raise FragmentSplitDebugError(f"transcript not found: {transcript}")
    source, units = _load_units(transcript, video)
    spans = _merge_short_spans(units, _initial_spans(units))
    fragments: list[dict[str, Any]] = []
    for number, span in enumerate(spans, 1):
        start = units[span.first].start
        end = units[span.last].end
        fragments.append(
            {
                "id": f"{number:03d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(end - start, 6),
                "text": _span_text(units, span),
            }
        )
    return {
        "version": "fragment_split_debug_v1",
        "video": video.name,
        "transcript": transcript.name,
        "timeline": "original_audio_video",
        "time_unit": "second",
        "source_review_status": (source.get("review") or {}).get("status"),
        "review": {"status": "pending"},
        "rules": {
            "ideal_duration_sec": [IDEAL_MIN_SEC, IDEAL_MAX_SEC],
            "normal_max_duration_sec": NORMAL_MAX_SEC,
            "semantic_max_duration_sec": SEMANTIC_MAX_SEC,
            "minimum_standalone_duration_sec": MIN_FRAGMENT_SEC,
            "pause_thresholds_sec": [PAUSE_MIN_SEC, PAUSE_MEDIUM_SEC, PAUSE_STRONG_SEC],
            "boundary_priority": (
                "semantic completion through 12 seconds, then strong punctuation, "
                "long pause, weak punctuation, shorter pause"
            ),
            "avoid_incomplete_suffixes": list(_INCOMPLETE_SUFFIXES),
        },
        "fragment_count": len(fragments),
        "fragments": fragments,
    }


def _write_json_atomic(path: Path, document: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FragmentSplitDebugError(
            f"output already exists (use --force to replace it): {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(document, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create diagnostic fragments from one Qwen timestamp document"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="output fragments.json")
    parser.add_argument("--force", action="store_true", help="replace existing output")
    args = parser.parse_args(argv)
    try:
        document = split_fragments(
            args.video.expanduser().resolve(),
            args.transcript.expanduser().resolve(),
        )
        _write_json_atomic(args.output.expanduser().resolve(), document, args.force)
    except (OSError, ValueError, FragmentSplitDebugError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    durations = [fragment["duration_sec"] for fragment in document["fragments"]]
    print(f"Fragments: {document['fragment_count']}")
    print(f"Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
