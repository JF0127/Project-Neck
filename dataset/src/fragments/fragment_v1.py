"""Fragment V1.2.1 soft-threshold natural-boundary lookahead construction.

This stage only groups Speech V1 segments and slices existing PCM and Neck Pose
V1 ground truth. It never runs ASR, resamples audio, filters motion, or changes
the neck reference.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FEATURE_VERSION = "fragment_v1_2_1"
DEFAULT_MIN_DURATION = 2.0
DEFAULT_SOFT_MAX_DURATION = 20.0
DEFAULT_ABSOLUTE_MAX_DURATION = 30.0
# Backward-compatible names; the effective safety cap is now the absolute max.
DEFAULT_HARD_MAX_DURATION = DEFAULT_SOFT_MAX_DURATION
DEFAULT_MAX_DURATION = DEFAULT_ABSOLUTE_MAX_DURATION
DEFAULT_STRONG_PAUSE = 0.45
# Backward-compatible exported name; there is only one grouping pause setting.
DEFAULT_SENTENCE_PAUSE = DEFAULT_STRONG_PAUSE
DEFAULT_LONG_SPLIT_PAUSE = 0.35
DEFAULT_MAX_CONTEXT_PADDING = 0.25
OVERSIZED_SPLIT_TARGET = 18.5
ABSOLUTE_FALLBACK_TARGET = 24.5
# A short utterance must not be joined across a clearly unrelated silence.
MAX_SHORT_MERGE_PAUSE = 2.0
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_ROUNDING_TOLERANCE_SAMPLES = 2
OVERLAP_WARNING_SECONDS = 0.05
OVERLAP_ERROR_SECONDS = 0.50
NECK_QUALITY_THRESHOLD = 0.90

HARD_END_PUNCTUATION = frozenset("。！？!?")
CHINESE_SOFT_PUNCTUATION = frozenset("，；")
ENGLISH_SOFT_PUNCTUATION = frozenset(",;")
COLON_PUNCTUATION = frozenset("：:")
TRAILING_CLOSERS = frozenset("\"'”’」』】）》〉〕）)]}")
REQUIRED_FRAGMENT_FILES = (
    "audio.wav",
    "neck_rpy.npy",
    "neck_timestamps.npy",
    "neck_valid.npy",
    "transcript.json",
    "metadata.json",
)


class FragmentError(RuntimeError):
    """Raised for an invalid input or unsafe multimodal alignment."""


@dataclass(frozen=True)
class FragmentConfig:
    min_duration: float = DEFAULT_MIN_DURATION
    soft_max_duration: float = DEFAULT_SOFT_MAX_DURATION
    absolute_max_duration: float = DEFAULT_ABSOLUTE_MAX_DURATION
    strong_pause: float = DEFAULT_STRONG_PAUSE
    long_split_pause: float = DEFAULT_LONG_SPLIT_PAUSE
    max_context_padding: float = DEFAULT_MAX_CONTEXT_PADDING

    def validate(self) -> None:
        values = (
            self.min_duration,
            self.soft_max_duration,
            self.absolute_max_duration,
            self.strong_pause,
            self.long_split_pause,
            self.max_context_padding,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Fragment V1.2.1 thresholds must be finite and non-negative")
        if not 0 < self.min_duration < self.soft_max_duration < self.absolute_max_duration:
            raise ValueError(
                "duration thresholds must satisfy 0 < min < soft < absolute max"
            )

    @property
    def max_duration(self) -> float:
        """Compatibility alias for the absolute 30 s safety limit."""
        return self.absolute_max_duration


@dataclass
class UtteranceUnit:
    """One atomic Whisper segment whose exact core is defined by its valid words."""

    start: float
    end: float
    text: str
    words: list[dict[str, Any]]
    segment_id: int
    segment_start: float | None = None
    segment_end: float | None = None
    terminal_punctuation: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


# Compatibility for code importing the previous internal type name.
SentenceUnit = UtteranceUnit


@dataclass
class Fragment:
    words: list[dict[str, Any]]
    segment_ids: tuple[int, ...]
    core_start: float
    core_end: float
    start_reason: str
    end_reason: str
    split_reason: str
    terminal_punctuation: bool = False
    short_merged: bool = False
    crossed_soft_threshold: bool = False
    source_start: float | None = None
    source_end: float | None = None
    duration_exception: bool = False
    duration_exception_reason: str | None = None
    alignment_warnings: list[str] = field(default_factory=list)

    @property
    def speech_duration(self) -> float:
        return self.core_end - self.core_start

    @property
    def duration(self) -> float:
        start = self.source_start if self.source_start is not None else self.core_start
        end = self.source_end if self.source_end is not None else self.core_end
        return end - start

    @property
    def segment_count(self) -> int:
        return len(set(self.segment_ids))

    @property
    def natural_boundary(self) -> bool:
        return self.end_reason in {"strong_pause", "clip_end"}

    @property
    def text(self) -> str:
        return _text_from_words(self.words)


def _text_from_words(words: Sequence[dict[str, Any]]) -> str:
    # Preserve ASR word content and only remove formatting whitespace at the edges.
    return "".join(str(word["text"]) for word in words).strip()


def _ends_with(text: str, punctuation: frozenset[str]) -> bool:
    value = text.rstrip()
    while value and value[-1] in TRAILING_CLOSERS:
        value = value[:-1].rstrip()
    return bool(value and value[-1] in punctuation)


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validated_segment_words(
    raw_words: Any,
    segment_id: int,
    previous_start: float,
    previous_end: float,
) -> tuple[list[dict[str, Any]], float, float]:
    if not isinstance(raw_words, list):
        raise FragmentError(f"Whisper segment {segment_id} words must be a list")
    words: list[dict[str, Any]] = []
    for word_index, raw in enumerate(raw_words):
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise FragmentError(f"segment {segment_id} word {word_index} has invalid text")
        try:
            start, end = float(raw["start"]), float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FragmentError(
                f"segment {segment_id} word {word_index} has invalid timestamp"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise FragmentError(
                f"segment {segment_id} word {word_index} has invalid interval"
            )
        if start < previous_start - 1e-9 or end < previous_end - 1e-9:
            raise FragmentError(
                f"Whisper segment word timeline is not monotonic at segment {segment_id}"
            )
        item: dict[str, Any] = {
            "text": raw["text"],
            "start": start,
            "end": end,
            "segment_id": segment_id,
        }
        if raw.get("probability") is not None:
            probability = float(raw["probability"])
            if math.isfinite(probability):
                item["probability"] = probability
        words.append(item)
        previous_start, previous_end = start, end
    return words, previous_start, previous_end


def build_utterance_units(transcript: dict[str, Any]) -> list[UtteranceUnit]:
    """Build one minimum safe unit per non-empty Whisper segment."""
    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list):
        raise FragmentError("Speech V1 transcript.segments must be a list")
    units: list[UtteranceUnit] = []
    previous_start = previous_end = -math.inf
    seen_ids: set[int] = set()
    for fallback_id, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            raise FragmentError(f"Whisper segment {fallback_id} is not an object")
        try:
            segment_id = int(segment.get("id", fallback_id))
        except (TypeError, ValueError) as exc:
            raise FragmentError(f"Whisper segment {fallback_id} has invalid id") from exc
        if segment_id in seen_ids:
            raise FragmentError(f"duplicate Whisper segment id {segment_id}")
        seen_ids.add(segment_id)
        words, previous_start, previous_end = _validated_segment_words(
            segment.get("words"), segment_id, previous_start, previous_end
        )
        # Empty ASR segments have no safe timeline and therefore cannot form fragments.
        if not words:
            continue
        units.append(
            UtteranceUnit(
                start=float(words[0]["start"]),
                end=float(words[-1]["end"]),
                text=_text_from_words(words),
                words=words,
                segment_id=segment_id,
                segment_start=_optional_finite_float(segment.get("start")),
                segment_end=_optional_finite_float(segment.get("end")),
                terminal_punctuation=(
                    _ends_with(str(segment.get("text", "")), HARD_END_PUNCTUATION)
                    or _ends_with(str(words[-1]["text"]), HARD_END_PUNCTUATION)
                ),
            )
        )
    if not units:
        raise FragmentError("Speech V1 transcript contains no timed segment words")
    return units


def build_sentence_units(
    transcript: dict[str, Any], sentence_pause: float = DEFAULT_SENTENCE_PAUSE
) -> list[UtteranceUnit]:
    """Compatibility wrapper; V1.2.1 units are Whisper segments, not sentences."""
    if not math.isfinite(sentence_pause) or sentence_pause < 0:
        raise ValueError("sentence_pause must be finite and non-negative")
    return build_utterance_units(transcript)


def _candidate_kind(
    words: Sequence[dict[str, Any]], boundary: int, pause_threshold: float
) -> tuple[int, str, float] | None:
    """Classify an oversized-segment internal word boundary."""
    previous = words[boundary - 1]
    following = words[boundary]
    pause = max(0.0, float(following["start"] - previous["end"]))
    if pause >= pause_threshold:
        return 0, "oversized_segment_pause", pause
    text = str(previous["text"])
    if _ends_with(
        text,
        HARD_END_PUNCTUATION
        | CHINESE_SOFT_PUNCTUATION
        | ENGLISH_SOFT_PUNCTUATION
        | COLON_PUNCTUATION,
    ):
        return 1, "oversized_segment_punctuation", pause
    return None


def _choose_long_boundary(
    words: Sequence[dict[str, Any]], config: FragmentConfig
) -> tuple[int, str]:
    start = float(words[0]["start"])
    total_end = float(words[-1]["end"])
    candidates: list[tuple[tuple[float, ...], int, str]] = []
    for boundary in range(1, len(words)):
        left_duration = float(words[boundary - 1]["end"] - start)
        if left_duration > config.max_duration + 1e-9:
            continue
        classified = _candidate_kind(words, boundary, config.long_split_pause)
        if classified is None:
            continue
        rank, reason, pause = classified
        remaining_duration = float(total_end - words[boundary]["start"])
        # Avoid a short side whenever the boundary would finish this split in two pieces.
        short_penalty = float(
            left_duration < config.min_duration
            or (remaining_duration <= config.max_duration and remaining_duration < config.min_duration)
        )
        target_penalty = float(not (15.0 <= left_duration <= 22.0))
        score = (
            short_penalty,
            float(rank),
            -pause,
            target_penalty,
            abs(left_duration - OVERSIZED_SPLIT_TARGET),
            float(boundary),
        )
        candidates.append((score, boundary, reason))
    if candidates:
        _, boundary, reason = min(candidates, key=lambda item: item[0])
        return boundary, reason

    # Last resort for one oversized segment only: ordinary word boundary near 18.5 s.
    fallback: list[tuple[tuple[float, ...], int]] = []
    for boundary in range(1, len(words)):
        left_duration = float(words[boundary - 1]["end"] - start)
        if left_duration > config.max_duration + 1e-9:
            continue
        remaining_duration = float(total_end - words[boundary]["start"])
        short_penalty = float(
            left_duration < config.min_duration
            or (remaining_duration <= config.max_duration and remaining_duration < config.min_duration)
        )
        fallback.append(
            ((short_penalty, abs(left_duration - OVERSIZED_SPLIT_TARGET), float(boundary)), boundary)
        )
    if not fallback:
        # A single ASR word longer than max-duration cannot legally be split.
        return len(words), "oversized_segment_word_fallback"
    return min(fallback, key=lambda item: item[0])[1], "oversized_segment_word_fallback"


def split_oversized_segment(
    unit: UtteranceUnit, config: FragmentConfig
) -> list[Fragment]:
    """Split only a single Whisper segment that itself exceeds hard max duration."""
    remaining = list(unit.words)
    pieces: list[Fragment] = []
    start_reason = "segment_boundary"
    while remaining:
        duration = float(remaining[-1]["end"] - remaining[0]["start"])
        if duration <= config.max_duration + 1e-9:
            pieces.append(
                Fragment(
                    words=remaining,
                    segment_ids=(unit.segment_id,),
                    core_start=float(remaining[0]["start"]),
                    core_end=float(remaining[-1]["end"]),
                    start_reason=start_reason,
                    end_reason="segment_boundary",
                    split_reason="",
                    terminal_punctuation=(
                        unit.terminal_punctuation
                        or _ends_with(str(remaining[-1]["text"]), HARD_END_PUNCTUATION)
                    ),
                )
            )
            break
        boundary, split_reason = _choose_long_boundary(remaining, config)
        if boundary >= len(remaining):
            pieces.append(
                Fragment(
                    words=remaining,
                    segment_ids=(unit.segment_id,),
                    core_start=float(remaining[0]["start"]),
                    core_end=float(remaining[-1]["end"]),
                    start_reason=start_reason,
                    end_reason=split_reason,
                    split_reason=split_reason,
                    duration_exception=True,
                    duration_exception_reason="oversized_unsplittable_word",
                )
            )
            break
        left = remaining[:boundary]
        pieces.append(
            Fragment(
                words=left,
                segment_ids=(unit.segment_id,),
                core_start=float(left[0]["start"]),
                core_end=float(left[-1]["end"]),
                start_reason=start_reason,
                end_reason=split_reason,
                split_reason=split_reason,
                terminal_punctuation=_ends_with(
                    str(left[-1]["text"]), HARD_END_PUNCTUATION
                ),
            )
        )
        remaining = remaining[boundary:]
        start_reason = split_reason
    return pieces


# Compatibility for callers of the previous internal helper.
split_long_sentence = split_oversized_segment


def _atomic_segment_fragments(
    units: Sequence[UtteranceUnit], config: FragmentConfig
) -> list[Fragment]:
    atomic: list[Fragment] = []
    for unit in units:
        if unit.duration > config.max_duration + 1e-9:
            atomic.extend(split_oversized_segment(unit, config))
        else:
            atomic.append(
                Fragment(
                    words=list(unit.words),
                    segment_ids=(unit.segment_id,),
                    core_start=unit.start,
                    core_end=unit.end,
                    start_reason="segment_boundary",
                    end_reason="segment_boundary",
                    split_reason="",
                    terminal_punctuation=(
                        unit.terminal_punctuation
                        or _ends_with(str(unit.words[-1]["text"]), HARD_END_PUNCTUATION)
                    ),
                )
            )
    return atomic


def _merge_segment_fragments(first: Fragment, second: Fragment) -> Fragment:
    return Fragment(
        words=[*first.words, *second.words],
        segment_ids=tuple(dict.fromkeys((*first.segment_ids, *second.segment_ids))),
        core_start=first.core_start,
        core_end=second.core_end,
        start_reason=first.start_reason,
        end_reason=second.end_reason,
        # A forced oversized-segment internal boundary belongs to the right edge.
        split_reason=second.split_reason,
        terminal_punctuation=second.terminal_punctuation,
        short_merged=first.short_merged or second.short_merged,
        crossed_soft_threshold=(
            first.crossed_soft_threshold or second.crossed_soft_threshold
        ),
        duration_exception=first.duration_exception or second.duration_exception,
        duration_exception_reason=(
            first.duration_exception_reason or second.duration_exception_reason
        ),
    )


def _has_terminal_punctuation(fragment: Fragment) -> bool:
    return fragment.terminal_punctuation or _ends_with(
        str(fragment.words[-1]["text"]), HARD_END_PUNCTUATION
    )


def _finish_fragment(
    fragment: Fragment,
    reason: str,
    start_reason: str,
    min_duration: float,
    soft_max_duration: float,
) -> Fragment:
    fragment.start_reason = start_reason
    fragment.end_reason = reason
    fallback_reason = reason.startswith(
        ("oversized_segment_", "absolute_max_")
    )
    fragment.split_reason = (
        "short_merge" if fragment.short_merged and not fallback_reason else reason
    )
    fragment.crossed_soft_threshold = (
        fragment.speech_duration > soft_max_duration + 1e-9
    )
    if fragment.speech_duration < min_duration - 1e-9:
        fragment.duration_exception = True
        fragment.duration_exception_reason = "short_residual"
    return fragment


def _combine_atomic_fragments(items: Sequence[Fragment]) -> Fragment:
    if not items:
        raise FragmentError("cannot combine an empty atomic fragment list")
    combined = items[0]
    for item in items[1:]:
        combined = _merge_segment_fragments(combined, item)
    return combined


def _absolute_fallback_boundary(
    current_items: Sequence[Fragment], following: Fragment, config: FragmentConfig
) -> tuple[int, str]:
    """Choose a segment boundary only when the next segment would exceed 30 s."""
    candidates: list[tuple[tuple[float, ...], int, str]] = []
    for boundary in range(1, len(current_items) + 1):
        previous = current_items[boundary - 1]
        after = current_items[boundary] if boundary < len(current_items) else following
        # Absolute fallback never introduces a new within-segment cut.
        if previous.segment_ids[-1] == after.segment_ids[0]:
            continue
        left_duration = previous.core_end - current_items[0].core_start
        right_start = (
            current_items[boundary].core_start
            if boundary < len(current_items)
            else following.core_start
        )
        right_duration = following.core_end - right_start
        in_lookahead_range = (
            config.soft_max_duration - 1e-9
            <= left_duration
            <= config.absolute_max_duration + 1e-9
        )
        range_penalty = float(not in_lookahead_range)
        short_penalty = float(
            left_duration < config.min_duration or right_duration < config.min_duration
        )
        gap = max(0.0, after.core_start - previous.core_end)
        target_penalty = float(not (22.0 <= left_duration <= 27.0))
        target_distance = abs(left_duration - ABSOLUTE_FALLBACK_TARGET)
        if gap > 1e-9:
            reason = "absolute_max_pause_fallback"
            # Maximum historical gap dominates duration target and recency.
            score = (
                range_penalty, short_penalty, 0.0, -gap,
                target_penalty, target_distance, -float(boundary),
            )
        elif previous.terminal_punctuation or _has_terminal_punctuation(previous):
            reason = "absolute_max_punctuation_fallback"
            score = (
                range_penalty, short_penalty, 1.0, 0.0,
                target_penalty, target_distance, -float(boundary),
            )
        else:
            reason = "absolute_max_segment_fallback"
            score = (
                range_penalty, short_penalty, 2.0, 0.0,
                target_penalty, target_distance, -float(boundary),
            )
        candidates.append((score, boundary, reason))
    if not candidates:
        raise FragmentError(
            "absolute max fallback found no legal Whisper segment boundary"
        )
    _, boundary, reason = min(candidates, key=lambda item: item[0])
    return boundary, reason


def group_utterance_units(
    units: Sequence[UtteranceUnit], config: FragmentConfig
) -> list[Fragment]:
    """Group by natural pause with 20–30 s complete-segment lookahead."""
    atomic = _atomic_segment_fragments(units, config)
    if not atomic:
        return []
    grouped: list[Fragment] = []
    current_items = [atomic[0]]
    current_short_merged = False
    start_reason = "clip_start"
    next_index = 1
    oversized_reasons = {
        "oversized_segment_pause",
        "oversized_segment_punctuation",
        "oversized_segment_word_fallback",
    }
    while True:
        current = _combine_atomic_fragments(current_items)
        current.short_merged = current_short_merged
        # Internal oversized-segment cuts remain forced and are the only normal
        # boundaries that may occur inside a Whisper segment.
        if current_items[-1].split_reason in oversized_reasons:
            reason = current_items[-1].split_reason
        elif next_index >= len(atomic):
            reason = "clip_end"
        else:
            following = atomic[next_index]
            gap = max(0.0, following.core_start - current.core_end)
            proposed_duration = following.core_end - current.core_start
            if current.speech_duration < config.min_duration - 1e-9:
                if gap >= MAX_SHORT_MERGE_PAUSE - 1e-9:
                    reason = "strong_pause"
                elif proposed_duration <= config.absolute_max_duration + 1e-9:
                    current_items.append(following)
                    current_short_merged = True
                    next_index += 1
                    continue
                else:
                    boundary, reason = _absolute_fallback_boundary(
                        current_items, following, config
                    )
                    if boundary < len(current_items):
                        left = _combine_atomic_fragments(
                            current_items[:boundary]
                        )
                        left.short_merged = current_short_merged
                        grouped.append(
                            _finish_fragment(
                                left,
                                reason,
                                start_reason,
                                config.min_duration,
                                config.soft_max_duration,
                            )
                        )
                        current_items = current_items[boundary:]
                        start_reason = reason
                        current_short_merged = False
                        continue
            elif gap >= config.strong_pause - 1e-9:
                reason = "strong_pause"
            elif proposed_duration <= config.absolute_max_duration + 1e-9:
                # This is the lookahead path: crossing 20 s alone never ends
                # the utterance; complete segments may accumulate up to 30 s.
                current_items.append(following)
                next_index += 1
                continue
            else:
                boundary, reason = _absolute_fallback_boundary(
                    current_items, following, config
                )
                if boundary < len(current_items):
                    left = _combine_atomic_fragments(current_items[:boundary])
                    left.short_merged = current_short_merged
                    grouped.append(
                        _finish_fragment(
                            left,
                            reason,
                            start_reason,
                            config.min_duration,
                            config.soft_max_duration,
                        )
                    )
                    current_items = current_items[boundary:]
                    start_reason = reason
                    current_short_merged = False
                    continue
        grouped.append(
            _finish_fragment(
                current,
                reason,
                start_reason,
                config.min_duration,
                config.soft_max_duration,
            )
        )
        start_reason = reason
        if next_index >= len(atomic):
            break
        current_items = [atomic[next_index]]
        current_short_merged = False
        next_index += 1
    return grouped


def build_fragments_from_transcript(
    transcript: dict[str, Any], audio_duration: float, config: FragmentConfig | None = None
) -> tuple[list[UtteranceUnit], list[Fragment]]:
    """Group Whisper segments, then allocate unchanged half-gap context padding."""
    config = config or FragmentConfig()
    config.validate()
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise FragmentError("audio duration must be positive")
    units = build_utterance_units(transcript)
    fragments = group_utterance_units(units, config)
    for index, fragment in enumerate(fragments):
        previous_end = fragments[index - 1].core_end if index else 0.0
        next_start = fragments[index + 1].core_start if index + 1 < len(fragments) else audio_duration
        available_before = max(0.0, fragment.core_start - previous_end)
        available_after = max(0.0, next_start - fragment.core_end)
        pre = min(config.max_context_padding, available_before / 2.0)
        post = min(config.max_context_padding, available_after / 2.0)
        # Context must not turn a legal speech unit into an over-max sample.
        context_budget = max(0.0, config.max_duration - fragment.speech_duration)
        if pre + post > context_budget and pre + post > 0.0:
            scale = context_budget / (pre + post)
            pre *= scale
            post *= scale
        # Snap boundaries to the PCM grid so WAV duration and the local timeline agree exactly.
        # Outward snapping guarantees every word remains inside the local interval.
        total_samples = round(audio_duration * AUDIO_SAMPLE_RATE)
        start_sample = max(0, min(math.floor((fragment.core_start - pre) * AUDIO_SAMPLE_RATE), total_samples))
        end_sample = max(start_sample + 1, min(math.ceil((fragment.core_end + post) * AUDIO_SAMPLE_RATE), total_samples))
        fragment.source_start = start_sample / AUDIO_SAMPLE_RATE
        fragment.source_end = end_sample / AUDIO_SAMPLE_RATE
    validate_fragment_overlap(fragments)
    return units, fragments


def validate_fragment_overlap(fragments: Sequence[Fragment]) -> list[str]:
    warnings: list[str] = []
    for previous, current in zip(fragments, fragments[1:]):
        if previous.source_end is None or current.source_start is None:
            raise FragmentError("fragment context boundaries have not been assigned")
        overlap = previous.source_end - current.source_start
        if overlap > OVERLAP_ERROR_SECONDS + 1e-9:
            raise FragmentError(f"adjacent fragment overlap {overlap:.6f}s exceeds 0.50s")
        if overlap > OVERLAP_WARNING_SECONDS + 1e-9:
            message = f"adjacent fragment overlap {overlap:.6f}s exceeds 0.05s"
            previous.alignment_warnings.append(message)
            current.alignment_warnings.append(message)
            warnings.append(message)
    return warnings


def slice_pcm_wav(
    input_path: Path, output_path: Path, source_start: float, source_end: float
) -> dict[str, Any]:
    """Copy an exact sample range from a 16 kHz mono signed-16-bit PCM WAV."""
    try:
        with wave.open(str(input_path), "rb") as source:
            if (
                source.getframerate() != AUDIO_SAMPLE_RATE
                or source.getnchannels() != AUDIO_CHANNELS
                or source.getsampwidth() != AUDIO_SAMPLE_WIDTH_BYTES
                or source.getcomptype() != "NONE"
            ):
                raise FragmentError("Speech V1 audio must be 16 kHz mono PCM s16 WAV")
            total_samples = source.getnframes()
            start_sample = max(0, min(round(source_start * AUDIO_SAMPLE_RATE), total_samples))
            end_sample = max(start_sample, min(round(source_end * AUDIO_SAMPLE_RATE), total_samples))
            if end_sample <= start_sample:
                raise FragmentError("fragment audio interval contains no samples")
            source.setpos(start_sample)
            frames = source.readframes(end_sample - start_sample)
    except (OSError, wave.Error) as exc:
        raise FragmentError(f"could not slice PCM WAV: {exc}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(output_path), "wb") as target:
            target.setnchannels(AUDIO_CHANNELS)
            target.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
            target.setframerate(AUDIO_SAMPLE_RATE)
            target.writeframes(frames)
    except (OSError, wave.Error) as exc:
        raise FragmentError(f"could not write fragment WAV: {exc}") from exc
    return {
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "sample_width_bits": 16,
        "format": "pcm_s16le",
        "sample_count": end_sample - start_sample,
        "duration": (end_sample - start_sample) / AUDIO_SAMPLE_RATE,
        "source_start_sample": start_sample,
        "source_end_sample": end_sample,
    }


def slice_neck_arrays(
    timestamps: np.ndarray,
    valid: np.ndarray,
    neck_rpy: np.ndarray,
    source_start: float,
    source_end: float,
    include_final_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Slice Neck Pose V1 by real timestamps and convert only timestamps to local time."""
    times = np.asarray(timestamps)
    mask = np.asarray(valid)
    rpy = np.asarray(neck_rpy)
    if times.ndim != 1 or mask.shape != times.shape or rpy.shape != (len(times), 3):
        raise FragmentError("Neck Pose V1 arrays have incompatible shapes")
    if mask.dtype != np.bool_:
        raise FragmentError("Neck Pose V1 valid must be boolean")
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise FragmentError("Neck Pose V1 timestamps must be finite and strictly increasing")
    if not np.isfinite(rpy[mask]).all() or not np.isnan(rpy[~mask]).all():
        raise FragmentError("Neck Pose V1 finite/NaN values do not agree with valid")
    start_index = int(np.searchsorted(times, source_start, side="left"))
    side = "right" if include_final_frame else "left"
    end_index = int(np.searchsorted(times, source_end, side=side))
    if end_index <= start_index:
        raise FragmentError("fragment interval contains no Neck Pose V1 frame")
    # Copy exactly; no zeroing, Euler subtraction, interpolation, filtering, or resampling.
    sliced_rpy = rpy[start_index:end_index].copy()
    sliced_valid = mask[start_index:end_index].copy()
    local_times = times[start_index:end_index].astype(np.float64, copy=True) - source_start
    return sliced_rpy, local_times, sliced_valid, (start_index, end_index)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FragmentError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise FragmentError(f"invalid object at {path}:{number}")
                records.append(item)
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentError(f"could not read JSONL {path}: {exc}") from exc
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_source_video_ids(clean_input: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load grouped-split IDs from Clean V1, with an explicit naming-rule fallback."""
    metadata_path = clean_input / "metadata.jsonl" if clean_input.is_dir() else clean_input
    mapping: dict[str, str] = {}
    methods: dict[str, str] = {}
    for record in _read_jsonl(metadata_path):
        clip_id, source_id = record.get("clip_id"), record.get("source_video_id")
        if clip_id and source_id:
            mapping[str(clip_id)] = str(source_id)
            methods[str(clip_id)] = "clean_v1_metadata"
    return mapping, methods


def source_video_id_for_clip(
    clip_id: str, mapping: dict[str, str], methods: dict[str, str] | None = None
) -> tuple[str, str]:
    if clip_id in mapping:
        return mapping[clip_id], (methods or {}).get(clip_id, "clean_v1_metadata")
    # Clean V1's stable schema is <source_video_id>_<four-digit-shot-index>.
    match = re.fullmatch(r"(.+)_\d{4}", clip_id)
    if not match:
        raise FragmentError(f"cannot determine source_video_id for clip {clip_id}")
    return match.group(1), "clip_id_suffix_fallback"


def _word_output(word: dict[str, Any], source_start: float) -> dict[str, Any]:
    item: dict[str, Any] = {
        "text": word["text"],
        "local_start": float(word["start"] - source_start),
        "local_end": float(word["end"] - source_start),
        "source_start": float(word["start"]),
        "source_end": float(word["end"]),
    }
    for key in ("probability", "segment_id"):
        if key in word:
            item[key] = word[key]
    return item


def _validate_fragment_directory(directory: Path, expected_id: str | None = None) -> dict[str, Any] | None:
    if not all((directory / name).is_file() for name in REQUIRED_FRAGMENT_FILES):
        return None
    try:
        metadata = _load_json(directory / "metadata.json")
        transcript = _load_json(directory / "transcript.json")
        if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("status") != "completed":
            return None
        fragment_id = str(metadata["fragment_id"])
        if expected_id is not None and fragment_id != expected_id:
            return None
        if transcript.get("fragment_id") != fragment_id or metadata.get("clip_id") != transcript.get("clip_id"):
            return None
        with wave.open(str(directory / "audio.wav"), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            width = source.getsampwidth()
            compression = source.getcomptype()
            samples = source.getnframes()
        if (sample_rate, channels, width, compression) != (AUDIO_SAMPLE_RATE, 1, 2, "NONE"):
            return None
        duration = float(metadata["duration"])
        source_start, source_end = float(metadata["source_start"]), float(metadata["source_end"])
        tolerance = AUDIO_ROUNDING_TOLERANCE_SAMPLES / sample_rate
        if samples != int(metadata["audio"]["sample_count"]):
            return None
        if (
            abs(samples / sample_rate - duration) > tolerance
            or abs((source_end - source_start) - duration) > tolerance
            or float(transcript["source_start"]) != source_start
            or float(transcript["source_end"]) != source_end
            or transcript.get("text") != metadata.get("text")
            or transcript.get("fragment_type") != "spoken_utterance"
            or transcript.get("segment_ids") != metadata.get("segment_ids")
            or int(transcript["segment_count"]) != int(metadata["segment_count"])
            or int(metadata["segment_count"]) != len(metadata["segment_ids"])
            or metadata.get("fragment_type") != "spoken_utterance"
            or not isinstance(metadata.get("crossed_soft_threshold"), bool)
            or len(transcript["words"]) != int(metadata["word_count"])
        ):
            return None
        word_segment_ids = list(
            dict.fromkeys(int(word["segment_id"]) for word in transcript["words"])
        )
        if word_segment_ids != [int(value) for value in transcript["segment_ids"]]:
            return None
        neck_rpy = np.load(directory / "neck_rpy.npy", allow_pickle=False)
        neck_times = np.load(directory / "neck_timestamps.npy", allow_pickle=False)
        neck_valid = np.load(directory / "neck_valid.npy", allow_pickle=False)
        frame_count = int(metadata["neck"]["frame_count"])
        if neck_rpy.shape != (frame_count, 3) or neck_times.shape != (frame_count,) or neck_valid.shape != (frame_count,):
            return None
        if neck_valid.dtype != np.bool_ or (frame_count and (np.any(np.diff(neck_times) <= 0) or neck_times[0] < -1e-9)):
            return None
        if not np.isfinite(neck_rpy[neck_valid]).all() or not np.isnan(neck_rpy[~neck_valid]).all():
            return None
        if int(neck_valid.sum()) != int(metadata["neck"]["valid_frames"]):
            return None
        frame_tolerance = float(metadata["neck"]["one_frame_tolerance_seconds"])
        if frame_count and neck_times[-1] > duration + frame_tolerance + 1e-9:
            return None
        _validate_local_words(
            transcript.get("words"), duration, float(metadata["source_start"])
        )
        return metadata
    except (OSError, ValueError, TypeError, KeyError, wave.Error, FragmentError):
        return None


def _validate_local_words(
    words: Any, duration: float, fragment_source_start: float | None = None
) -> None:
    if not isinstance(words, list) or not words:
        raise FragmentError("fragment transcript must contain words")
    previous_start = previous_end = -math.inf
    tolerance = AUDIO_ROUNDING_TOLERANCE_SAMPLES / AUDIO_SAMPLE_RATE
    for index, word in enumerate(words):
        try:
            local_start = float(word["local_start"])
            local_end = float(word["local_end"])
            source_start = float(word["source_start"])
            source_end = float(word["source_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FragmentError(f"fragment word {index} has invalid timestamps") from exc
        if (
            local_start < -tolerance
            or local_end < local_start
            or local_end > duration + tolerance
            or source_end < source_start
            or local_start < previous_start - 1e-9
            or local_end < previous_end - 1e-9
            or (
                fragment_source_start is not None
                and abs((source_start - fragment_source_start) - local_start) > tolerance
            )
            or (
                fragment_source_start is not None
                and abs((source_end - fragment_source_start) - local_end) > tolerance
            )
        ):
            raise FragmentError(f"fragment word {index} fails local/source alignment")
        previous_start, previous_end = local_start, local_end


def is_complete_clip_output(
    clip_record: dict[str, Any] | None,
    fragment_records: dict[str, dict[str, Any]],
    fragments_root: Path,
) -> bool:
    """Resume only if the completed clip manifest and every fragment validate."""
    if not clip_record:
        return False
    if (
        clip_record.get("status") != "completed"
        or clip_record.get("feature_version") != FEATURE_VERSION
        or clip_record.get("fragment_type") != "spoken_utterance"
    ):
        return False
    fragment_ids = clip_record.get("fragment_ids")
    if not isinstance(fragment_ids, list) or len(fragment_ids) != int(clip_record.get("fragment_count", -1)):
        return False
    validated_records: list[dict[str, Any]] = []
    total_duration = 0.0
    for fragment_id in fragment_ids:
        record = fragment_records.get(str(fragment_id))
        if (
            not record
            or record.get("clip_id") != clip_record.get("clip_id")
            or record.get("status") != "completed"
            or record.get("feature_version") != FEATURE_VERSION
            or record.get("fragment_type") != "spoken_utterance"
        ):
            return False
        metadata = _validate_fragment_directory(
            fragments_root / str(fragment_id), str(fragment_id)
        )
        if metadata is None:
            return False
        for key in ("source_start", "source_end", "duration"):
            if abs(float(record[key]) - float(metadata[key])) > 1e-9:
                return False
        if (
            int(record["word_count"]) != int(metadata["word_count"])
            or int(record["segment_count"]) != int(metadata["segment_count"])
            or record["segment_ids"] != metadata["segment_ids"]
            or record["split_reason"] != metadata["boundary"]["split_reason"]
            or record["end_reason"] != metadata["boundary"]["end_reason"]
            or bool(record["natural_boundary"]) != bool(metadata["natural_boundary"])
            or bool(record["crossed_soft_threshold"])
            != bool(metadata["crossed_soft_threshold"])
        ):
            return False
        validated_records.append(record)
        total_duration += float(record["duration"])
    try:
        validate_manifest_overlap(validated_records)
    except FragmentError:
        return False
    return abs(total_duration - float(clip_record["total_fragment_duration"])) <= 1e-9


def _load_neck(neck_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [neck_dir / name for name in ("timestamps.npy", "valid.npy", "neck_rpy.npy")]
    metadata_path = neck_dir / "metadata.json"
    if not all(path.is_file() for path in paths) or not metadata_path.is_file():
        raise FragmentError(f"missing Neck Pose V1 input in {neck_dir}")
    metadata = _load_json(metadata_path)
    if metadata.get("feature_version") != "neck_pose_v1" or metadata.get("status") != "completed":
        raise FragmentError(f"invalid Neck Pose V1 metadata in {neck_dir}")
    try:
        arrays = tuple(np.load(path, allow_pickle=False) for path in paths)
    except (OSError, ValueError) as exc:
        raise FragmentError(f"could not load Neck Pose V1 arrays: {exc}") from exc
    if metadata.get("clip_id") not in (None, neck_dir.name):
        raise FragmentError("Neck Pose V1 metadata clip_id mismatch")
    timestamps, valid, neck_rpy = arrays
    if int(metadata.get("frame_count", -1)) != len(timestamps):
        raise FragmentError("Neck Pose V1 metadata frame count mismatch")
    if timestamps.ndim != 1 or valid.shape != timestamps.shape or neck_rpy.shape != (len(timestamps), 3):
        raise FragmentError("Neck Pose V1 arrays have incompatible shapes")
    if valid.dtype != np.bool_ or len(timestamps) == 0:
        raise FragmentError("Neck Pose V1 timestamps/valid are empty or invalid")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise FragmentError("Neck Pose V1 timestamps are not strictly increasing")
    if not np.isfinite(neck_rpy[valid]).all() or not np.isnan(neck_rpy[~valid]).all():
        raise FragmentError("Neck Pose V1 finite/NaN values do not agree with valid")
    return arrays  # type: ignore[return-value]


def _write_fragment(
    directory: Path,
    fragment_id: str,
    clip_id: str,
    source_video_id: str,
    source_video_id_method: str,
    fragment: Fragment,
    audio_path: Path,
    neck_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    audio_duration: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if fragment.source_start is None or fragment.source_end is None:
        raise FragmentError("fragment context has not been assigned")
    source_start, source_end = fragment.source_start, fragment.source_end
    directory.mkdir(parents=True, exist_ok=False)
    audio = slice_pcm_wav(audio_path, directory / "audio.wav", source_start, source_end)
    include_final = source_end >= audio_duration - AUDIO_ROUNDING_TOLERANCE_SAMPLES / AUDIO_SAMPLE_RATE
    neck_rpy, neck_times, neck_valid, frame_range = slice_neck_arrays(
        *neck_arrays, source_start, source_end, include_final_frame=include_final
    )
    np.save(directory / "neck_rpy.npy", neck_rpy, allow_pickle=False)
    np.save(directory / "neck_timestamps.npy", neck_times, allow_pickle=False)
    np.save(directory / "neck_valid.npy", neck_valid, allow_pickle=False)

    duration = audio["duration"]
    valid_frames = int(neck_valid.sum())
    frame_count = len(neck_valid)
    source_neck_times = np.asarray(neck_arrays[0], dtype=np.float64)
    one_frame_tolerance = (
        float(np.median(np.diff(source_neck_times))) if len(source_neck_times) > 1 else 0.0
    )
    valid_ratio = valid_frames / frame_count if frame_count else 0.0
    quality_warning = valid_ratio < NECK_QUALITY_THRESHOLD
    natural_boundary = fragment.natural_boundary
    words = [_word_output(word, source_start) for word in fragment.words]
    transcript = {
        "fragment_id": fragment_id,
        "clip_id": clip_id,
        "text": fragment.text,
        "fragment_type": "spoken_utterance",
        "segment_count": fragment.segment_count,
        "segment_ids": list(fragment.segment_ids),
        "source_start": source_start,
        "source_end": source_end,
        "words": words,
    }
    _write_json(directory / "transcript.json", transcript)
    metadata: dict[str, Any] = {
        "fragment_id": fragment_id,
        "clip_id": clip_id,
        "source_video_id": source_video_id,
        "source_video_id_method": source_video_id_method,
        "source_start": source_start,
        "source_end": source_end,
        "speech_start": fragment.core_start,
        "speech_end": fragment.core_end,
        "duration": duration,
        "fragment_type": "spoken_utterance",
        "segment_count": fragment.segment_count,
        "segment_ids": list(fragment.segment_ids),
        "word_count": len(words),
        "boundary": {
            "start_reason": fragment.start_reason,
            "end_reason": fragment.end_reason,
            "split_reason": fragment.split_reason,
        },
        "duration_exception": fragment.duration_exception,
        "natural_boundary": natural_boundary,
        "crossed_soft_threshold": fragment.crossed_soft_threshold,
        "audio": audio,
        "neck": {
            "frame_count": frame_count,
            "valid_frames": valid_frames,
            "valid_ratio": valid_ratio,
            "source_start_index": frame_range[0],
            "source_end_index_exclusive": frame_range[1],
            "one_frame_tolerance_seconds": one_frame_tolerance,
            "slicing_interval": (
                "[source_start, source_end] at clip end"
                if include_final
                else "[source_start, source_end)"
            ),
            "values": "unchanged Neck Pose V1 neck_rpy slice",
        },
        "quality_warning": quality_warning,
        "text": fragment.text,
        "feature_version": FEATURE_VERSION,
        "status": "completed",
    }
    if fragment.duration_exception_reason:
        metadata["duration_exception_reason"] = fragment.duration_exception_reason
    if fragment.alignment_warnings:
        metadata["alignment_warnings"] = fragment.alignment_warnings
    _write_json(directory / "metadata.json", metadata)
    _validate_written_fragment(
        directory, metadata, transcript, neck_times, one_frame_tolerance
    )
    record = {
        "fragment_id": fragment_id,
        "clip_id": clip_id,
        "source_video_id": source_video_id,
        "source_start": source_start,
        "source_end": source_end,
        "duration": duration,
        "text": fragment.text,
        "fragment_type": "spoken_utterance",
        "segment_count": fragment.segment_count,
        "segment_ids": list(fragment.segment_ids),
        "word_count": len(words),
        "audio_path": str((directory / "audio.wav").resolve()),
        "neck_rpy_path": str((directory / "neck_rpy.npy").resolve()),
        "transcript_path": str((directory / "transcript.json").resolve()),
        "neck_frame_count": frame_count,
        "neck_valid_ratio": valid_ratio,
        "start_reason": fragment.start_reason,
        "end_reason": fragment.end_reason,
        "split_reason": fragment.split_reason,
        "natural_boundary": natural_boundary,
        "crossed_soft_threshold": fragment.crossed_soft_threshold,
        "duration_exception": fragment.duration_exception,
        "duration_exception_reason": fragment.duration_exception_reason,
        "quality_warning": quality_warning,
        "feature_version": FEATURE_VERSION,
        "status": "completed",
    }
    return metadata, record


def _validate_written_fragment(
    directory: Path,
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    neck_times: np.ndarray,
    one_frame_tolerance: float,
) -> None:
    duration = float(metadata["duration"])
    interval = float(metadata["source_end"] - metadata["source_start"])
    tolerance = AUDIO_ROUNDING_TOLERANCE_SAMPLES / AUDIO_SAMPLE_RATE
    if abs(duration - interval) > tolerance:
        raise FragmentError("audio duration differs from source interval by more than two samples")
    _validate_local_words(
        transcript["words"], duration, float(metadata["source_start"])
    )
    if len(neck_times):
        if np.any(np.diff(neck_times) <= 0) or neck_times[0] < -1e-9:
            raise FragmentError("fragment neck timestamps are not non-negative and monotonic")
        neck_source = np.load(directory / "neck_timestamps.npy", allow_pickle=False)
        if not np.array_equal(neck_source, neck_times):
            raise FragmentError("saved neck timestamps differ from validated values")
        if neck_times[-1] > duration + one_frame_tolerance + 1e-9:
            raise FragmentError("last neck timestamp exceeds fragment duration tolerance")
    if _validate_fragment_directory(directory, str(metadata["fragment_id"])) is None:
        raise FragmentError("written fragment failed completeness validation")


def process_fragment_clip(
    speech_dir: Path,
    neck_dir: Path,
    output_fragments_root: Path,
    source_video_id: str,
    source_video_id_method: str,
    config: FragmentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build and atomically commit every Fragment V1.2.1 item for one source clip."""
    clip_id = speech_dir.name
    audio_path = speech_dir / "audio.wav"
    transcript_path = speech_dir / "transcript.json"
    speech_metadata_path = speech_dir / "metadata.json"
    if not audio_path.is_file() or not transcript_path.is_file() or not speech_metadata_path.is_file():
        raise FragmentError(f"incomplete Speech V1 input for {clip_id}")
    speech_metadata = _load_json(speech_metadata_path)
    if speech_metadata.get("feature_version") != "speech_v1" or speech_metadata.get("status") != "completed":
        raise FragmentError(f"invalid Speech V1 metadata for {clip_id}")
    with wave.open(str(audio_path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
            raise FragmentError("Speech V1 audio must be 16 kHz mono PCM s16 WAV")
        audio_duration = source.getnframes() / source.getframerate()
    transcript = _load_json(transcript_path)
    if transcript.get("clip_id") not in (None, clip_id):
        raise FragmentError("Speech V1 transcript clip_id mismatch")
    neck_arrays = _load_neck(neck_dir)
    units, fragments = build_fragments_from_transcript(transcript, audio_duration, config)
    if not fragments:
        raise FragmentError("utterance grouping produced no fragments")

    output_fragments_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output_fragments_root.parent))
    records: list[dict[str, Any]] = []
    try:
        for index, fragment in enumerate(fragments):
            fragment_id = f"{clip_id}_f{index:04d}"
            _, record = _write_fragment(
                staging / fragment_id,
                fragment_id,
                clip_id,
                source_video_id,
                source_video_id_method,
                fragment,
                audio_path,
                neck_arrays,
                audio_duration,
            )
            # Manifests describe committed paths, never per-clip staging paths.
            final_directory = output_fragments_root / fragment_id
            record.update(
                {
                    "audio_path": str((final_directory / "audio.wav").resolve()),
                    "neck_rpy_path": str((final_directory / "neck_rpy.npy").resolve()),
                    "transcript_path": str((final_directory / "transcript.json").resolve()),
                }
            )
            records.append(record)
        validate_manifest_overlap(records)
        old_directories = list(output_fragments_root.glob(f"{clip_id}_f[0-9][0-9][0-9][0-9]"))
        for old in old_directories:
            shutil.rmtree(old)
        for record in records:
            fragment_id = str(record["fragment_id"])
            (staging / fragment_id).replace(output_fragments_root / fragment_id)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    durations = [float(record["duration"]) for record in records]
    clip_record = {
        "clip_id": clip_id,
        "source_video_id": source_video_id,
        "source_video_id_method": source_video_id_method,
        "source_duration": audio_duration,
        "fragment_count": len(records),
        "fragment_ids": [record["fragment_id"] for record in records],
        "total_fragment_duration": sum(durations),
        "fragment_type": "spoken_utterance",
        "segment_count": len(units),
        "word_count": sum(len(unit.words) for unit in units),
        "normal_fragment_count": sum(not bool(record["duration_exception"]) for record in records),
        "short_exception_count": sum(
            record.get("duration_exception_reason") == "short_residual" for record in records
        ),
        "oversized_segment_word_fallback_count": sum(
            record["split_reason"] == "oversized_segment_word_fallback"
            for record in records
        ),
        "crossed_soft_threshold_count": sum(
            bool(record["crossed_soft_threshold"]) for record in records
        ),
        "natural_boundary_count": sum(bool(record["natural_boundary"]) for record in records),
        "fallback_boundary_count": sum(not bool(record["natural_boundary"]) for record in records),
        "quality_warning_count": sum(bool(record["quality_warning"]) for record in records),
        "feature_version": FEATURE_VERSION,
        "status": "completed",
    }
    return clip_record, records


def validate_manifest_overlap(records: Sequence[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    ordered = sorted(records, key=lambda item: float(item["source_start"]))
    for previous, current in zip(ordered, ordered[1:]):
        overlap = float(previous["source_end"]) - float(current["source_start"])
        if overlap > OVERLAP_ERROR_SECONDS + 1e-9:
            raise FragmentError(f"manifest fragment overlap {overlap:.6f}s exceeds 0.50s")
        if overlap > OVERLAP_WARNING_SECONDS + 1e-9:
            warnings.append(
                f"{previous['fragment_id']} and {current['fragment_id']} overlap {overlap:.6f}s"
            )
    return warnings


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def dataset_statistics(
    clip_records: Sequence[dict[str, Any]], fragment_records: Sequence[dict[str, Any]],
    *, input_clips: int, processed_clips: int, failed_clips: int, skipped_clips: int,
) -> dict[str, Any]:
    durations = [float(record["duration"]) for record in fragment_records]
    valid_ratios = [float(record["neck_valid_ratio"]) for record in fragment_records]
    ranges = {
        "<2s": sum(value < 2 for value in durations),
        "2-3s": sum(2 <= value < 3 for value in durations),
        "3-5s": sum(3 <= value < 5 for value in durations),
        "5-8s": sum(5 <= value < 8 for value in durations),
        "8-12s": sum(8 <= value < 12 for value in durations),
        "12-16s": sum(12 <= value < 16 for value in durations),
        "16-20s": sum(16 <= value < 20 for value in durations),
        "20-25s": sum(20 <= value < 25 for value in durations),
        "25-30s": sum(25 <= value <= 30 for value in durations),
        ">30s": sum(value > 30 for value in durations),
    }
    boundary_names = (
        "strong_pause",
        "absolute_max_pause_fallback",
        "absolute_max_punctuation_fallback",
        "absolute_max_segment_fallback",
        "oversized_segment_pause",
        "oversized_segment_punctuation",
        "oversized_segment_word_fallback",
        "clip_end",
    )
    segment_counts = [int(record["segment_count"]) for record in fragment_records]
    natural_count = sum(
        bool(
            record.get(
                "natural_boundary",
                record.get("end_reason", record["split_reason"])
                in {"strong_pause", "clip_end"},
            )
        )
        for record in fragment_records
    )
    crossed_soft_count = sum(
        bool(record.get("crossed_soft_threshold", float(record["duration"]) > 20.0))
        for record in fragment_records
    )

    def boundary_reason(record: dict[str, Any]) -> str:
        return str(record.get("end_reason", record["split_reason"]))
    return {
        "feature_version": FEATURE_VERSION,
        "input_clips": input_clips,
        "processed_clips": processed_clips,
        "skipped_clips": skipped_clips,
        "failed_clips": failed_clips,
        "completed_clip_records": len(clip_records),
        "total_fragments": len(fragment_records),
        "duration": {
            "total": sum(durations),
            "mean": statistics.mean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p05": _percentile(durations, 5),
            "p25": _percentile(durations, 25),
            "p75": _percentile(durations, 75),
            "p95": _percentile(durations, 95),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "ranges": ranges,
        },
        "grouping": {
            "single_segment_fragments": sum(value == 1 for value in segment_counts),
            "multi_segment_fragments": sum(value > 1 for value in segment_counts),
            "mean_segments_per_fragment": (
                statistics.mean(segment_counts) if segment_counts else None
            ),
            "median_segments_per_fragment": (
                statistics.median(segment_counts) if segment_counts else None
            ),
        },
        "boundary_reasons": {
            name: sum(boundary_reason(record) == name for record in fragment_records)
            for name in boundary_names
        },
        "oversized_segment_word_fallback_count": sum(
            record["split_reason"] == "oversized_segment_word_fallback"
            for record in fragment_records
        ),
        "crossed_soft_threshold_fragments": crossed_soft_count,
        "soft_threshold_resolved_by_natural_boundary": sum(
            bool(record.get("crossed_soft_threshold", float(record["duration"]) > 20.0))
            and boundary_reason(record) in {"strong_pause", "clip_end"}
            for record in fragment_records
        ),
        "absolute_max_fallback_fragments": sum(
            boundary_reason(record).startswith("absolute_max_")
            for record in fragment_records
        ),
        "natural_boundary_fragments": natural_count,
        "fallback_boundary_fragments": len(fragment_records) - natural_count,
        "natural_boundary_ratio": (
            natural_count / len(fragment_records) if fragment_records else 0.0
        ),
        "neck": {
            "mean_valid_ratio": statistics.mean(valid_ratios) if valid_ratios else None,
            "fragments_below_90_percent_valid": sum(value < NECK_QUALITY_THRESHOLD for value in valid_ratios),
        },
    }


def _print_value(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{float(value):.3f}{suffix}"


def print_statistics(summary: dict[str, Any]) -> None:
    duration, grouping, boundaries, neck = (
        summary["duration"],
        summary["grouping"],
        summary["boundary_reasons"],
        summary["neck"],
    )
    print()
    print(f"Input clips: {summary['input_clips']}")
    print(f"Processed clips: {summary['processed_clips']}")
    print(f"Skipped clips: {summary['skipped_clips']}")
    print(f"Failed clips: {summary['failed_clips']}")
    print("\nFragments:")
    print(f"Total fragments: {summary['total_fragments']}")
    print("\nDuration:")
    for label, key in (("Total fragment duration", "total"), ("Mean", "mean"), ("Median", "median"), ("P05", "p05"), ("P25", "p25"), ("P75", "p75"), ("P95", "p95"), ("Min", "min"), ("Max", "max")):
        print(f"{label}: {_print_value(duration[key], 's')}")
    print("\nDuration ranges:")
    for label, count in duration["ranges"].items():
        print(f"{label}: {count}")
    print("\nGrouping:")
    print(f"single-segment fragments: {grouping['single_segment_fragments']}")
    print(f"multi-segment fragments: {grouping['multi_segment_fragments']}")
    print(
        "mean segments per fragment: "
        f"{_print_value(grouping['mean_segments_per_fragment'])}"
    )
    print(
        "median segments per fragment: "
        f"{_print_value(grouping['median_segments_per_fragment'])}"
    )
    print("\nBoundary reasons:")
    for reason, count in boundaries.items():
        print(f"{reason}: {count}")
    print(
        "oversized_segment_word_fallback count: "
        f"{summary['oversized_segment_word_fallback_count']}"
    )
    print(
        "crossed soft threshold fragments: "
        f"{summary['crossed_soft_threshold_fragments']}"
    )
    print(
        "soft threshold resolved by natural boundary: "
        f"{summary['soft_threshold_resolved_by_natural_boundary']}"
    )
    print(
        "absolute max fallback fragments: "
        f"{summary['absolute_max_fallback_fragments']}"
    )
    print(f"natural boundary fragments: {summary['natural_boundary_fragments']}")
    print(f"fallback boundary fragments: {summary['fallback_boundary_fragments']}")
    print(f"natural boundary ratio: {summary['natural_boundary_ratio']:.2%}")
    print("\nNeck:")
    ratio = neck["mean_valid_ratio"]
    print(f"mean valid ratio: {'n/a' if ratio is None else f'{ratio:.2%}'}")
    print(f"fragments below 90% valid: {neck['fragments_below_90_percent_valid']}")


def build_fragment_dataset(
    speech_input: Path,
    neck_input: Path,
    clean_input: Path,
    output_root: Path,
    *, config: FragmentConfig | None = None, limit: int | None = None, force: bool = False,
) -> dict[str, Any]:
    """Build Fragment V1.2.1 with unchanged staging, validation, resume, and manifests."""
    config = config or FragmentConfig()
    config.validate()
    for label, path in (("Speech V1", speech_input), ("Neck Pose V1", neck_input), ("Clean V1", clean_input)):
        if not path.is_dir():
            raise FragmentError(f"{label} input directory not found: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("fragment limit must be greater than zero")
    speech_dirs = sorted(path for path in speech_input.iterdir() if path.is_dir() and not path.name.startswith("."))
    if limit is not None:
        speech_dirs = speech_dirs[:limit]
    if not speech_dirs:
        raise FragmentError(f"no Speech V1 clip directories found in: {speech_input}")

    output_root = output_root.resolve()
    fragments_root = output_root / "fragments"
    fragments_root.mkdir(parents=True, exist_ok=True)
    clips_path, fragments_path, failed_path = (
        output_root / "clips.jsonl", output_root / "fragments.jsonl", output_root / "failed.jsonl"
    )
    clip_records = {str(item["clip_id"]): item for item in _read_jsonl(clips_path) if item.get("clip_id")}
    fragment_records = {str(item["fragment_id"]): item for item in _read_jsonl(fragments_path) if item.get("fragment_id")}
    failures = {str(item["clip_id"]): item for item in _read_jsonl(failed_path) if item.get("clip_id")}
    source_ids, source_methods = load_source_video_ids(clean_input)
    selected_ids = {path.name for path in speech_dirs}
    processed = skipped = failed = 0

    for index, speech_dir in enumerate(speech_dirs, 1):
        clip_id = speech_dir.name
        print(f"[{index:03d}/{len(speech_dirs):03d}] {clip_id}", flush=True)
        if not force and is_complete_clip_output(clip_records.get(clip_id), fragment_records, fragments_root):
            skipped += 1
            failures.pop(clip_id, None)
            print("Status: skipped", flush=True)
            continue
        old_record = clip_records.pop(clip_id, None)
        old_ids = old_record.get("fragment_ids", []) if isinstance(old_record, dict) else []
        for fragment_id in old_ids:
            fragment_records.pop(str(fragment_id), None)
        try:
            source_id, source_method = source_video_id_for_clip(clip_id, source_ids, source_methods)
            clip_record, records = process_fragment_clip(
                speech_dir,
                neck_input / clip_id,
                fragments_root,
                source_id,
                source_method,
                config,
            )
            clip_records[clip_id] = clip_record
            fragment_records.update({str(record["fragment_id"]): record for record in records})
            failures.pop(clip_id, None)
            processed += 1
            print(f"Fragments: {len(records)}")
            print("Status: completed", flush=True)
        except Exception as exc:
            failed += 1
            failures[clip_id] = {
                "clip_id": clip_id,
                "stage": "fragment_build",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "feature_version": FEATURE_VERSION,
                "status": "failed",
            }
            print("Status: failed")
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        _write_jsonl(clips_path, sorted(clip_records.values(), key=lambda item: str(item["clip_id"])))
        _write_jsonl(fragments_path, sorted(fragment_records.values(), key=lambda item: str(item["fragment_id"])))
        _write_jsonl(failed_path, sorted(failures.values(), key=lambda item: str(item["clip_id"])))

    selected_clips = [record for clip_id, record in clip_records.items() if clip_id in selected_ids]
    selected_fragments = [record for record in fragment_records.values() if record.get("clip_id") in selected_ids]
    summary = dataset_statistics(
        selected_clips,
        selected_fragments,
        input_clips=len(speech_dirs),
        processed_clips=processed,
        failed_clips=failed,
        skipped_clips=skipped,
    )
    _write_json(output_root / "summary.json", summary)
    print_statistics(summary)
    return summary
