"""Low-dependency phrase alignment against complete 16 kHz TTS PCM."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import re
import sys
import time
from typing import Sequence

SAMPLE_RATE = 16_000
_FRAME_SEC = 0.01
_MIN_PAUSE_SEC = 0.04
_MAX_PHRASES = 4
_BOUNDARY_RE = re.compile(r"[，。？！；：、,.!?;:\n]+")
_SEMANTIC_SPLIT_RE = re.compile(
    r"(?=(?:但是|不过|然后|所以|而且|同时|接下来))"
)
_TRIM_CHARS = " \t\r\n“”‘’\"'（）()【】[]《》<>"


class SpeechAlignmentError(RuntimeError):
    """TTS PCM could not be converted into safe phrase intervals."""


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    start: float
    end: float

    def as_dict(self) -> dict[str, str | float]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class SpeechAlignment:
    segments: tuple[SpeechSegment, ...]
    method: str
    latency_sec: float
    fallback_reason: str | None = None

    def segments_as_dicts(self) -> list[dict[str, str | float]]:
        return [segment.as_dict() for segment in self.segments]

    def metadata(self) -> dict[str, str | float]:
        document: dict[str, str | float] = {
            "method": self.method,
            "latency_sec": self.latency_sec,
        }
        if self.fallback_reason is not None:
            document["fallback_reason"] = self.fallback_reason
        return document


def _lexical_length(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _clean_phrase(value: str) -> str:
    return value.strip(_TRIM_CHARS)


def split_phrases(robot_text: str) -> tuple[str, ...]:
    """Split Chinese text conservatively into at most four semantic phrases."""
    if not isinstance(robot_text, str) or not robot_text.strip():
        raise SpeechAlignmentError("robot_text is empty")
    clean_text = robot_text.strip()
    parts = [_clean_phrase(part) for part in _BOUNDARY_RE.split(clean_text)]
    phrases = [part for part in parts if part]

    if len(phrases) == 1 and _lexical_length(phrases[0]) > 12:
        semantic_parts = [
            _clean_phrase(part) for part in _SEMANTIC_SPLIT_RE.split(phrases[0])
        ]
        semantic_parts = [part for part in semantic_parts if part]
        if len(semantic_parts) > 1:
            phrases = semantic_parts

    index = 0
    while len(phrases) > 1 and index < len(phrases):
        if _lexical_length(phrases[index]) > 1:
            index += 1
            continue
        if index == 0:
            phrases[1] = phrases[0] + phrases[1]
            del phrases[0]
        else:
            phrases[index - 1] += phrases[index]
            del phrases[index]
            index -= 1

    while len(phrases) > _MAX_PHRASES:
        pair_index = min(
            range(len(phrases) - 1),
            key=lambda item: (
                _lexical_length(phrases[item])
                + _lexical_length(phrases[item + 1]),
                item,
            ),
        )
        phrases[pair_index : pair_index + 2] = [
            phrases[pair_index] + phrases[pair_index + 1]
        ]

    if not phrases:
        raise SpeechAlignmentError("robot_text contains no phrase text")
    return tuple(phrases)


def _pcm_samples(pcm_s16le: bytes) -> array:
    if not isinstance(pcm_s16le, bytes) or not pcm_s16le:
        raise SpeechAlignmentError("TTS PCM is empty")
    if len(pcm_s16le) % 2:
        raise SpeechAlignmentError("TTS PCM byte length must be even")
    samples = array("h")
    samples.frombytes(pcm_s16le)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _frame_rms(samples: Sequence[int], frame_samples: int) -> list[float]:
    values: list[float] = []
    for start in range(0, len(samples), frame_samples):
        frame = samples[start : start + frame_samples]
        if not frame:
            continue
        square_sum = sum(float(sample) * float(sample) for sample in frame)
        values.append(math.sqrt(square_sum / len(frame)))
    return values


def _inactive_runs(
    active: Sequence[bool], first: int, last: int, minimum_frames: int
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = first + 1
    while index < last:
        if active[index]:
            index += 1
            continue
        start = index
        while index < last and not active[index]:
            index += 1
        if index - start >= minimum_frames:
            runs.append((start, index))
    return runs


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _choose_energy_valley(
    rms: Sequence[float],
    target: float,
    lower: float,
    upper: float,
    span: float,
    phrase_count: int,
) -> tuple[float, float] | None:
    radius = max(0.12, min(0.5, span / max(2.0, phrase_count * 1.5)))
    first = max(0, int(math.floor(max(lower, target - radius) / _FRAME_SEC)))
    last = min(
        len(rms) - 1,
        int(math.ceil(min(upper, target + radius) / _FRAME_SEC)),
    )
    if last <= first:
        return None
    local = rms[first : last + 1]
    local_median = _median(local)
    if local_median <= 0.0:
        return None
    valley_index = min(
        range(first, last + 1),
        key=lambda index: (
            rms[index] / local_median
            + 0.15 * abs((index + 0.5) * _FRAME_SEC - target) / radius,
            abs((index + 0.5) * _FRAME_SEC - target),
        ),
    )
    if rms[valley_index] > local_median * 0.65:
        return None
    start = max(lower, valley_index * _FRAME_SEC)
    end = min(upper, (valley_index + 1) * _FRAME_SEC)
    if end < start:
        return None
    return start, end


def _validate_segments(
    segments: Sequence[SpeechSegment], duration_sec: float
) -> None:
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if (
            not segment.text
            or not math.isfinite(segment.start)
            or not math.isfinite(segment.end)
            or segment.start < -1e-9
            or segment.end > duration_sec + 1e-9
            or segment.start >= segment.end
            or (index and segment.start < previous_end - 1e-9)
        ):
            raise SpeechAlignmentError(
                f"invalid aligned segment at index {index}"
            )
        previous_end = segment.end


def align_speech(
    robot_text: str,
    pcm_s16le: bytes,
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> SpeechAlignment:
    """Align a few text phrases to pauses/energy valleys in the actual PCM."""
    started = time.perf_counter()
    if sample_rate != SAMPLE_RATE:
        raise SpeechAlignmentError("speech alignment requires 16000 Hz PCM")
    try:
        duration = float(duration_sec)
    except (TypeError, ValueError) as exc:
        raise SpeechAlignmentError("duration_sec must be numeric") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise SpeechAlignmentError("duration_sec must be positive")

    phrases = split_phrases(robot_text)
    samples = _pcm_samples(pcm_s16le)
    audio_duration = len(samples) / sample_rate
    timeline_end = min(duration, audio_duration)
    if timeline_end <= 0.0:
        raise SpeechAlignmentError("TTS PCM duration is empty")

    frame_samples = max(1, round(sample_rate * _FRAME_SEC))
    usable_samples = samples[: int(math.ceil(timeline_end * sample_rate))]
    rms = _frame_rms(usable_samples, frame_samples)
    if not rms:
        raise SpeechAlignmentError("TTS PCM contains no analysis frames")
    peak = max(rms)
    if peak < 8.0:
        raise SpeechAlignmentError("TTS PCM contains no detectable speech")
    ordered_rms = sorted(rms)
    noise_floor = ordered_rms[min(len(ordered_rms) - 1, len(ordered_rms) // 5)]
    threshold = max(8.0, peak * 0.06)
    if noise_floor < peak * 0.3:
        threshold = max(threshold, noise_floor * 3.0)
    active = [value >= threshold for value in rms]
    active_indices = [index for index, value in enumerate(active) if value]
    if not active_indices:
        raise SpeechAlignmentError("TTS PCM contains no active speech frames")

    first_active = active_indices[0]
    last_active = active_indices[-1]
    speech_start = max(0.0, (first_active - 1) * _FRAME_SEC)
    speech_end = min(timeline_end, (last_active + 2) * _FRAME_SEC)
    if speech_end <= speech_start:
        raise SpeechAlignmentError("detected speech interval is empty")

    if len(phrases) == 1:
        segments = (
            SpeechSegment(phrases[0], round(speech_start, 6), round(speech_end, 6)),
        )
        _validate_segments(segments, duration)
        return SpeechAlignment(
            segments=segments,
            method="pcm_speech_bounds",
            latency_sec=time.perf_counter() - started,
        )

    minimum_pause_frames = max(1, round(_MIN_PAUSE_SEC / _FRAME_SEC))
    pause_runs = _inactive_runs(
        active, first_active, last_active, minimum_pause_frames
    )
    pause_intervals = [
        (start * _FRAME_SEC, end * _FRAME_SEC)
        for start, end in pause_runs
    ]
    weights = [max(1, _lexical_length(phrase)) for phrase in phrases]
    weight_total = sum(weights)
    span = speech_end - speech_start
    boundaries: list[tuple[float, float]] = []
    previous_right = speech_start
    used_pause_indices: set[int] = set()
    used_energy_valley = False
    used_proportional = False
    cumulative_weight = 0
    minimum_segment = min(0.05, span / (len(phrases) * 3.0))

    for boundary_index in range(len(phrases) - 1):
        cumulative_weight += weights[boundary_index]
        target = speech_start + span * cumulative_weight / weight_total
        remaining_phrases = len(phrases) - boundary_index - 1
        lower = previous_right + minimum_segment
        upper = speech_end - remaining_phrases * minimum_segment
        if upper <= lower:
            raise SpeechAlignmentError("audio is too short for phrase alignment")

        candidates = [
            (index, interval)
            for index, interval in enumerate(pause_intervals)
            if index not in used_pause_indices
            and interval[0] >= lower
            and interval[1] <= upper
        ]
        search_radius = max(0.2, span / max(2.0, len(phrases)))
        candidates = [
            item
            for item in candidates
            if abs((item[1][0] + item[1][1]) * 0.5 - target)
            <= search_radius
        ]
        if candidates:
            selected_index, boundary = min(
                candidates,
                key=lambda item: abs(
                    (item[1][0] + item[1][1]) * 0.5 - target
                ),
            )
            used_pause_indices.add(selected_index)
        else:
            boundary = _choose_energy_valley(
                rms, target, lower, upper, span, len(phrases)
            )
            if boundary is not None:
                used_energy_valley = True
            else:
                point = min(upper, max(lower, target))
                boundary = (point, point)
                used_proportional = True
        boundaries.append(boundary)
        previous_right = boundary[1]

    segments_list: list[SpeechSegment] = []
    for index, phrase in enumerate(phrases):
        start = speech_start if index == 0 else boundaries[index - 1][1]
        end = speech_end if index == len(phrases) - 1 else boundaries[index][0]
        segments_list.append(
            SpeechSegment(phrase, round(start, 6), round(end, 6))
        )
    segments = tuple(segments_list)
    _validate_segments(segments, duration)

    fallback_reason: str | None = None
    if used_proportional:
        method = "proportional_fallback"
        fallback_reason = "insufficient PCM pause or energy-valley boundaries"
    elif used_energy_valley:
        method = "pcm_energy_valley_alignment"
    else:
        method = "pcm_pause_alignment"
    return SpeechAlignment(
        segments=segments,
        method=method,
        latency_sec=time.perf_counter() - started,
        fallback_reason=fallback_reason,
    )
