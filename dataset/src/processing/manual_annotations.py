"""Load and validate manually segmented sentence annotations."""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManualSegment:
    id: str
    source: str
    file: str
    start: float
    end: float
    duration: float
    text: str


def load_annotations(path: Path, *, validate_audio: bool = False) -> list[ManualSegment]:
    """Load the manual JSONL contract and optionally verify every segment WAV."""
    root = path.parent
    segments: list[ManualSegment] = []
    ids: set[str] = set()
    source_name: str | None = None
    previous_end = 0.0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                segment_id = str(item["id"]).strip()
                source_file = str(item["source"]).strip()
                relative_file = str(item["file"]).strip()
                start = float(item["start_sec"])
                end = float(item["end_sec"])
                duration = float(item["duration_sec"])
                text = str(item["text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid annotation at {path}:{line_number}") from exc

            relative_path = Path(relative_file)
            if (
                not segment_id
                or segment_id in ids
                or not source_file
                or Path(source_file).name != source_file
                or "\\" in source_file
                or Path(source_file).suffix.lower() != ".wav"
                or not text
                or relative_path.suffix.lower() != ".wav"
                or "\\" in relative_file
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or not all(math.isfinite(value) for value in (start, end, duration))
                or start < previous_end - 1e-6
                or end <= start
                or duration <= 0
                or abs(duration - (end - start)) > 1e-3
            ):
                raise ValueError(f"invalid annotation values at {path}:{line_number}")
            if source_name is not None and source_file != source_name:
                raise ValueError(f"mixed annotation sources at {path}:{line_number}")

            segment = ManualSegment(
                id=segment_id,
                source=source_file,
                file=relative_file,
                start=start,
                end=end,
                duration=duration,
                text=text,
            )
            if validate_audio:
                _validate_wav(root / relative_path, segment)
            segments.append(segment)
            ids.add(segment_id)
            source_name = source_file
            previous_end = end

    if not segments:
        raise ValueError(f"annotation file contains no segments: {path}")
    return segments


def _validate_wav(path: Path, segment: ManualSegment) -> None:
    try:
        with wave.open(str(path), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"segment WAV must be 16 kHz mono PCM s16: {path}")
            actual_duration = source.getnframes() / source.getframerate()
    except (OSError, wave.Error) as exc:
        raise ValueError(f"could not read segment WAV: {path}") from exc
    if abs(actual_duration - segment.duration) > 1.5 / 16000:
        raise ValueError(
            f"segment WAV duration mismatch for {segment.id}: "
            f"metadata={segment.duration:.6f}, wav={actual_duration:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a manual sentence annotation package")
    parser.add_argument("--annotations", required=True, type=Path, help="package metadata.jsonl")
    args = parser.parse_args()
    path = args.annotations.resolve()
    if not path.is_file():
        parser.error(f"manual annotations not found: {path}")
    segments = load_annotations(path, validate_audio=True)
    print(f"Source: {segments[0].source}")
    print(f"Segments: {len(segments)}")
    print(f"Annotated duration: {sum(segment.duration for segment in segments):.3f}s")
    print(f"Timeline: {segments[0].start:.3f}s - {segments[-1].end:.3f}s")
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
