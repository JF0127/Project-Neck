"""Place each top-level MP4 in its own directory and extract 16 kHz PCM WAV."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import uuid
import wave
from pathlib import Path


class VideoAudioPreparationError(RuntimeError):
    """Raised when a video/audio pair cannot be prepared safely."""


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as source:
            audio_format = (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getcomptype(),
            )
            frame_count = source.getnframes()
    except (OSError, wave.Error) as exc:
        raise VideoAudioPreparationError(f"could not read extracted WAV: {path}") from exc

    if audio_format != (16000, 1, 2, "NONE"):
        raise VideoAudioPreparationError(
            f"extracted WAV is not 16 kHz mono PCM s16: {path}"
        )
    if frame_count <= 0:
        raise VideoAudioPreparationError(f"extracted WAV is empty: {path}")


def _extract_wav(video: Path, output: Path, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        raise VideoAudioPreparationError(
            f"ffmpeg failed for {video.name}: "
            f"{message[-1] if message else 'unknown error'}"
        )
    _validate_wav(output)


def prepare_video_audio(root: Path) -> list[dict[str, str]]:
    """Prepare every MP4 directly under ``root`` without overwriting outputs.

    Audio extraction completes in a staging directory before the source MP4 is
    moved. Each final per-video directory therefore appears as one rename.
    """
    root = root.resolve()
    if not root.is_dir():
        raise VideoAudioPreparationError(f"input directory not found: {root}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoAudioPreparationError("ffmpeg is required")

    videos = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise VideoAudioPreparationError(f"no top-level MP4 files found in: {root}")

    destinations = [root / video.stem for video in videos]
    existing = [path for path in destinations if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise VideoAudioPreparationError(
            f"destination directories already exist; refusing to overwrite: {names}"
        )

    prepared: list[dict[str, str]] = []
    for video, destination in zip(videos, destinations):
        staging = root / f".{video.stem}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        staged_video = staging / video.name
        staged_wav = staging / f"{video.stem}.wav"
        video_moved = False
        try:
            _extract_wav(video, staged_wav, ffmpeg)
            os.replace(video, staged_video)
            video_moved = True
            os.replace(staging, destination)
            video_moved = False
        except Exception:
            if video_moved and staged_video.exists() and not video.exists():
                os.replace(staged_video, video)
            shutil.rmtree(staging, ignore_errors=True)
            raise

        prepared.append(
            {
                "directory": str(destination),
                "video": str(destination / video.name),
                "audio": str(destination / staged_wav.name),
            }
        )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Place each MP4 in its own directory and extract 16 kHz mono PCM WAV"
    )
    parser.add_argument("--input", required=True, type=Path, help="directory containing MP4 files")
    args = parser.parse_args()
    try:
        prepared = prepare_video_audio(args.input)
    except (OSError, VideoAudioPreparationError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    for item in prepared:
        print(f"Prepared: {item['directory']}")
    print(f"Videos prepared: {len(prepared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
