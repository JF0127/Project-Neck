"""Validate discovered videos using full yt-dlp metadata extraction."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any, Callable

from src.crawler.discover import DiscoveryError

ProgressCallback = Callable[[int, int, str], None]
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _error_detail(stderr: str) -> str:
    lines = [line.strip() for line in _ANSI_ESCAPE.sub("", stderr).splitlines() if line.strip()]
    error_lines = [line for line in lines if line.startswith("ERROR:")]
    return (error_lines[-1] if error_lines else lines[-1] if lines else "unknown yt-dlp error")[:500]


def _has_video_format(metadata: dict[str, Any]) -> bool:
    for item in metadata.get("formats") or []:
        if (
            isinstance(item, dict)
            and item.get("url")
            and item.get("vcodec") not in (None, "none")
        ):
            return True
    return False


def _normalized_metadata(
    metadata: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    video_id = metadata.get("id") or candidate.get("id")
    return {
        "id": video_id,
        "title": metadata.get("title"),
        "url": metadata.get("webpage_url")
        or (f"https://www.youtube.com/watch?v={video_id}" if video_id else candidate.get("url")),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "channel_id": metadata.get("channel_id") or metadata.get("uploader_id"),
        "upload_date": metadata.get("upload_date"),
        "duration": metadata.get("duration"),
        "description": metadata.get("description"),
        "view_count": metadata.get("view_count"),
        "availability": metadata.get("availability"),
        "live_status": metadata.get("live_status"),
    }


def _reject_reasons(
    metadata: dict[str, Any], record: dict[str, Any], title_keyword: str, allowed_channels: set[str]
) -> list[str]:
    reasons: list[str] = []
    title = record.get("title")
    channel = record.get("channel")
    duration = record.get("duration")
    live_status = record.get("live_status")

    if not isinstance(title, str) or title_keyword not in title:
        reasons.append(f'title does not contain "{title_keyword}"')
    if channel not in allowed_channels:
        reasons.append(f"channel is not allowed: {channel or 'missing'}")
    if metadata.get("is_live") or metadata.get("was_live") or live_status not in (None, "not_live"):
        reasons.append(f"live content is not allowed: {live_status or 'live'}")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        reasons.append("duration is missing or non-positive")
    elif duration > 21600:
        reasons.append(f"duration is clearly unreasonable: {duration}s")
    if record.get("availability") in {"private", "needs_auth", "subscriber_only"}:
        reasons.append(f"video is not publicly accessible: {record['availability']}")
    if not _has_video_format(metadata):
        reasons.append("no downloadable video format could be parsed")
    return reasons


def validate_candidates(
    candidates: list[dict[str, Any]],
    title_keyword: str,
    allowed_channels: set[str],
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate each candidate without downloading media."""
    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total = len(candidates)

    for index, candidate in enumerate(candidates, start=1):
        video_id = str(candidate.get("id") or "")
        url = candidate.get("url") or (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        )
        if progress:
            progress(index, total, video_id)
        if not url:
            rejected.append({**candidate, "validated": False, "reject_reason": "missing video URL"})
            continue

        command = [
            "yt-dlp",
            str(url),
            "--dump-single-json",
            "--skip-download",
            "--no-playlist",
            "--extractor-retries",
            "3",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise DiscoveryError(
                "yt-dlp executable was not found. Install it with: pip install -r requirements.txt"
            ) from exc
        except OSError as exc:
            raise DiscoveryError(f"Could not start yt-dlp: {exc}") from exc

        if result.returncode != 0:
            rejected.append(
                {
                    **candidate,
                    "validated": False,
                    "reject_reason": f"yt-dlp metadata error: {_error_detail(result.stderr)}",
                }
            )
            continue
        try:
            metadata = json.loads(result.stdout)
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            rejected.append(
                {
                    **candidate,
                    "validated": False,
                    "reject_reason": f"invalid yt-dlp metadata: {exc}",
                }
            )
            continue

        record = _normalized_metadata(metadata, candidate)
        reasons = _reject_reasons(metadata, record, title_keyword, allowed_channels)
        if reasons:
            rejected.append(
                {**record, "validated": False, "reject_reason": "; ".join(reasons)}
            )
        else:
            validated.append({**record, "validated": True})

    return validated, rejected


def stderr_progress(index: int, total: int, video_id: str) -> None:
    """Print validation progress without mixing it with final stdout statistics."""
    print(f"Validating [{index}/{total}] {video_id}", file=sys.stderr, flush=True)
