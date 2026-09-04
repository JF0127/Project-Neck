"""Discover candidate videos by invoking yt-dlp's YouTube search extractor."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class DiscoveryError(RuntimeError):
    """Raised when yt-dlp cannot produce a valid search result."""


def _metadata(entry: dict[str, Any], query: str) -> dict[str, Any]:
    video_id = entry.get("id")
    duration = entry.get("duration")
    if isinstance(duration, float) and duration.is_integer():
        duration = int(duration)

    return {
        "id": str(video_id) if video_id is not None else None,
        "title": entry.get("title"),
        "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
        "channel": entry.get("channel") or entry.get("uploader"),
        "channel_id": entry.get("channel_id") or entry.get("uploader_id"),
        "duration": duration,
        "view_count": entry.get("view_count"),
        "upload_date": entry.get("upload_date"),
        "search_query": query,
    }


def discover_videos_with_stats(
    query: str, limit: int, title_keyword: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run yt-dlp search and return normalized candidates plus counters.

    Missing optional metadata is retained as ``None``. A missing ID is rejected
    because it cannot be represented as a playable candidate or deduplicated.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    command = [
        "yt-dlp",
        f"ytsearch{limit}:{query}",
        "--flat-playlist",
        "--dump-json",
        "--skip-download",
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
        details = result.stderr.strip() or "no error details were provided"
        raise DiscoveryError(f"yt-dlp search failed (exit {result.returncode}):\n{details}")

    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(
                f"yt-dlp returned invalid JSON on output line {line_number}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise DiscoveryError(
                f"yt-dlp returned a non-object JSON value on output line {line_number}"
            )
        entries.append(entry)

    keyword = query if title_keyword is None else title_keyword
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rejected = 0
    duplicates = 0

    for entry in entries:
        item = _metadata(entry, query)
        title = item["title"]
        video_id = item["id"]
        if not video_id or not isinstance(title, str) or keyword not in title:
            rejected += 1
            continue
        if video_id in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(video_id)
        candidates.append(item)

    stats = {
        "raw_results": len(entries),
        "accepted": len(candidates),
        "duplicates": duplicates,
        "rejected": rejected,
    }
    return candidates, stats


def discover_videos(query: str, limit: int) -> list[dict[str, Any]]:
    """Discover videos whose titles contain *query*, deduplicated by video ID."""
    candidates, _ = discover_videos_with_stats(query, limit, query)
    return candidates
