"""Download validated YouTube videos without transcoding or other processing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.crawler.storage import load_jsonl, save_jsonl

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"


class DownloadError(RuntimeError):
    """Raised when the download stage cannot be started safely."""


def _load_if_exists(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def _archive_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            ids.add(parts[-1])
    return ids


def _remove_archive_id(path: Path, video_id: str) -> None:
    """Remove a stale archive entry when its corresponding MP4 is absent."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = [line for line in lines if not line.split() or line.split()[-1] != video_id]
    if len(retained) == len(lines):
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write("\n".join(retained) + ("\n" if retained else ""))
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _error_detail(stderr: str, stdout: str) -> str:
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    errors = [line for line in lines if "ERROR:" in line]
    return (errors[-1] if errors else lines[-1] if lines else "unknown yt-dlp error")[:1000]


def _downloaded_record(source: dict[str, Any], local_path: Path) -> dict[str, Any]:
    return {
        "id": source.get("id"),
        "title": source.get("title"),
        "url": source.get("url"),
        "channel": source.get("channel"),
        "upload_date": source.get("upload_date"),
        "duration": source.get("duration"),
        "local_path": str(local_path.resolve()),
        "downloaded": True,
    }


def download_videos(
    records: list[dict[str, Any]], dataset_root: Path
) -> dict[str, Any]:
    """Download records sequentially and persist state after every item."""
    if shutil.which("yt-dlp") is None:
        raise DownloadError("yt-dlp executable was not found")
    if shutil.which("ffmpeg") is None:
        raise DownloadError("ffmpeg executable was not found; it is required for stream merging")

    dataset_root = dataset_root.resolve()
    videos_dir = dataset_root / "videos"
    metadata_dir = dataset_root / "metadata"
    archive_path = dataset_root / "download_archive.txt"
    metadata_path = metadata_dir / "videos.jsonl"
    failed_path = metadata_dir / "download_failed.jsonl"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    archive_path.touch(exist_ok=True)

    downloaded_by_id = {
        str(item.get("id")): item for item in _load_if_exists(metadata_path) if item.get("id")
    }
    failed_by_id = {
        str(item.get("id")): item for item in _load_if_exists(failed_path) if item.get("id")
    }
    save_jsonl(downloaded_by_id.values(), metadata_path)
    save_jsonl(failed_by_id.values(), failed_path)

    downloaded = skipped = failed = 0
    total = len(records)
    for index, source in enumerate(records, start=1):
        video_id = str(source.get("id") or "")
        title = str(source.get("title") or "Untitled")
        url = source.get("url")
        final_path = videos_dir / f"{video_id}.mp4"
        archive_ids = _archive_ids(archive_path)

        print(f"[{index:03d}/{total:03d}] Downloading {video_id or 'Unknown'}")
        print(f"Title: {title}")

        if video_id and video_id in archive_ids and final_path.is_file():
            downloaded_by_id[video_id] = _downloaded_record(source, final_path)
            failed_by_id.pop(video_id, None)
            save_jsonl(downloaded_by_id.values(), metadata_path)
            save_jsonl(failed_by_id.values(), failed_path)
            skipped += 1
            print("Result: SKIPPED (archive)")
            print(f"File: {final_path}")
            print()
            continue

        error: str | None = None
        if not video_id or not _VIDEO_ID.fullmatch(video_id):
            error = "missing or invalid video ID"
        elif not url:
            error = "missing video URL"
        else:
            if video_id in archive_ids and not final_path.exists():
                _remove_archive_id(archive_path, video_id)
            command = [
                "yt-dlp",
                str(url),
                "--no-playlist",
                "--continue",
                "--format",
                _FORMAT,
                "--merge-output-format",
                "mp4",
                "--remux-video",
                "mp4",
                "--output",
                str(videos_dir / "%(id)s.%(ext)s"),
                "--download-archive",
                str(archive_path),
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    error = _error_detail(result.stderr, result.stdout)
                elif not final_path.is_file() or final_path.stat().st_size == 0:
                    error = "yt-dlp exited successfully but the final MP4 is missing or empty"
            except OSError as exc:
                error = f"could not start yt-dlp: {exc}"

        if error is not None:
            failed += 1
            failed_by_id[video_id or f"missing-{index}"] = {
                "id": source.get("id"),
                "url": url,
                "title": source.get("title"),
                "error": error,
            }
            save_jsonl(failed_by_id.values(), failed_path)
            print("Result: FAILED")
            print(f"Error: {error}")
        else:
            downloaded += 1
            downloaded_by_id[video_id] = _downloaded_record(source, final_path)
            failed_by_id.pop(video_id, None)
            save_jsonl(downloaded_by_id.values(), metadata_path)
            save_jsonl(failed_by_id.values(), failed_path)
            print("Result: OK")
            print(f"File: {final_path}")
        print()

    return {
        "total": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "videos_dir": videos_dir,
        "metadata_path": metadata_path,
        "failed_path": failed_path,
        "archive_path": archive_path,
    }
