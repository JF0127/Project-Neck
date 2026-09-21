"""Terminal review tool for the remaining fragment text audit queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from src.processing.fragment_quality_debug import REASON_ORDER, _fragment_reasons

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "zhubo_shuo_lianbo"
EXPECTED_VIDEO_COUNT = 96
EXPECTED_FRAGMENT_COUNT = 1511
VALID_REVIEW_TYPES = {
    "listen_check",
    "possible_truncation",
    "possible_missing_text",
    "ambiguous_asr",
    "proper_noun_check",
}


class ReviewError(RuntimeError):
    """Raised when review cannot continue without risking data consistency."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"could not read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReviewError(f"could not read JSONL: {path}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ReviewError(f"blank JSONL line at {path}:{number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(record, dict):
            raise ReviewError(f"JSONL record must be an object at {path}:{number}")
        records.append(record)
    return records


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")


def _atomic_replace_many(updates: dict[Path, bytes]) -> None:
    """Publish several prepared files, rolling back ordinary write failures."""
    temporary: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    try:
        for path, content in updates.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.review-", suffix=".tmp", dir=path.parent
            )
            candidate = Path(name)
            temporary[path] = candidate
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        for path in updates:
            backup: Path | None = None
            if path.exists():
                backup = path.with_name(f".{path.name}.review-backup-{uuid.uuid4().hex}")
                os.replace(path, backup)
            backups[path] = backup
            os.replace(temporary[path], path)
            published.append(path)
        for backup in backups.values():
            if backup is not None:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
            backup = backups.get(path)
            if backup is not None and backup.exists():
                os.replace(backup, path)
        for path, backup in backups.items():
            if path not in published and backup is not None and backup.exists():
                os.replace(backup, path)
        raise
    finally:
        for candidate in temporary.values():
            candidate.unlink(missing_ok=True)


def _paths(data_root: Path) -> dict[str, Path]:
    quality = data_root / "fragment_quality_debug"
    return {
        "videos": data_root / "videos",
        "fragments": data_root / "fragment_split_debug",
        "review_text": data_root / "fragment_review.txt",
        "quality_results": quality / "results.jsonl",
        "quality_manifest": quality / "manifest.json",
        "remaining": quality / "remaining_review.jsonl",
    }


def _validate_queue(records: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    required = {"video_id", "fragment_id", "text", "reason", "review_type"}
    for number, record in enumerate(records, 1):
        if set(record) != required:
            raise ReviewError(f"unexpected fields in remaining record {number}")
        if not all(isinstance(record[field], str) for field in required):
            raise ReviewError(f"non-string field in remaining record {number}")
        if record["review_type"] not in VALID_REVIEW_TYPES:
            raise ReviewError(f"invalid review_type in remaining record {number}")
        key = (record["video_id"], record["fragment_id"])
        if key in seen:
            raise ReviewError(f"duplicate remaining review key: {key}")
        seen.add(key)


def _fragment_document(
    paths: dict[str, Path], record: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    video_id = record["video_id"]
    fragment_id = record["fragment_id"]
    if Path(video_id).name != video_id or Path(video_id).suffix.lower() != ".mp4":
        raise ReviewError(f"invalid video_id: {video_id!r}")
    path = paths["fragments"] / Path(video_id).stem / "fragments.json"
    document = _load_json(path)
    if document.get("video") != video_id:
        raise ReviewError(f"video mismatch in {path}")
    fragments = document.get("fragments")
    if not isinstance(fragments, list):
        raise ReviewError(f"invalid fragments list: {path}")
    matches = [item for item in fragments if str(item.get("id")) == fragment_id]
    if len(matches) != 1:
        raise ReviewError(
            f"expected one fragment for {video_id}/{fragment_id}, found {len(matches)}"
        )
    fragment = matches[0]
    if fragment.get("text") != record["text"]:
        raise ReviewError(
            f"text mismatch for {video_id}/{fragment_id}; current fragments.json was not overwritten"
        )
    for field in ("start_sec", "end_sec", "duration_sec"):
        try:
            value = float(fragment[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError(f"invalid {field} for {video_id}/{fragment_id}") from exc
        if not math.isfinite(value):
            raise ReviewError(f"non-finite {field} for {video_id}/{fragment_id}")
    return path, document, fragment


def _replace_review_block(
    content: str, video_id: str, fragment_id: str, old_text: str, new_text: str
) -> str:
    header = f"[video={video_id}][fragment={fragment_id}]"
    blocks = content.rstrip("\n").split("\n\n")
    matches = [index for index, block in enumerate(blocks) if block.startswith(header + "\n")]
    if len(matches) != 1:
        raise ReviewError(f"expected one fragment_review block for {video_id}/{fragment_id}")
    index = matches[0]
    existing = blocks[index][len(header) + 1 :]
    if existing != old_text:
        raise ReviewError(f"fragment_review.txt text mismatch for {video_id}/{fragment_id}")
    blocks[index] = f"{header}\n{new_text}"
    return "\n\n".join(blocks) + "\n"


def _updated_quality(
    paths: dict[str, Path],
    video_id: str,
    fragment_id: str,
    old_text: str,
    new_text: str,
    video_fragment_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _load_jsonl(paths["quality_results"])
    matches = [
        record
        for record in records
        if record.get("video") == video_id and str(record.get("fragment_id")) == fragment_id
    ]
    if len(matches) != 1:
        raise ReviewError(f"expected one quality record for {video_id}/{fragment_id}")
    target = matches[0]
    if target.get("text") != old_text:
        raise ReviewError(f"quality results text mismatch for {video_id}/{fragment_id}")
    duration = float(target["duration_sec"])
    reasons = _fragment_reasons(new_text, duration)
    if video_fragment_count < 5:
        reasons.append("video_fragment_count_too_low")
    if video_fragment_count > 30:
        reasons.append("video_fragment_count_too_high")
    reasons = [reason for reason in REASON_ORDER if reason in reasons]
    target["text"] = new_text
    target["status"] = "suspected" if reasons else "ok"
    target["reasons"] = reasons

    manifest = _load_json(paths["quality_manifest"])
    status_counts = Counter(str(record.get("status")) for record in records)
    reason_counts = Counter(
        reason for record in records for reason in record.get("reasons", [])
    )
    manifest["ok_count"] = status_counts.get("ok", 0)
    manifest["suspected_count"] = status_counts.get("suspected", 0)
    manifest["reason_counts"] = {
        reason: reason_counts.get(reason, 0) for reason in REASON_ORDER
    }
    return records, manifest


def _edit_fragment(
    paths: dict[str, Path],
    pending: list[dict[str, Any]],
    record: dict[str, Any],
    new_text: str,
) -> None:
    fragment_path, document, fragment = _fragment_document(paths, record)
    old_text = record["text"]
    if not new_text.strip():
        raise ReviewError("new text must not be empty")
    review_content = paths["review_text"].read_text(encoding="utf-8")
    updated_review = _replace_review_block(
        review_content,
        record["video_id"],
        record["fragment_id"],
        old_text,
        new_text,
    )
    quality_records, manifest = _updated_quality(
        paths,
        record["video_id"],
        record["fragment_id"],
        old_text,
        new_text,
        len(document["fragments"]),
    )
    fragment["text"] = new_text
    new_pending = [item for item in pending if item is not record]
    _atomic_replace_many(
        {
            fragment_path: _json_bytes(document),
            paths["review_text"]: updated_review.encode("utf-8"),
            paths["quality_results"]: _jsonl_bytes(quality_records),
            paths["quality_manifest"]: _json_bytes(manifest),
            paths["remaining"]: _jsonl_bytes(new_pending),
        }
    )
    pending[:] = new_pending


def _confirm_fragment(
    paths: dict[str, Path], pending: list[dict[str, Any]], record: dict[str, Any]
) -> None:
    _fragment_document(paths, record)
    new_pending = [item for item in pending if item is not record]
    _atomic_replace_many({paths["remaining"]: _jsonl_bytes(new_pending)})
    pending[:] = new_pending


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    signature: list[list[Any]] = []
    documents = sorted(paths["fragments"].glob("*/fragments.json"))
    for path in documents:
        document = _load_json(path)
        video = document.get("video")
        fragments = document.get("fragments")
        if not isinstance(video, str) or not isinstance(fragments, list):
            raise ReviewError(f"invalid fragment document: {path}")
        for fragment in fragments:
            signature.append(
                [
                    video,
                    str(fragment["id"]),
                    float(fragment["start_sec"]),
                    float(fragment["end_sec"]),
                    float(fragment["duration_sec"]),
                ]
            )
    videos = sorted(paths["videos"].glob("*.mp4"))
    qwen = sorted(paths["videos"].glob("*.qwen_asr_v1.json"))
    return {
        "video_count": len(videos),
        "fragment_count": len(signature),
        "signature": signature,
        "media": {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in videos},
        "qwen": {str(path): _sha256(path) for path in qwen},
    }


def _validate_snapshot(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        raise ReviewError("dataset structure, media, or raw Qwen metadata changed during review")
    if after["video_count"] != EXPECTED_VIDEO_COUNT:
        raise ReviewError(
            f"expected {EXPECTED_VIDEO_COUNT} videos, found {after['video_count']}"
        )
    if after["fragment_count"] != EXPECTED_FRAGMENT_COUNT:
        raise ReviewError(
            f"expected {EXPECTED_FRAGMENT_COUNT} fragments, found {after['fragment_count']}"
        )


def _select_player(explicit: str | None, no_play: bool) -> tuple[str, list[str]] | None:
    if no_play:
        return None
    candidates = [explicit] if explicit else ["paplay", "ffplay", "aplay"]
    for name in candidates:
        if name is None:
            continue
        executable = shutil.which(name)
        if executable is None:
            continue
        base = Path(executable).name
        if base == "paplay":
            return executable, []
        if base == "ffplay":
            return executable, ["-nodisp", "-autoexit", "-loglevel", "error"]
        if base == "aplay":
            return executable, ["-q"]
        return executable, []
    raise ReviewError("no audio player found; install paplay/ffplay/aplay or use --player")


def _extract_audio(video: Path, fragment: dict[str, Any], output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ReviewError("ffmpeg is required to decode a temporary review WAV")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-ss",
        str(fragment["start_sec"]),
        "-t",
        str(fragment["duration_sec"]),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 44:
        details = result.stderr.strip().splitlines()
        raise ReviewError(
            f"ffmpeg failed for {video.name}: {details[-1] if details else 'unknown error'}"
        )


def _play(player: tuple[str, list[str]] | None, audio: Path) -> None:
    if player is None:
        print("(playback disabled)")
        return
    executable, arguments = player
    result = subprocess.run([executable, *arguments, str(audio)], check=False)
    if result.returncode != 0:
        print(f"Warning: player exited with code {result.returncode}", file=sys.stderr)


def _print_item(number: int, total: int, record: dict[str, Any], audio: Path) -> None:
    print(f"\n[{number}/{total}]\n")
    print(f"video_id: {record['video_id']}")
    print(f"fragment_id: {record['fragment_id']}")
    print(f"review_type: {record['review_type']}")
    print(f"reason: {record['reason']}")
    print("current text:")
    print(record["text"])
    print("\naudio:")
    print(audio)


def review(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    paths = _paths(data_root)
    records = _load_jsonl(paths["remaining"])
    _validate_queue(records)
    if not records:
        print("Review complete.\n\nconfirmed: 0\nedited: 0\nskipped: 0\nremaining: 0")
        return 0
    player = _select_player(args.player, args.no_play)
    before = _dataset_snapshot(paths)
    if before["video_count"] != EXPECTED_VIDEO_COUNT or before["fragment_count"] != EXPECTED_FRAGMENT_COUNT:
        raise ReviewError(
            f"unexpected dataset size: videos={before['video_count']}, fragments={before['fragment_count']}"
        )

    pending = list(records)
    confirmed = 0
    edited = 0
    skipped = 0
    paused = False
    total = len(records)
    for number, record in enumerate(records, 1):
        if record not in pending:
            continue
        try:
            _, _, fragment = _fragment_document(paths, record)
            video = paths["videos"] / record["video_id"]
            if not video.is_file():
                raise ReviewError(f"video not found: {video}")
            with tempfile.TemporaryDirectory(prefix="neck-fragment-review-") as directory:
                audio = Path(directory) / f"{Path(record['video_id']).stem}_{record['fragment_id']}.wav"
                _extract_audio(video, fragment, audio)
                _print_item(number, total, record, audio)
                _play(player, audio)
                while True:
                    try:
                        action = input(
                            "\nEnter=keep, e=edit, r=replay, s=skip, q=quit: "
                        ).strip().lower()
                    except EOFError:
                        action = "q"
                    if action == "":
                        _confirm_fragment(paths, pending, record)
                        confirmed += 1
                        break
                    if action == "e":
                        new_text = input("New text: ").strip()
                        if not new_text:
                            print("New text must not be empty.")
                            continue
                        print("\nBefore:")
                        print(record["text"])
                        print("\nAfter:")
                        print(new_text)
                        _edit_fragment(paths, pending, record, new_text)
                        edited += 1
                        break
                    if action == "r":
                        _play(player, audio)
                        continue
                    if action == "s":
                        skipped += 1
                        break
                    if action == "q":
                        paused = True
                        break
                    print("Unknown action. Use Enter, e, r, s, or q.")
        except ReviewError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Item left in remaining_review.jsonl.", file=sys.stderr)
            skipped += 1
        if paused:
            break

    after = _dataset_snapshot(paths)
    _validate_snapshot(before, after)
    heading = "Review paused." if paused else "Review complete."
    print(f"\n{heading}\n")
    print(f"confirmed: {confirmed}")
    print(f"edited: {edited}")
    print(f"skipped: {skipped}")
    print(f"remaining: {len(pending)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review remaining fragment texts with audio")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--player", help="override audio player executable")
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="decode but do not play audio (for software diagnostics)",
    )
    args = parser.parse_args(argv)
    lock_path = Path(tempfile.gettempdir()) / "project-neck-fragment-review.lock"
    try:
        with lock_path.open("w") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ReviewError("another fragment review process is running") from exc
            return review(args)
    except (OSError, ValueError, ReviewError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
