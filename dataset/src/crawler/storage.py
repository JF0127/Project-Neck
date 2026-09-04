"""UTF-8 JSONL and human-readable preview storage helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Overwrite *path* with one UTF-8 JSON object per line."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    content = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write(Path(path), content)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON objects from a UTF-8 JSONL file, ignoring blank lines."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(value)
    return records


def save_preview(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write a compact text view for manual review."""
    blocks: list[str] = []
    for index, record in enumerate(records, start=1):
        duration = record.get("duration")
        duration_text = f"{duration}s" if duration is not None else "Unknown"
        blocks.append(
            f"[{index:03d}] {record.get('title') or 'Untitled'}\n"
            f"Channel: {record.get('channel') or 'Unknown'}\n"
            f"Duration: {duration_text}\n"
            f"URL: {record.get('url') or 'Unknown'}"
        )
    _atomic_write(Path(path), "\n\n".join(blocks) + ("\n" if blocks else ""))


def save_validation_preview(
    validated: Iterable[dict[str, Any]],
    rejected: Iterable[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write validation outcomes in a format suitable for manual review."""
    blocks: list[str] = []
    index = 0
    for status, records in (("VALID", validated), ("REJECTED", rejected)):
        for record in records:
            index += 1
            duration = record.get("duration")
            duration_text = f"{duration}s" if duration is not None else "Unknown"
            block = (
                f"[{index:03d}] {status} | {record.get('id') or 'Unknown'}\n"
                f"Title: {record.get('title') or 'Untitled'}\n"
                f"Channel: {record.get('channel') or 'Unknown'}\n"
                f"Duration: {duration_text}\n"
                f"Upload date: {record.get('upload_date') or 'Unknown'}\n"
                f"URL: {record.get('url') or 'Unknown'}"
            )
            if not record.get("validated"):
                block += f"\nReject reason: {record.get('reject_reason') or 'Unknown'}"
            blocks.append(block)
    _atomic_write(Path(path), "\n\n".join(blocks) + ("\n" if blocks else ""))
