"""PyTorch data access for the canonical Dataset V1 fragments."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")
SAMPLE_RATE = 16_000


class DatasetValidationError(ValueError):
    """A canonical artifact does not satisfy the Dataset V1 contract."""


def _default_dataset_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "dataset"
        / "datasets"
        / "zhubo_shuo_lianbo"
    )


def _read_json(path: Path, fragment_id: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"fragment_id={fragment_id} path={path}: cannot read JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(
            f"fragment_id={fragment_id} path={path}: expected a JSON object"
        )
    return value


def _load_npy(path: Path, fragment_id: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(
            f"fragment_id={fragment_id} path={path}: cannot read NPY: {exc}"
        ) from exc


class NeckMotionDataset(Dataset[dict[str, Any]]):
    """One canonical Fragment V1.2.1 utterance per item.

    ``reference_valid`` explicitly records whether the first neck sample can
    define the requested fragment-relative target. If it is false, no fallback
    reference is chosen: ``target_rpy`` remains NaN and ``target_valid_mask`` is
    false for the whole item. ``valid_mask`` always remains the unmodified
    canonical neck validity mask.
    """

    def __init__(
        self,
        split: str,
        dataset_root: str | Path | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.dataset_root = Path(dataset_root) if dataset_root else _default_dataset_root()
        self.dataset_root = self.dataset_root.expanduser().resolve()
        self.manifest_path = self.dataset_root / "splits" / "split_v1" / f"{split}.jsonl"
        self._rows = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Any]]:
        try:
            lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DatasetValidationError(
                f"split={self.split} path={self.manifest_path}: cannot read split_v1 manifest: {exc}"
            ) from exc

        rows: list[dict[str, Any]] = []
        required = {
            "fragment_id",
            "clip_id",
            "source_video_id",
            "audio_path",
            "neck_rpy_path",
            "transcript_path",
            "split",
        }
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"split={self.split} path={self.manifest_path} line={line_number}: {exc}"
                ) from exc
            missing = required - row.keys() if isinstance(row, dict) else required
            if missing:
                raise DatasetValidationError(
                    f"split={self.split} path={self.manifest_path} line={line_number}: "
                    f"missing fields {sorted(missing)}"
                )
            if row["split"] != self.split:
                raise DatasetValidationError(
                    f"fragment_id={row['fragment_id']} path={self.manifest_path}: "
                    f"row split {row['split']!r} != requested split {self.split!r}"
                )
            rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def manifest_rows(self) -> tuple[Mapping[str, Any], ...]:
        """Read-only-style access for membership/leakage checks without loading artifacts."""
        return tuple(self._rows)

    def iter_word_entries(self) -> Iterator[list[dict[str, Any]]]:
        """Read only transcript words, so train-vocab construction does not load audio/neck arrays."""
        for row in self._rows:
            fragment_id = str(row["fragment_id"])
            transcript_path = self._artifact_path(row, "transcript_path")
            transcript = _read_json(transcript_path, fragment_id)
            words = transcript.get("words")
            if not isinstance(words, list):
                raise DatasetValidationError(
                    f"fragment_id={fragment_id} path={transcript_path}: expected list words"
                )
            yield words

    def _artifact_path(self, row: Mapping[str, Any], field: str) -> Path:
        """Use the manifest path, with an explicit checkout-relocation fallback."""
        manifest_path = Path(row[field]).expanduser()
        if manifest_path.is_file():
            return manifest_path
        fragment_dir = (
            self.dataset_root
            / "fragments"
            / "fragment_v1"
            / "fragments"
            / str(row["fragment_id"])
        )
        relocated_path = fragment_dir / manifest_path.name
        if relocated_path.is_file():
            return relocated_path
        raise DatasetValidationError(
            f"fragment_id={row['fragment_id']} path={manifest_path}: artifact does not exist "
            f"(relocation fallback also missing: {relocated_path})"
        )

    def _read_audio(self, path: Path, fragment_id: str) -> torch.Tensor:
        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                compression = wav.getcomptype()
                frame_count = wav.getnframes()
                pcm = wav.readframes(frame_count)
        except (OSError, EOFError, wave.Error) as exc:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={path}: cannot read WAV: {exc}"
            ) from exc

        if sample_rate != SAMPLE_RATE:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={path}: sample rate {sample_rate}, expected {SAMPLE_RATE}"
            )
        if channels != 1 or sample_width != 2 or compression != "NONE":
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={path}: expected mono PCM signed 16-bit WAV, "
                f"got channels={channels}, sample_width={sample_width}, compression={compression}"
            )
        expected_bytes = frame_count * 2
        if len(pcm) != expected_bytes:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={path}: read {len(pcm)} PCM bytes, "
                f"expected {expected_bytes}"
            )
        # Copy because torch tensors cannot safely own the read-only frombuffer view.
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return torch.from_numpy(waveform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        fragment_id = str(row["fragment_id"])
        audio_path = self._artifact_path(row, "audio_path")
        rpy_path = self._artifact_path(row, "neck_rpy_path")
        transcript_path = self._artifact_path(row, "transcript_path")
        fragment_dir = rpy_path.parent
        timestamps_path = fragment_dir / "neck_timestamps.npy"
        valid_path = fragment_dir / "neck_valid.npy"

        audio = self._read_audio(audio_path, fragment_id)
        raw_rpy_array = _load_npy(rpy_path, fragment_id)
        timestamps_array = _load_npy(timestamps_path, fragment_id)
        valid_array = _load_npy(valid_path, fragment_id)
        transcript = _read_json(transcript_path, fragment_id)

        if raw_rpy_array.ndim != 2 or raw_rpy_array.shape[1] != 3:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={rpy_path}: raw_rpy shape "
                f"{raw_rpy_array.shape}, expected [T, 3]"
            )
        if timestamps_array.ndim != 1:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={timestamps_path}: neck timestamps shape "
                f"{timestamps_array.shape}, expected [T]"
            )
        if valid_array.ndim != 1:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={valid_path}: valid mask shape "
                f"{valid_array.shape}, expected [T]"
            )
        lengths = (len(raw_rpy_array), len(timestamps_array), len(valid_array))
        if len(set(lengths)) != 1:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={fragment_dir}: length mismatch "
                f"raw_rpy/timestamps/valid={lengths}"
            )
        if not lengths[0]:
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={fragment_dir}: empty neck sequence"
            )
        if not np.issubdtype(valid_array.dtype, np.bool_):
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={valid_path}: expected bool dtype, got {valid_array.dtype}"
            )
        if not np.all(np.isfinite(timestamps_array)) or not np.all(np.diff(timestamps_array) > 0):
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={timestamps_path}: timestamps must be finite and strictly increasing"
            )
        if not np.all(np.isfinite(raw_rpy_array[valid_array])):
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={rpy_path}: valid=True RPY contains non-finite values"
            )

        text = transcript.get("text")
        words = transcript.get("words")
        if not isinstance(text, str) or not isinstance(words, list):
            raise DatasetValidationError(
                f"fragment_id={fragment_id} path={transcript_path}: expected string text and list words"
            )
        for word_index, word in enumerate(words):
            if not isinstance(word, dict):
                raise DatasetValidationError(
                    f"fragment_id={fragment_id} path={transcript_path}: word[{word_index}] is not an object"
                )
            for timeline in ("local", "source"):
                start = word.get(f"{timeline}_start")
                end = word.get(f"{timeline}_end")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    raise DatasetValidationError(
                        f"fragment_id={fragment_id} path={transcript_path}: word[{word_index}] "
                        f"has missing/non-numeric {timeline}_start or {timeline}_end"
                    )
                if not np.isfinite(start) or not np.isfinite(end) or start > end:
                    raise DatasetValidationError(
                        f"fragment_id={fragment_id} path={transcript_path}: word[{word_index}] "
                        f"has invalid {timeline} interval [{start}, {end}]"
                    )

        # np.array(copy=True) prevents target construction or consumers from modifying artifacts/views.
        raw_rpy = torch.from_numpy(np.array(raw_rpy_array, dtype=np.float32, copy=True))
        neck_timestamps = torch.from_numpy(np.array(timestamps_array, dtype=np.float64, copy=True))
        valid_mask = torch.from_numpy(np.array(valid_array, dtype=np.bool_, copy=True))
        reference_valid = bool(valid_mask[0].item())
        target_rpy = raw_rpy - raw_rpy[0]
        target_valid_mask = valid_mask & reference_valid

        return {
            "fragment_id": fragment_id,
            "clip_id": str(row["clip_id"]),
            "source_video_id": str(row["source_video_id"]),
            "split": self.split,
            "audio": audio,
            "audio_length": int(audio.shape[0]),
            "text": text,
            "words": words,
            "neck_timestamps": neck_timestamps,
            "raw_rpy": raw_rpy,
            "target_rpy": target_rpy,
            "valid_mask": valid_mask,
            "target_valid_mask": target_valid_mask,
            "reference_valid": reference_valid,
            "metadata": {
                "source_start": row.get("source_start"),
                "source_end": row.get("source_end"),
                "duration": row.get("duration"),
                "feature_version": row.get("feature_version"),
                "audio_path": str(audio_path),
                "neck_rpy_path": str(rpy_path),
                "transcript_path": str(transcript_path),
            },
        }


def neck_motion_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad variable audio/neck axes while retaining separate sequence and GT masks."""
    if not samples:
        raise ValueError("cannot collate an empty batch")

    audio_lengths = torch.tensor([sample["audio_length"] for sample in samples], dtype=torch.long)
    neck_lengths = torch.tensor([len(sample["neck_timestamps"]) for sample in samples], dtype=torch.long)
    max_neck_length = int(neck_lengths.max().item())
    sequence_mask = torch.arange(max_neck_length).unsqueeze(0) < neck_lengths.unsqueeze(1)

    return {
        "fragment_id": [sample["fragment_id"] for sample in samples],
        "clip_id": [sample["clip_id"] for sample in samples],
        "source_video_id": [sample["source_video_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "audio": pad_sequence([sample["audio"] for sample in samples], batch_first=True),
        "audio_lengths": audio_lengths,
        "text": [sample["text"] for sample in samples],
        "words": [sample["words"] for sample in samples],
        "neck_lengths": neck_lengths,
        "neck_timestamps": pad_sequence(
            [sample["neck_timestamps"] for sample in samples],
            batch_first=True,
            padding_value=float("nan"),
        ),
        "raw_rpy": pad_sequence(
            [sample["raw_rpy"] for sample in samples],
            batch_first=True,
            padding_value=float("nan"),
        ),
        "target_rpy": pad_sequence(
            [sample["target_rpy"] for sample in samples],
            batch_first=True,
            padding_value=float("nan"),
        ),
        "sequence_mask": sequence_mask,
        "valid_mask": pad_sequence(
            [sample["valid_mask"] for sample in samples],
            batch_first=True,
            padding_value=False,
        ),
        "target_valid_mask": pad_sequence(
            [sample["target_valid_mask"] for sample in samples],
            batch_first=True,
            padding_value=False,
        ),
        "reference_valid": torch.tensor(
            [sample["reference_valid"] for sample in samples], dtype=torch.bool
        ),
        "metadata": [sample["metadata"] for sample in samples],
    }
