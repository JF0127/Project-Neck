from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.data import NeckMotionDataset, neck_motion_collate


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "datasets" / "zhubo_shuo_lianbo"


def _write_fragment(
    root: Path,
    fragment_id: str,
    audio_length: int,
    raw_rpy: np.ndarray,
    valid: np.ndarray,
) -> dict:
    fragment_dir = root / "fragments" / "fragment_v1" / "fragments" / fragment_id
    fragment_dir.mkdir(parents=True)
    audio_path = fragment_dir / "audio.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.arange(audio_length, dtype=np.int16).astype("<i2").tobytes())

    rpy_path = fragment_dir / "neck_rpy.npy"
    np.save(rpy_path, raw_rpy.astype(np.float32))
    np.save(fragment_dir / "neck_timestamps.npy", np.arange(len(raw_rpy), dtype=np.float64) * 0.04)
    np.save(fragment_dir / "neck_valid.npy", valid.astype(np.bool_))
    transcript_path = fragment_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "fragment_id": fragment_id,
                "text": "测试",
                "words": [
                    {
                        "text": "测试",
                        "local_start": 0.0,
                        "local_end": 0.1,
                        "source_start": 1.0,
                        "source_end": 1.1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "fragment_id": fragment_id,
        "clip_id": fragment_id.rsplit("_f", 1)[0],
        "source_video_id": f"source-{fragment_id}",
        "audio_path": str(audio_path),
        "neck_rpy_path": str(rpy_path),
        "transcript_path": str(transcript_path),
        "source_start": 1.0,
        "source_end": 1.0 + audio_length / 16_000,
        "duration": audio_length / 16_000,
        "feature_version": "fragment_v1_2_1",
        "split": "train",
    }


def _write_manifest(root: Path, rows: list[dict]) -> None:
    split_dir = root / "splits" / "split_v1"
    split_dir.mkdir(parents=True)
    (split_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@unittest.skipUnless(DATASET_ROOT.exists(), "canonical Dataset V1 is not present")
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = NeckMotionDataset(split="train", dataset_root=DATASET_ROOT)

    def test_real_fragment_schema_and_shapes(self) -> None:
        sample = self.dataset[0]
        expected = {
            "audio",
            "text",
            "words",
            "raw_rpy",
            "target_rpy",
            "neck_timestamps",
            "valid_mask",
        }
        self.assertTrue(expected <= sample.keys())
        self.assertEqual(sample["audio"].ndim, 1)
        self.assertEqual(sample["audio_length"], len(sample["audio"]))
        self.assertIsInstance(sample["text"], str)
        self.assertIsInstance(sample["words"], list)
        self.assertEqual(sample["raw_rpy"].shape[1:], (3,))
        self.assertEqual(sample["target_rpy"].shape, sample["raw_rpy"].shape)
        self.assertEqual(sample["neck_timestamps"].shape, sample["valid_mask"].shape)
        self.assertEqual(len(sample["raw_rpy"]), len(sample["neck_timestamps"]))

    def test_first_valid_reference_makes_zero_target(self) -> None:
        sample = next(item for item in self.dataset if item["reference_valid"])
        torch.testing.assert_close(sample["target_rpy"][0], torch.zeros(3))

    def test_raw_rpy_is_unchanged(self) -> None:
        sample = self.dataset[0]
        canonical = np.load(sample["metadata"]["neck_rpy_path"], allow_pickle=False)
        np.testing.assert_array_equal(sample["raw_rpy"].numpy(), canonical)

    def test_canonical_splits_have_no_source_overlap(self) -> None:
        datasets = {
            split: NeckMotionDataset(split=split, dataset_root=DATASET_ROOT)
            for split in ("train", "val", "test")
        }
        self.assertEqual({split: len(ds) for split, ds in datasets.items()}, {"train": 314, "val": 42, "test": 115})
        sources = {
            split: {row["source_video_id"] for row in ds.manifest_rows}
            for split, ds in datasets.items()
        }
        self.assertTrue(sources["train"].isdisjoint(sources["val"]))
        self.assertTrue(sources["train"].isdisjoint(sources["test"]))
        self.assertTrue(sources["val"].isdisjoint(sources["test"]))


class SyntheticDatasetTests(unittest.TestCase):
    def test_collate_variable_lengths_and_dataloader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = [
                _write_fragment(
                    root,
                    "clip_a_f0000",
                    5,
                    np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32),
                    np.array([True, True]),
                ),
                _write_fragment(
                    root,
                    "clip_b_f0000",
                    8,
                    np.array([[0, 0, 0], [1, 1, 1], [np.nan, np.nan, np.nan]], dtype=np.float32),
                    np.array([True, True, False]),
                ),
            ]
            _write_manifest(root, rows)
            dataset = NeckMotionDataset("train", root)
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=neck_motion_collate)))

            self.assertEqual(batch["audio"].shape, (2, 8))
            torch.testing.assert_close(batch["audio_lengths"], torch.tensor([5, 8]))
            self.assertEqual(batch["target_rpy"].shape, (2, 3, 3))
            self.assertEqual(batch["neck_timestamps"].shape, (2, 3))
            torch.testing.assert_close(
                batch["sequence_mask"],
                torch.tensor([[True, True, False], [True, True, True]]),
            )
            torch.testing.assert_close(
                batch["valid_mask"],
                torch.tensor([[True, True, False], [True, True, False]]),
            )
            self.assertTrue(torch.isnan(batch["target_rpy"][0, 2]).all())
            self.assertTrue(torch.isnan(batch["neck_timestamps"][0, 2]))
            self.assertTrue(torch.equal(batch["audio"][0, 5:], torch.zeros(3)))

    def test_invalid_frame_is_not_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = [
                _write_fragment(
                    root,
                    "clip_invalid_f0000",
                    4,
                    np.array([[0, 0, 0], [np.nan, np.nan, np.nan], [1, 2, 3]], dtype=np.float32),
                    np.array([True, False, True]),
                )
            ]
            _write_manifest(root, rows)
            sample = NeckMotionDataset("train", root)[0]
            self.assertEqual(len(sample["raw_rpy"]), 3)
            self.assertEqual(len(sample["neck_timestamps"]), 3)
            torch.testing.assert_close(sample["valid_mask"], torch.tensor([True, False, True]))
            self.assertTrue(torch.isnan(sample["raw_rpy"][1]).all())

    def test_invalid_first_reference_has_no_implicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = [
                _write_fragment(
                    root,
                    "clip_reference_f0000",
                    4,
                    np.array([[np.nan, np.nan, np.nan], [1, 2, 3]], dtype=np.float32),
                    np.array([False, True]),
                )
            ]
            _write_manifest(root, rows)
            sample = NeckMotionDataset("train", root)[0]
            self.assertFalse(sample["reference_valid"])
            self.assertTrue(torch.isnan(sample["target_rpy"]).all())
            self.assertFalse(sample["target_valid_mask"].any())
            torch.testing.assert_close(sample["valid_mask"], torch.tensor([False, True]))


if __name__ == "__main__":
    unittest.main()
