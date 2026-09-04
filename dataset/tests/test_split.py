"""Tests for deterministic source_video_id-grouped Split V1 manifests."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from src.splits.split_v1 import (
    SplitError,
    allocate_source_counts,
    build_split_dataset,
    validate_ratios,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def records_for_sources(count: int, fragments_per_source: int = 2) -> tuple[list[dict], list[dict]]:
    fragments: list[dict] = []
    clips: list[dict] = []
    for source_index in range(count):
        source_id = f"source-{source_index:02d}"
        clip_id = f"{source_id}-clip"
        clips.append({"source_video_id": source_id, "clip_id": clip_id})
        for fragment_index in range(fragments_per_source):
            fragments.append(
                {
                    "fragment_id": f"{clip_id}-fragment-{fragment_index}",
                    "clip_id": clip_id,
                    "source_video_id": source_id,
                    "source_start": float(fragment_index * 2),
                    "source_end": float(fragment_index * 2 + source_index + 1),
                    "duration": float(source_index + 1),
                    "text": f"文本 {source_index}-{fragment_index}",
                    "audio_path": f"/reference/{source_id}/{fragment_index}/audio.wav",
                    "feature_version": "fragment_v1_2_1",
                }
            )
    return fragments, clips


class SplitV1Tests(unittest.TestCase):
    def _input(self, root: Path, count: int = 10) -> tuple[Path, list[dict], list[dict]]:
        input_root = root / "input"
        fragments, clips = records_for_sources(count)
        write_jsonl(input_root / "fragments.jsonl", fragments)
        write_jsonl(input_root / "clips.jsonl", clips)
        return input_root, fragments, clips

    def _build(self, root: Path, count: int = 10, **kwargs):
        input_root, fragments, clips = self._input(root, count)
        output_root = root / "output"
        summary = build_split_dataset(input_root, output_root, **kwargs)
        outputs = {
            name: read_jsonl(output_root / f"{name}.jsonl")
            for name in ("train", "val", "test")
        }
        mapping = json.loads((output_root / "source_split.json").read_text(encoding="utf-8"))
        return fragments, clips, outputs, mapping, summary

    def test_same_source_fragments_always_share_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, outputs, _, _ = self._build(Path(temporary))
            source_splits: dict[str, set[str]] = {}
            for name, records in outputs.items():
                for record in records:
                    source_splits.setdefault(record["source_video_id"], set()).add(name)
                    self.assertEqual(record["split"], name)
            self.assertTrue(all(len(names) == 1 for names in source_splits.values()))

    def test_source_sets_have_no_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, outputs, _, _ = self._build(Path(temporary))
            sets = {
                name: {record["source_video_id"] for record in records}
                for name, records in outputs.items()
            }
            self.assertFalse(sets["train"] & sets["val"])
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["val"] & sets["test"])

    def test_fragments_have_no_overlap_or_omission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, _, outputs, _, _ = self._build(Path(temporary))
            expected = {record["fragment_id"] for record in original}
            sets = {
                name: {record["fragment_id"] for record in records}
                for name, records in outputs.items()
            }
            self.assertFalse(sets["train"] & sets["val"])
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["val"] & sets["test"])
            self.assertEqual(set().union(*sets.values()), expected)
            self.assertEqual(sum(map(len, sets.values())), len(expected))

    def test_same_seed_produces_identical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, _, _ = self._input(root)
            first = root / "first"
            second = root / "second"
            build_split_dataset(input_root, first, seed=42)
            build_split_dataset(input_root, second, seed=42)
            for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            first_map = json.loads((first / "source_split.json").read_text(encoding="utf-8"))
            second_map = json.loads((second / "source_split.json").read_text(encoding="utf-8"))
            self.assertEqual(first_map["source_to_split"], second_map["source_to_split"])

    def test_different_seed_changes_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, _, _ = self._input(root, 20)
            build_split_dataset(input_root, root / "seed-42", seed=42)
            build_split_dataset(input_root, root / "seed-43", seed=43)
            mappings = []
            for name in ("seed-42", "seed-43"):
                payload = json.loads((root / name / "source_split.json").read_text())
                mappings.append(payload["source_to_split"])
            self.assertNotEqual(*mappings)

    def test_shuffled_input_order_does_not_change_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments, clips = records_for_sources(12)
            first_input = root / "ordered"
            second_input = root / "shuffled"
            write_jsonl(first_input / "fragments.jsonl", fragments)
            write_jsonl(first_input / "clips.jsonl", clips)
            shuffled_fragments = list(fragments)
            shuffled_clips = list(clips)
            random.Random(99).shuffle(shuffled_fragments)
            random.Random(100).shuffle(shuffled_clips)
            write_jsonl(second_input / "fragments.jsonl", shuffled_fragments)
            write_jsonl(second_input / "clips.jsonl", shuffled_clips)
            build_split_dataset(first_input, root / "first", seed=42)
            build_split_dataset(second_input, root / "second", seed=42)
            for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual(
                    (root / "first" / filename).read_bytes(),
                    (root / "second" / filename).read_bytes(),
                )
            mappings = []
            for output in ("first", "second"):
                payload = json.loads((root / output / "source_split.json").read_text())
                mappings.append(payload["source_to_split"])
            self.assertEqual(*mappings)

    def test_largest_remainder_allocation_for_57_sources(self) -> None:
        self.assertEqual(
            allocate_source_counts(57), {"train": 40, "test": 11, "val": 6}
        )

    def test_small_source_counts_are_legal_and_sum_exactly(self) -> None:
        expected = {
            3: {"train": 2, "test": 1, "val": 0},
            5: {"train": 4, "test": 1, "val": 0},
            7: {"train": 5, "test": 1, "val": 1},
        }
        for count, allocation in expected.items():
            with self.subTest(count=count):
                actual = allocate_source_counts(count)
                self.assertEqual(actual, allocation)
                self.assertEqual(sum(actual.values()), count)

    def test_invalid_ratio_sum_and_nonpositive_ratio_raise(self) -> None:
        with self.assertRaises(SplitError):
            validate_ratios(0.7, 0.2, 0.2)
        with self.assertRaises(SplitError):
            validate_ratios(0.8, 0.2, 0.0)

    def test_missing_source_video_id_does_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments, clips = records_for_sources(3)
            fragments[0].pop("source_video_id")
            write_jsonl(root / "input/fragments.jsonl", fragments)
            write_jsonl(root / "input/clips.jsonl", clips)
            with self.assertRaisesRegex(SplitError, "source_video_id"):
                build_split_dataset(root / "input", root / "output")

    def test_duplicate_fragment_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments, clips = records_for_sources(3)
            fragments[1]["fragment_id"] = fragments[0]["fragment_id"]
            write_jsonl(root / "input/fragments.jsonl", fragments)
            write_jsonl(root / "input/clips.jsonl", clips)
            with self.assertRaisesRegex(SplitError, "duplicate fragment_id"):
                build_split_dataset(root / "input", root / "output")

    def test_clips_manifest_mapping_conflict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments, clips = records_for_sources(3)
            clips[0]["source_video_id"] = "source-01"
            write_jsonl(root / "input/fragments.jsonl", fragments)
            write_jsonl(root / "input/clips.jsonl", clips)
            with self.assertRaisesRegex(SplitError, "conflicts with fragments"):
                build_split_dataset(root / "input", root / "output")

    def test_summary_counts_durations_and_ratios_are_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fragments, clips, outputs, _, summary = self._build(Path(temporary), 10)
            self.assertEqual(summary["source_videos"], {"total": 10, "train": 7, "val": 1, "test": 2})
            self.assertEqual(summary["clips"]["total"], len(clips))
            self.assertEqual(summary["fragments"]["total"], len(fragments))
            self.assertEqual(
                sum(summary["fragments"][name] for name in ("train", "val", "test")),
                len(fragments),
            )
            expected_duration = sum(record["duration"] for record in fragments)
            self.assertEqual(summary["duration_seconds"]["total"], expected_duration)
            self.assertAlmostEqual(summary["duration_hours"]["total"], expected_duration / 3600)
            self.assertEqual(
                summary["leakage_check"],
                {"source_overlap": False, "fragment_overlap": False, "missing_fragments": 0},
            )
            output_count = sum(len(records) for records in outputs.values())
            self.assertEqual(output_count, len(fragments))

    def test_output_preserves_metadata_and_only_adds_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fragments, _, outputs, _, _ = self._build(Path(temporary), 3)
            by_id = {
                record["fragment_id"]: record
                for records in outputs.values()
                for record in records
            }
            for original in fragments:
                emitted = dict(by_id[original["fragment_id"]])
                self.assertIn(emitted.pop("split"), {"train", "val", "test"})
                self.assertEqual(emitted, original)

    def test_existing_output_requires_force_and_force_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, _, _ = self._input(root)
            output_root = root / "output"
            build_split_dataset(input_root, output_root)
            original = (output_root / "source_split.json").read_bytes()
            with self.assertRaisesRegex(SplitError, "--force"):
                build_split_dataset(input_root, output_root)
            (output_root / "stale.txt").write_text("stale", encoding="utf-8")
            build_split_dataset(input_root, output_root, force=True)
            self.assertFalse((output_root / "stale.txt").exists())
            self.assertEqual((output_root / "source_split.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
