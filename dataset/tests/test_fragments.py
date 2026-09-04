"""Tests for Fragment V1.2.1 lookahead and unchanged multimodal slicing."""

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.fragments.fragment_v1 import (
    FEATURE_VERSION,
    FragmentConfig,
    build_fragments_from_transcript,
    build_utterance_units,
    dataset_statistics,
    is_complete_clip_output,
    process_fragment_clip,
    slice_neck_arrays,
    slice_pcm_wav,
    source_video_id_for_clip,
    validate_fragment_overlap,
)


def word(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end, "probability": 0.95}


def segment(segment_id: int, words: list[dict], text: str | None = None) -> dict:
    return {
        "id": segment_id,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text if text is not None else "".join(item["text"] for item in words),
        "words": words,
    }


def transcript(segments: list[dict], clip_id: str = "video_0000") -> dict:
    flat_words = [
        {**item, "segment_id": item_segment["id"]}
        for item_segment in segments
        for item in item_segment["words"]
    ]
    return {
        "clip_id": clip_id,
        "text": "".join(str(item["text"]) for item in segments),
        "words": flat_words,
        "segments": segments,
    }


def one_word_segment(segment_id: int, text: str, start: float, end: float) -> dict:
    return segment(segment_id, [word(text, start, end)])


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class UtteranceGroupingTests(unittest.TestCase):
    def test_twenty_four_seconds_waits_for_strong_pause(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "A", 0.0, 8.0),
                one_word_segment(2, "B", 8.1, 16.0),
                one_word_segment(3, "C", 16.2, 24.0),
                one_word_segment(4, "D", 24.5, 27.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 27.2)
        self.assertEqual(fragments[0].segment_ids, (1, 2, 3))
        self.assertAlmostEqual(fragments[0].speech_duration, 24.0)
        self.assertEqual(fragments[0].split_reason, "strong_pause")
        self.assertTrue(fragments[0].natural_boundary)
        self.assertTrue(fragments[0].crossed_soft_threshold)

    def test_twenty_one_seconds_with_small_gap_keeps_looking_ahead(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "A", 0.0, 12.0),
                one_word_segment(2, "B", 12.1, 21.0),
                one_word_segment(3, "C", 21.1, 25.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 25.2)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0].segment_ids, (1, 2, 3))
        self.assertAlmostEqual(fragments[0].speech_duration, 25.0)

    def test_twenty_seven_seconds_can_end_at_natural_pause(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "A", 0.0, 10.0),
                one_word_segment(2, "B", 10.1, 20.0),
                one_word_segment(3, "C", 20.2, 27.0),
                one_word_segment(4, "D", 27.5, 29.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 29.2)
        self.assertEqual(fragments[0].split_reason, "strong_pause")
        self.assertEqual(fragments[0].core_end, 27.0)
        self.assertTrue(fragments[0].natural_boundary)
        self.assertTrue(fragments[0].crossed_soft_threshold)

    def test_strong_pause_before_soft_threshold_is_unchanged(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "前段", 0.0, 5.0),
                one_word_segment(2, "后段", 5.50, 7.5),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 8.0)
        self.assertEqual(len(fragments), 2)
        self.assertEqual(fragments[0].split_reason, "strong_pause")

    def test_absolute_max_fallback_uses_only_segment_boundary(self) -> None:
        payload = transcript(
            [
                segment(1, [word("产业", 0.0, 5.0), word("升级", 5.0, 10.0)]),
                one_word_segment(2, "第二段", 10.1, 20.0),
                one_word_segment(3, "第三段", 20.2, 28.0),
                segment(4, [word("240", 28.3, 31.0), word("家", 31.0, 34.0)]),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 34.2)
        self.assertEqual(fragments[0].segment_ids, (1, 2, 3))
        self.assertEqual(
            fragments[0].split_reason, "absolute_max_pause_fallback"
        )
        self.assertEqual([item["text"] for item in fragments[0].words], [
            "产业", "升级", "第二段", "第三段"
        ])
        self.assertEqual([item["text"] for item in fragments[1].words], ["240", "家"])

    def test_absolute_fallback_prefers_maximum_historical_gap(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "一", 0.0, 10.0),
                one_word_segment(2, "二", 10.1, 20.0),
                one_word_segment(3, "三", 20.4, 28.0),
                one_word_segment(4, "四", 28.2, 34.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 34.2)
        self.assertEqual(fragments[0].segment_ids, (1, 2))
        self.assertEqual(fragments[0].core_end, 20.0)
        self.assertEqual(
            fragments[0].split_reason, "absolute_max_pause_fallback"
        )

    def test_absolute_fallback_uses_plain_segment_boundary_as_last_resort(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "一", 0.0, 10.0),
                one_word_segment(2, "二", 10.0, 20.0),
                one_word_segment(3, "三", 20.0, 28.0),
                one_word_segment(4, "四", 28.0, 34.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 34.2)
        self.assertEqual(fragments[0].segment_ids, (1, 2, 3))
        self.assertEqual(
            fragments[0].split_reason, "absolute_max_segment_fallback"
        )

    def test_absolute_fallback_uses_punctuation_when_gaps_are_zero(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "一", 0.0, 10.0),
                segment(2, [word("二", 10.0, 20.0)], text="二。"),
                one_word_segment(3, "三", 20.0, 28.0),
                one_word_segment(4, "四", 28.0, 34.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 34.2)
        self.assertEqual(fragments[0].segment_ids, (1, 2))
        self.assertEqual(
            fragments[0].split_reason, "absolute_max_punctuation_fallback"
        )

    def test_single_normal_segment_under_thirty_is_never_word_split(self) -> None:
        words = [word("产业升级" if index == 14 else f"词{index}", float(index), float(index + 1)) for index in range(29)]
        payload = transcript([segment(7, words)])
        _, fragments = build_fragments_from_transcript(payload, 29.2)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(len(fragments[0].words), 29)
        self.assertEqual(fragments[0].split_reason, "clip_end")

    def test_only_segment_over_thirty_uses_internal_word_fallback(self) -> None:
        words = [word(f"词{index}", float(index), float(index + 1)) for index in range(35)]
        payload = transcript([segment(9, words)])
        _, fragments = build_fragments_from_transcript(payload, 35.2)
        self.assertEqual(len(fragments), 2)
        self.assertEqual(
            fragments[0].split_reason, "oversized_segment_word_fallback"
        )
        self.assertTrue(15.0 <= fragments[0].core_end <= 22.0)
        self.assertTrue(all(item.segment_ids == (9,) for item in fragments))
        self.assertEqual(sum(len(item.words) for item in fragments), len(words))

    def test_short_segment_merges_forward(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "好的", 0.0, 0.8),
                one_word_segment(2, "我们继续", 0.9, 4.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 4.2)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0].segment_count, 2)
        self.assertEqual(fragments[0].split_reason, "short_merge")
        self.assertFalse(fragments[0].duration_exception)

    def test_long_silence_allows_short_residual(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "短句", 0.0, 0.8),
                one_word_segment(2, "后句", 2.8, 5.0),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 5.2)
        self.assertEqual(len(fragments), 2)
        self.assertTrue(fragments[0].duration_exception)
        self.assertEqual(
            fragments[0].duration_exception_reason, "short_residual"
        )

    def test_padding_does_not_create_large_overlap(self) -> None:
        payload = transcript(
            [
                one_word_segment(1, "第一段", 0.1, 3.1),
                one_word_segment(2, "第二段", 3.6, 7.6),
            ]
        )
        _, fragments = build_fragments_from_transcript(payload, 8.0)
        self.assertEqual(validate_fragment_overlap(fragments), [])
        self.assertLessEqual(
            fragments[0].source_end, fragments[1].source_start + 2 / 16000
        )


class SlicingTests(unittest.TestCase):
    def test_audio_sample_slicing_is_exact_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_samples = np.arange(1600, dtype=np.int16)
            write_wav(root / "source.wav", source_samples)
            info = slice_pcm_wav(root / "source.wav", root / "fragment.wav", 0.01, 0.04)
            self.assertEqual(info["sample_count"], 480)
            with wave.open(str(root / "fragment.wav"), "rb") as source:
                output = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            np.testing.assert_array_equal(output, source_samples[160:640])

    def test_neck_timestamp_slicing_and_local_conversion(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        valid = np.array([True, False, True, True, True])
        rpy = np.arange(15, dtype=np.float32).reshape(5, 3)
        rpy[~valid] = np.nan
        sliced_rpy, local, sliced_valid, frame_range = slice_neck_arrays(
            times, valid, rpy, 0.15, 0.36
        )
        self.assertEqual(frame_range, (2, 4))
        np.testing.assert_allclose(local, [0.05, 0.15])
        np.testing.assert_array_equal(sliced_rpy, rpy[2:4])
        np.testing.assert_array_equal(sliced_valid, valid[2:4])


class MetadataSummaryAndResumeTests(unittest.TestCase):
    def _build_one_clip(self, root: Path):
        clip_id = "youtube-id_0000"
        speech = root / "speech" / clip_id
        neck = root / "neck" / clip_id
        output = root / "output" / "fragments"
        speech.mkdir(parents=True)
        neck.mkdir(parents=True)
        write_wav(speech / "audio.wav", np.zeros(25 * 16000, dtype=np.int16))
        payload = transcript(
            [
                one_word_segment(3, "今天", 0.20, 1.0),
                one_word_segment(4, "新闻", 1.0, 24.0),
            ],
            clip_id,
        )
        (speech / "transcript.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (speech / "metadata.json").write_text(
            json.dumps(
                {"clip_id": clip_id, "feature_version": "speech_v1", "status": "completed"}
            ),
            encoding="utf-8",
        )
        times = np.arange(0.0, 25.0, 0.04, dtype=np.float64)
        valid = np.ones(len(times), dtype=np.bool_)
        valid[20] = False
        rpy = np.zeros((len(times), 3), dtype=np.float32)
        rpy[~valid] = np.nan
        np.save(neck / "timestamps.npy", times)
        np.save(neck / "valid.npy", valid)
        np.save(neck / "neck_rpy.npy", rpy)
        (neck / "metadata.json").write_text(
            json.dumps(
                {
                    "clip_id": clip_id,
                    "frame_count": len(times),
                    "feature_version": "neck_pose_v1",
                    "status": "completed",
                }
            ),
            encoding="utf-8",
        )
        clip_record, records = process_fragment_clip(
            speech, neck, output, "youtube-id", "clean_v1_metadata", FragmentConfig()
        )
        return output, clip_record, records

    def test_segment_metadata_and_word_source_local_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, clip_record, records = self._build_one_clip(Path(temporary))
            fragment_id = records[0]["fragment_id"]
            fragment_dir = output / fragment_id
            payload = json.loads((fragment_dir / "transcript.json").read_text(encoding="utf-8"))
            metadata = json.loads((fragment_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["segment_ids"], [3, 4])
            self.assertEqual(payload["segment_count"], 2)
            self.assertEqual(metadata["fragment_type"], "spoken_utterance")
            self.assertEqual(metadata["segment_ids"], [3, 4])
            self.assertEqual(metadata["feature_version"], "fragment_v1_2_1")
            self.assertTrue(metadata["natural_boundary"])
            self.assertTrue(metadata["crossed_soft_threshold"])
            self.assertTrue(records[0]["natural_boundary"])
            self.assertTrue(records[0]["crossed_soft_threshold"])
            self.assertEqual(payload["words"][0]["source_start"], 0.2)
            self.assertAlmostEqual(payload["words"][0]["local_start"], 0.1)
            self.assertEqual(payload["words"][0]["segment_id"], 3)
            self.assertEqual(metadata["source_video_id"], "youtube-id")
            self.assertEqual(clip_record["source_video_id"], "youtube-id")

    def test_summary_grouping_statistics(self) -> None:
        records = [
            {
                "duration": 24.0,
                "neck_valid_ratio": 1.0,
                "segment_count": 3,
                "split_reason": "strong_pause",
                "end_reason": "strong_pause",
                "natural_boundary": True,
                "crossed_soft_threshold": True,
            },
            {
                "duration": 27.0,
                "neck_valid_ratio": 0.8,
                "segment_count": 3,
                "split_reason": "clip_end",
                "end_reason": "clip_end",
                "natural_boundary": True,
                "crossed_soft_threshold": True,
            },
            {
                "duration": 22.0,
                "neck_valid_ratio": 1.0,
                "segment_count": 2,
                "split_reason": "absolute_max_segment_fallback",
                "end_reason": "absolute_max_segment_fallback",
                "natural_boundary": False,
                "crossed_soft_threshold": True,
            },
        ]
        summary = dataset_statistics(
            [], records, input_clips=1, processed_clips=1, failed_clips=0, skipped_clips=0
        )
        grouping = summary["grouping"]
        self.assertEqual(grouping["single_segment_fragments"], 0)
        self.assertEqual(grouping["multi_segment_fragments"], 3)
        self.assertAlmostEqual(grouping["mean_segments_per_fragment"], 8 / 3)
        self.assertEqual(grouping["median_segments_per_fragment"], 3)
        self.assertEqual(summary["oversized_segment_word_fallback_count"], 0)
        self.assertEqual(summary["crossed_soft_threshold_fragments"], 3)
        self.assertEqual(summary["soft_threshold_resolved_by_natural_boundary"], 2)
        self.assertEqual(summary["absolute_max_fallback_fragments"], 1)
        self.assertEqual(summary["natural_boundary_fragments"], 2)
        self.assertEqual(summary["fallback_boundary_fragments"], 1)
        self.assertAlmostEqual(summary["natural_boundary_ratio"], 2 / 3)
        self.assertEqual(
            list(summary["duration"]["ranges"]),
            [
                "<2s", "2-3s", "3-5s", "5-8s", "8-12s",
                "12-16s", "16-20s", "20-25s", "25-30s", ">30s",
            ],
        )

    def test_source_video_id_fallback_is_explicit(self) -> None:
        source_id, method = source_video_id_for_clip("abc_def_0007", {})
        self.assertEqual(source_id, "abc_def")
        self.assertEqual(method, "clip_id_suffix_fallback")

    def test_resume_requires_v121_manifest_and_all_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, clip_record, records = self._build_one_clip(Path(temporary))
            by_id = {record["fragment_id"]: record for record in records}
            self.assertEqual(clip_record["feature_version"], FEATURE_VERSION)
            self.assertTrue(is_complete_clip_output(clip_record, by_id, output))
            fragment_id = records[0]["fragment_id"]
            (output / fragment_id / "neck_valid.npy").unlink()
            self.assertFalse(is_complete_clip_output(clip_record, by_id, output))

    def test_neck_rpy_is_an_unchanged_slice(self) -> None:
        times = np.arange(5, dtype=np.float64) * 0.1
        valid = np.array([True, True, False, True, True])
        rpy = np.arange(15, dtype=np.float32).reshape(5, 3)
        rpy[2] = np.nan
        sliced, _, sliced_valid, _ = slice_neck_arrays(times, valid, rpy, 0.1, 0.4)
        np.testing.assert_array_equal(sliced, rpy[1:4])
        np.testing.assert_array_equal(sliced_valid, valid[1:4])


if __name__ == "__main__":
    unittest.main()
