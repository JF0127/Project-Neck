"""Unit tests for Speech V1 timestamp/schema validation and resume checks."""

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from src.features.asr import (
    ASRError,
    FEATURE_VERSION,
    is_complete_speech_output,
    transcribe_audio,
    validate_transcript,
    write_json,
)


def transcript_fixture() -> dict:
    words = [
        {"text": "今天", "start": 0.1, "end": 0.4, "probability": 0.9, "segment_id": 0},
        {"text": "新闻", "start": 0.4, "end": 0.8, "probability": 0.8, "segment_id": 0},
    ]
    return {
        "clip_id": "clip_0000",
        "language": "zh",
        "text": "今天新闻",
        "segments": [
            {
                "id": 0,
                "start": 0.1,
                "end": 0.8,
                "text": "今天新闻",
                "words": [{key: value for key, value in word.items() if key != "segment_id"} for word in words],
            }
        ],
        "words": words,
    }


class FakeModel:
    def transcribe(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs
        words = [
            SimpleNamespace(word="今天", start=0.1, end=0.4, probability=0.95),
            SimpleNamespace(word="新闻", start=0.4, end=0.8, probability=0.90),
        ]
        segments = [SimpleNamespace(id=0, start=0.1, end=0.8, text="今天新闻", words=words)]
        info = SimpleNamespace(language="zh", language_probability=0.99)
        return iter(segments), info


class ASRValidationTests(unittest.TestCase):
    def test_mock_asr_preserves_word_units_and_configuration(self) -> None:
        model = FakeModel()
        transcript = transcribe_audio(model, Path("audio.wav"), "clip_0000")
        self.assertEqual([word["text"] for word in transcript["words"]], ["今天", "新闻"])
        self.assertEqual(model.kwargs["language"], "zh")
        self.assertEqual(model.kwargs["task"], "transcribe")
        self.assertTrue(model.kwargs["word_timestamps"])
        self.assertFalse(model.kwargs["vad_filter"])
        validate_transcript(transcript, 1.0)

    def test_valid_transcript_timestamps(self) -> None:
        validate_transcript(transcript_fixture(), 1.0)

    def test_word_timestamps_must_be_monotonic(self) -> None:
        transcript = transcript_fixture()
        transcript["words"][1]["start"] = 0.05
        with self.assertRaises(ASRError):
            validate_transcript(transcript, 1.0)

    def test_segment_timestamps_must_be_monotonic(self) -> None:
        transcript = transcript_fixture()
        transcript["segments"].append(
            {"id": 1, "start": 0.05, "end": 0.9, "text": "测试", "words": []}
        )
        with self.assertRaises(ASRError):
            validate_transcript(transcript, 1.0)

    def test_duration_boundary_is_checked(self) -> None:
        transcript = transcript_fixture()
        transcript["words"][-1]["end"] = 1.2
        with self.assertRaises(ASRError):
            validate_transcript(transcript, 1.0)

    def test_metadata_serialization_preserves_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            write_json(path, {"text": "中文测试", "status": "completed"})
            self.assertIn("中文测试", path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["text"], "中文测试")

    def test_resume_requires_complete_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip_0000"
            output.mkdir()
            with wave.open(str(output / "audio.wav"), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(b"\x00\x00" * 16000)
            write_json(output / "transcript.json", transcript_fixture())
            write_json(
                output / "metadata.json",
                {
                    "clip_id": "clip_0000",
                    "feature_version": FEATURE_VERSION,
                    "status": "completed",
                    "audio": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "duration": 1.0,
                    },
                    "asr": {"segment_count": 1, "word_count": 2},
                },
            )
            self.assertIsNotNone(is_complete_speech_output(output))
            (output / "transcript.json").unlink()
            self.assertIsNone(is_complete_speech_output(output))


if __name__ == "__main__":
    unittest.main()
