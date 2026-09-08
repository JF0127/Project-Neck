from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.asr import WhisperASR
from runtime.tts import EdgeTTS
from runtime.contracts import RobotSpeech, UserAudio, UserSpeech, WordTimestamp


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, waveform, **kwargs):
        self.calls.append({"waveform": waveform.copy(), **kwargs})
        words = [
            SimpleNamespace(word=" hello ", start=0.1, end=0.4),
            SimpleNamespace(word=" world ", start=0.5, end=0.8),
        ]
        segments = [SimpleNamespace(text=" hello world ", words=words)]
        return segments, SimpleNamespace(language="en")


class FakeEdgeCommunication:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    async def stream(self):
        for chunk in self.chunks:
            yield chunk


class ASRTests(unittest.TestCase):
    def test_pcm_conversion_uses_little_endian_int16(self) -> None:
        pcm = struct.pack("<hhh", -32768, 0, 32767)
        waveform = WhisperASR.pcm_to_float32(pcm)
        self.assertAlmostEqual(float(waveform[0]), -1.0)
        self.assertAlmostEqual(float(waveform[1]), 0.0)
        self.assertAlmostEqual(float(waveform[2]), 32767 / 32768)

    def test_transcribe_maps_user_audio_to_shared_user_speech(self) -> None:
        model = FakeWhisperModel()
        backend = WhisperASR("unused", model=model, language="en")
        pcm = struct.pack("<" + "h" * 160, *([100] * 160))
        audio = UserAudio(
            pcm_s16le=pcm,
            sample_rate=16_000,
            channels=1,
            duration_sec=0.01,
        )

        speech = backend.transcribe(audio)

        self.assertIsInstance(speech, UserSpeech)
        self.assertEqual(speech.text, "hello world")
        self.assertEqual(speech.language, "en")
        self.assertEqual(
            speech.words,
            (
                WordTimestamp("hello", 0.1, 0.4),
                WordTimestamp("world", 0.5, 0.8),
            ),
        )
        self.assertTrue(model.calls[0]["word_timestamps"])
        self.assertFalse(model.calls[0]["vad_filter"])

    def test_transcribe_rejects_nonstandard_user_audio(self) -> None:
        backend = WhisperASR("unused", model=FakeWhisperModel())
        audio = UserAudio(b"\x00\x00", 8_000, 1, 1 / 8_000)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            backend.transcribe(audio)


class TTSTests(unittest.TestCase):
    def test_synthesize_returns_shared_robot_speech(self) -> None:
        chunks = [
            {"type": "audio", "data": b"encoded"},
            {
                "type": "WordBoundary",
                "text": "hello",
                "offset": 1_000_000,
                "duration": 2_000_000,
            },
        ]
        calls: list[tuple] = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeEdgeCommunication(chunks)

        pcm = b"\x01\x00" * 320
        tts = EdgeTTS("test-voice", communicate_factory=factory)
        with patch.object(EdgeTTS, "_decode_audio", return_value=pcm):
            result = asyncio.run(tts.synthesize("  robot text  "))

        self.assertIsInstance(result, RobotSpeech)
        self.assertEqual(result.text, "robot text")
        self.assertEqual(result.pcm_s16le, pcm)
        self.assertEqual(result.duration_sec, 0.02)
        self.assertEqual(result.words[0].text, "hello")
        self.assertAlmostEqual(result.words[0].start_sec, 0.1)
        self.assertAlmostEqual(result.words[0].end_sec, 0.3)
        self.assertIsInstance(result.words[0], WordTimestamp)
        self.assertEqual(calls[0][0], ("robot text", "test-voice"))
        self.assertEqual(calls[0][1], {"boundary": "WordBoundary"})


if __name__ == "__main__":
    unittest.main()
