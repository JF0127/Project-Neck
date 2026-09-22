"""Pure-software tests for explainable PCM prosody features."""
from __future__ import annotations

from array import array
import unittest

from .prosody import extract_prosody
from .speech_alignment import SpeechAlignment, SpeechSegment

SAMPLE_RATE = 16_000


def _pcm(parts: list[tuple[float, int]]) -> bytes:
    samples: list[int] = []
    for duration, amplitude in parts:
        samples.extend([amplitude] * round(duration * SAMPLE_RATE))
    return array("h", samples).tobytes()


def _alignment(*segments: SpeechSegment) -> SpeechAlignment:
    return SpeechAlignment(tuple(segments), "test", 0.0)


class ProsodyTest(unittest.TestCase):
    def test_silence_has_zero_energy(self) -> None:
        result = extract_prosody(
            _pcm([(1.0, 0)]),
            SAMPLE_RATE,
            _alignment(SpeechSegment("测试", 0.0, 1.0)),
        )
        segment = result.segments[0]
        self.assertEqual(segment.rms_mean, 0.0)
        self.assertEqual(segment.rms_peak, 0.0)
        self.assertEqual(segment.relative_energy, 0.0)
        self.assertEqual(segment.energy_level, "low")

    def test_constant_amplitude_rms_and_rate(self) -> None:
        result = extract_prosody(
            _pcm([(1.0, 4096)]),
            SAMPLE_RATE,
            _alignment(SpeechSegment("你好，A1！", 0.0, 1.0)),
        )
        segment = result.segments[0]
        self.assertAlmostEqual(segment.rms_mean, 4096 / 32768, places=6)
        self.assertAlmostEqual(segment.rms_peak, 4096 / 32768, places=6)
        self.assertAlmostEqual(segment.relative_energy, 1.0, places=6)
        self.assertEqual(segment.speech_rate_chars_sec, 4.0)
        self.assertEqual(segment.rate_level, "normal")

    def test_relative_energy_orders_two_segments(self) -> None:
        result = extract_prosody(
            _pcm([(0.5, 1000), (0.5, 4000)]),
            SAMPLE_RATE,
            _alignment(
                SpeechSegment("低", 0.0, 0.5),
                SpeechSegment("高", 0.5, 1.0),
            ),
        )
        low, high = result.segments
        self.assertLess(low.rms_mean, high.rms_mean)
        self.assertLess(low.relative_energy, high.relative_energy)
        self.assertEqual(low.energy_level, "low")
        self.assertEqual(high.energy_level, "high")

    def test_pause_before_and_after_follow_alignment_gaps(self) -> None:
        result = extract_prosody(
            _pcm([(2.0, 2000)]),
            SAMPLE_RATE,
            _alignment(
                SpeechSegment("前", 0.2, 0.7),
                SpeechSegment("后", 1.0, 1.6),
            ),
        )
        first, second = result.segments
        self.assertAlmostEqual(first.pause_before_sec, 0.2)
        self.assertAlmostEqual(first.pause_after_sec, 0.3)
        self.assertAlmostEqual(second.pause_before_sec, 0.3)
        self.assertAlmostEqual(second.pause_after_sec, 0.4)
        self.assertGreaterEqual(first.energy_peak_time_sec, first.start_sec)
        self.assertLessEqual(first.energy_peak_time_sec, first.end_sec)


if __name__ == "__main__":
    unittest.main()
