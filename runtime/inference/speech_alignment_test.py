"""Pure software checks for the legacy, standalone PCM phrase aligner."""
from __future__ import annotations

from array import array
import math
import unittest

from .speech_alignment import align_speech, split_phrases

_RATE = 16_000


def _silence(seconds):
    return [0] * round(seconds * _RATE)


def _tone(seconds):
    return [round(9000 * math.sin(2 * math.pi * 220 * i / _RATE))
            for i in range(round(seconds * _RATE))]


class SpeechAlignmentTest(unittest.TestCase):
    def test_chinese_phrase_splitting_is_conservative(self):
        cases = {
            "好的，我们先确认这个方案，然后再继续。": ("好的", "我们先确认这个方案", "然后再继续"),
            "这个方法确实不错，但是还有一个问题。": ("这个方法确实不错", "但是还有一个问题"),
            "你确定现在就要开始吗？": ("你确定现在就要开始吗",),
            "好的。": ("好的",),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_phrases(text), expected)

    def test_pcm_pause_alignment_has_safe_ordered_intervals(self):
        samples = (_silence(.12) + _tone(.36) + _silence(.15) + _tone(1.29)
                   + _silence(.13) + _tone(.81) + _silence(.14))
        duration = len(samples) / _RATE
        result = align_speech("好的，我们先确认这个方案，然后再继续。",
                              array("h", samples).tobytes(), duration)
        self.assertEqual(result.method, "pcm_pause_alignment")
        self.assertEqual(len(result.segments), 3)
        previous_end = 0.0
        for segment in result.segments:
            self.assertGreaterEqual(segment.start, previous_end)
            self.assertLess(segment.start, segment.end)
            self.assertLessEqual(segment.end, duration)
            previous_end = segment.end
        self.assertAlmostEqual(result.segments[0].start, .11, places=2)
        self.assertAlmostEqual(result.segments[0].end, .48, places=2)
        self.assertAlmostEqual(result.segments[1].start, .63, places=2)
        self.assertAlmostEqual(result.segments[1].end, 1.92, places=2)

    def test_proportional_last_fallback_is_explicitly_marked(self):
        result = align_speech("好的，我们先确认这个方案，然后再继续。",
                              array("h", _tone(3)).tobytes(), 3.)
        self.assertEqual(result.method, "proportional_fallback")
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(len(result.segments), 3)


if __name__ == "__main__":
    unittest.main()
