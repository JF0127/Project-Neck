"""Pure-software tests for TTS PCM phrase alignment and Motion payloads."""
from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..contracts import (
    MotionRequest,
    RobotSpeech,
    RobotState,
    SessionContext,
    TurnContext,
)
from .deepseek_motion import DeepSeekMotionBackend
from .motion_plan import MotionPlan
from .trajectory_generator import TrajectoryGenerator
from .speech_alignment import (
    SpeechAlignmentError,
    align_speech,
    split_phrases,
)

_SAMPLE_RATE = 16_000


def _silence(duration_sec: float) -> list[int]:
    return [0] * round(duration_sec * _SAMPLE_RATE)


def _tone(duration_sec: float, amplitude: int = 9_000) -> list[int]:
    return [
        round(
            amplitude
            * math.sin(2.0 * math.pi * 220.0 * index / _SAMPLE_RATE)
        )
        for index in range(round(duration_sec * _SAMPLE_RATE))
    ]


def _example_speech(text: str) -> RobotSpeech:
    samples = (
        _silence(0.12)
        + _tone(0.36)
        + _silence(0.15)
        + _tone(1.29)
        + _silence(0.13)
        + _tone(0.81)
        + _silence(0.14)
    )
    pcm = array("h", samples).tobytes()
    return RobotSpeech(text, pcm, (), len(samples) / _SAMPLE_RATE)


class _CaptureResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        payload = json.loads(kwargs["input"][1]["content"])
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "mode": "speaking",
                    "duration_sec": payload["duration_sec"],
                    "segments": [
                        {
                            "start_sec": 0.3,
                            "end_sec": 1.2,
                            "action": "nod",
                            "primary_axis": "pitch",
                            "amplitude_deg": 2.0,
                            "reason": "test",
                        }
                    ],
                }
            )
        )


class _CaptureClient:
    def __init__(self) -> None:
        self.responses = _CaptureResponses()


class SpeechAlignmentTest(unittest.TestCase):
    def test_chinese_phrase_splitting_is_conservative(self) -> None:
        cases = {
            "好的，我们先确认这个方案，然后再继续。": (
                "好的",
                "我们先确认这个方案",
                "然后再继续",
            ),
            "这个方法确实不错，但是还有一个问题。": (
                "这个方法确实不错",
                "但是还有一个问题",
            ),
            "你确定现在就要开始吗？": ("你确定现在就要开始吗",),
            "好的。": ("好的",),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_phrases(text), expected)

    def test_pcm_pause_alignment_has_safe_ordered_intervals(self) -> None:
        speech = _example_speech("好的，我们先确认这个方案，然后再继续。")
        result = align_speech(
            speech.text,
            speech.pcm_s16le,
            speech.duration_sec,
        )
        self.assertEqual(result.method, "pcm_pause_alignment")
        self.assertEqual(len(result.segments), 3)
        previous_end = 0.0
        for segment in result.segments:
            self.assertGreaterEqual(segment.start, previous_end)
            self.assertLess(segment.start, segment.end)
            self.assertLessEqual(segment.end, speech.duration_sec)
            previous_end = segment.end
        self.assertAlmostEqual(result.segments[0].start, 0.11, places=2)
        self.assertAlmostEqual(result.segments[0].end, 0.48, places=2)
        self.assertAlmostEqual(result.segments[1].start, 0.63, places=2)
        self.assertAlmostEqual(result.segments[1].end, 1.92, places=2)

    def test_proportional_last_fallback_is_explicitly_marked(self) -> None:
        text = "好的，我们先确认这个方案，然后再继续。"
        samples = _tone(3.0)
        pcm = array("h", samples).tobytes()
        result = align_speech(text, pcm, 3.0)
        self.assertEqual(result.method, "proportional_fallback")
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(len(result.segments), 3)

    def test_motion_payload_contains_prosody_and_motion_plan_v2(self) -> None:
        text = "好的，我们先确认这个方案，然后再继续。"
        speech = _example_speech(text)
        client = _CaptureClient()
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=client,
            )
            request = MotionRequest.snapshot(
                TurnContext(
                    "aligned",
                    robot_text=text,
                    robot_speech=speech,
                ),
                SessionContext("test"),
                RobotState(),
            )
            output = backend.infer(request)
            payload = json.loads(client.responses.calls[0]["input"][1]["content"])
            self.assertEqual(payload["robot_text"], text)
            self.assertEqual(payload["duration_sec"], speech.duration_sec)
            self.assertEqual(len(payload["segments"]), 3)
            first_segment = payload["segments"][0]
            self.assertEqual(
                set(first_segment), {"text", "start_sec", "end_sec", "prosody"}
            )
            self.assertIn("relative_energy", first_segment["prosody"])
            self.assertIn("energy_peak_time_sec", first_segment["prosody"])
            artifact = json.loads(Path(directory, "motion_plan.json").read_text())
            self.assertEqual(artifact["mode"], "speaking")
            self.assertEqual(artifact["segments"][0]["action"], "nod")
            prosody = json.loads(Path(directory, "prosody.json").read_text())
            self.assertEqual(prosody["alignment"]["method"], "pcm_pause_alignment")
            direct = TrajectoryGenerator().generate(
                MotionPlan.from_dict(
                    {
                        "mode": "speaking",
                        "duration_sec": speech.duration_sec,
                        "segments": [
                            {
                                "start_sec": 0.3,
                                "end_sec": 1.2,
                                "action": "nod",
                                "primary_axis": "pitch",
                                "amplitude_deg": 2.0,
                                "reason": "test",
                            }
                        ],
                    }
                )
            )
            self.assertEqual(output, direct)

    def test_alignment_failure_uses_original_text_duration_payload(self) -> None:
        def fail_alignment(*_: object, **__: object):
            raise SpeechAlignmentError("injected alignment failure")

        text = "好的。"
        speech = _example_speech(text)
        client = _CaptureClient()
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=client,
                aligner=fail_alignment,
            )
            request = MotionRequest.snapshot(
                TurnContext(
                    "fallback",
                    robot_text=text,
                    robot_speech=speech,
                ),
                SessionContext("test"),
                RobotState(),
            )
            backend.infer(request)
            payload = json.loads(client.responses.calls[0]["input"][1]["content"])
            self.assertEqual(
                set(payload), {"robot_text", "duration_sec"}
            )
            artifact = json.loads(Path(directory, "prosody.json").read_text())
            self.assertIsNone(artifact["segments"])
            self.assertEqual(
                artifact["alignment"]["method"],
                "text_duration_fallback",
            )

    def test_three_turns_have_no_alignment_state_leak(self) -> None:
        client = _CaptureClient()
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=client,
            )
            for index, text in enumerate(
                (
                    "好的，我们先确认这个方案，然后再继续。",
                    "这个方法确实不错，但是还有一个问题。",
                    "你确定现在就要开始吗？",
                ),
                1,
            ):
                speech = _example_speech(text)
                request = MotionRequest.snapshot(
                    TurnContext(
                        f"turn_{index}",
                        robot_text=text,
                        robot_speech=speech,
                    ),
                    SessionContext("test"),
                    RobotState(),
                )
                backend.infer(request)
        self.assertEqual(len(client.responses.calls), 3)
        for call in client.responses.calls:
            payload = json.loads(call["input"][1]["content"])
            self.assertTrue(payload["segments"])


if __name__ == "__main__":
    unittest.main()
