from __future__ import annotations

import math
import unittest

from runtime.motion_core import MotionBackend, MotionProcessor
from runtime.runtime import Runtime
from runtime.contracts import (
    MotionOutput,
    MotionRequest,
    RobotSpeech,
    RobotState,
    SessionContext,
    TurnContext,
    TurnSummary,
    UserAudio,
    UserSpeech,
    WordTimestamp,
)


class UnusedVAD:
    in_speech = False

    def reset(self) -> None:
        pass

    def push(self, frame: bytes):
        return None

    def end_stream(self):
        return None


def make_runtime() -> Runtime:
    return Runtime(
        vad=UnusedVAD(),
        asr=object(),
        dialogue=object(),
        tts=object(),
        dialogue_fallback_text="fallback",
        cooldown_ms=0,
    )


class FakeMotionBackend(MotionBackend):
    def _infer(self, request: MotionRequest) -> MotionOutput:
        return MotionOutput(
            rpy_offset=[
                [0.03, 0.00, -0.03],
                [0.06, 0.02, -0.01],
            ],
            fps=30,
        )


class InvalidMotionBackend(MotionBackend):
    def __init__(self, output: MotionOutput) -> None:
        self.output = output

    def _infer(self, request: MotionRequest) -> MotionOutput:
        return self.output


def make_contexts() -> tuple[TurnContext, SessionContext, RobotState]:
    turn = TurnContext(
        turn_id="turn_current",
        user_audio=UserAudio(b"\x00\x00" * 160, 16_000, 1, 0.01),
        user_speech=UserSpeech(
            text="hello",
            words=[WordTimestamp("hello", 0.0, 0.4)],
            language="en",
        ),
        robot_text="hello back",
        status="processing",
    )
    session = SessionContext("session_test")
    session.add_turn(TurnSummary("turn_previous", "one", "two", "complete"))
    robot = RobotState(
        head_rpy=[0.1, -0.1, 0.05],
        head_rpy_timestamp=12.5,
        head_rpy_valid=True,
        motor_available=True,
        motion_executing=False,
    )
    return turn, session, robot


class SpeechTypeTests(unittest.TestCase):
    def test_robot_speech_uses_explicit_pcm_s16le(self) -> None:
        pcm = b"\x01\x00\xff\xff"
        speech = RobotSpeech(
            text="hello",
            pcm_s16le=pcm,
            words=[WordTimestamp("hello", 0.0, 0.4)],
            duration_sec=0.4,
        )

        self.assertEqual(speech.pcm_s16le, pcm)


class SessionContextTests(unittest.TestCase):
    def test_keeps_only_ten_recent_turns(self) -> None:
        session = SessionContext("session_test")
        for number in range(12):
            session.add_turn(
                TurnSummary(
                    turn_id=f"turn_{number}",
                    user_text=f"user {number}",
                    robot_text=f"robot {number}",
                    status="complete",
                )
            )

        self.assertEqual(len(session.recent_turns), 10)
        self.assertEqual(session.recent_turns[0].turn_id, "turn_2")
        self.assertEqual(session.recent_turns[-1].turn_id, "turn_11")


class MotionRequestTests(unittest.TestCase):
    def test_snapshot_does_not_follow_mutable_runtime_contexts(self) -> None:
        turn, session, robot = make_contexts()
        request = MotionRequest.snapshot(turn, session, robot)

        turn.robot_text = "changed"
        turn.user_speech.words[0] = WordTimestamp("changed", 1.0, 2.0)
        session.add_turn(TurnSummary("turn_new", "new", "new", "complete"))
        robot.head_rpy[0] = 9.0
        robot.motor_available = False

        self.assertEqual(request.current_turn.robot_text, "hello back")
        self.assertEqual(request.current_turn.user_speech.words[0].text, "hello")
        self.assertEqual(len(request.recent_turns), 1)
        self.assertEqual(request.robot_state.head_rpy, [0.1, -0.1, 0.05])
        self.assertTrue(request.robot_state.motor_available)


class MotionBackendTests(unittest.TestCase):
    def test_fake_backend_returns_normalized_contract(self) -> None:
        turn, session, robot = make_contexts()
        output = FakeMotionBackend().infer(MotionRequest.snapshot(turn, session, robot))

        self.assertEqual(output.fps, 30.0)
        self.assertEqual(len(output.rpy_offset), 2)
        self.assertTrue(all(len(frame) == 3 for frame in output.rpy_offset))
        self.assertTrue(
            all(math.isfinite(component) for frame in output.rpy_offset for component in frame)
        )

    def test_base_rejects_invalid_contracts(self) -> None:
        turn, session, robot = make_contexts()
        request = MotionRequest.snapshot(turn, session, robot)
        invalid_outputs = (
            MotionOutput(rpy_offset=[[0.0, float("nan"), 0.0]], fps=30),
            MotionOutput(rpy_offset=[[0.0, 0.0]], fps=30),
            MotionOutput(rpy_offset=[[0.0, 0.0, 0.0]], fps=25),
        )
        for output in invalid_outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                InvalidMotionBackend(output).infer(request)


class MotionProcessorTests(unittest.TestCase):
    def test_offset_conversion_and_neutral_return(self) -> None:
        turn, session, robot = make_contexts()
        output = FakeMotionBackend().infer(MotionRequest.snapshot(turn, session, robot))
        start = [0.1, -0.1, 0.05]
        processor = MotionProcessor(start_transition_frames=2, neutral_return_frames=3)

        absolute = processor.offset_to_absolute(output, start)
        self.assertEqual(absolute[0], (0.13, -0.1, 0.020000000000000004))
        self.assertEqual(absolute[1], (0.16, -0.08, 0.04))

        final = processor.process(output, start)
        self.assertEqual(final.fps, 30.0)
        self.assertEqual(final.rpy[0], tuple(start))
        self.assertEqual(final.rpy[-1], (0.0, 0.0, 0.0))
        self.assertEqual(len(final.rpy), len(final.states))
        self.assertEqual(final.states[-1], "silent")
        self.assertTrue(
            all(math.isfinite(component) for frame in final.rpy for component in frame)
        )


class RuntimeLifecycleTests(unittest.TestCase):
    def test_one_session_and_one_turn(self) -> None:
        runtime = make_runtime()
        session = runtime.start_session("session_one")
        turn = runtime.start_turn("turn_one")
        turn.user_speech = UserSpeech("hello", [], "en")
        turn.robot_text = "hello back"

        request = runtime.create_motion_request()
        self.assertEqual(request.current_turn.turn_id, "turn_one")

        summary = runtime.complete_turn()
        self.assertEqual(summary.user_text, "hello")
        self.assertEqual(session.recent_turns, [summary])
        self.assertIsNone(runtime.active_turn)
        self.assertIs(runtime.end_session(), session)
        self.assertIsNone(runtime.session)

    def test_rejects_a_second_active_turn(self) -> None:
        runtime = make_runtime()
        runtime.start_session("session_one")
        runtime.start_turn("turn_one")
        with self.assertRaises(RuntimeError):
            runtime.start_turn("turn_two")


if __name__ == "__main__":
    unittest.main()
