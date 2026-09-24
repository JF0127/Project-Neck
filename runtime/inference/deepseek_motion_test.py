"""Pure software text-only planner and downstream generator integration checks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..contracts import MotionRequest, RobotSpeech, RobotState, SessionContext, TurnContext
from .deepseek_motion import DeepSeekMotionBackend, DeepSeekMotionError
from .generator import TurnGenerator
from .processor import MotionProcessor
from .text_motion_plan import TextMotionPlan


class Client:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.loads(kwargs["input"][1]["content"])["reply_text"]
        return SimpleNamespace(output_text=json.dumps({"reply_text": text, "actions": self.actions}))


def request(text="是的，我同意。", duration=1.5):
    return MotionRequest.snapshot(
        TurnContext("test", robot_text=text,
                    robot_speech=RobotSpeech(text, b"\0\0" * round(duration * 16000), (), duration)),
        SessionContext("test"), RobotState(head_rpy_valid=True, motor_available=True),
    )


class TextMotionTest(unittest.TestCase):
    def test_planner_only_sends_reply_text_and_validates_anchor(self):
        client = Client([{"type": "nod", "anchor": "同意", "position": 4, "intensity": "medium"}])
        with tempfile.TemporaryDirectory() as tmp:
            backend = DeepSeekMotionBackend(client=client, output_dir=tmp)
            raw, plan = backend.plan_text("是的，我同意。")
            self.assertIsInstance(plan, TextMotionPlan)
            self.assertEqual(set(json.loads(client.calls[0]["input"][1]["content"])), {"reply_text"})
            self.assertEqual(plan.actions[0].position, 4)
            self.assertEqual(json.loads(raw), plan.to_dict())
            output = backend.infer(request())
            self.assertEqual(len(output.rpy_offset), 45)
            self.assertEqual(json.loads(Path(tmp, "motion_plan.json").read_text()), plan.to_dict())
            self.assertFalse(Path(tmp, "prosody.json").exists())

    def test_empty_actions_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = DeepSeekMotionBackend(client=Client([]), output_dir=tmp)
            self.assertEqual(backend.plan_text("天气晴朗。")[1].actions, ())
            self.assertEqual(len(backend.infer(request("天气晴朗。")).rpy_offset), 45)

    def test_bad_action_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = DeepSeekMotionBackend(client=Client([
                {"type": "turn", "anchor": "同意", "position": 4, "intensity": "high"}]), output_dir=tmp)
            with self.assertRaises(DeepSeekMotionError):
                backend.infer(request())
            self.assertFalse(Path(tmp, "motion_plan.json").exists())

    def test_existing_turn_generator_still_accepts_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = DeepSeekMotionBackend(client=Client([
                {"type": "nod", "anchor": "同意", "position": 4, "intensity": "medium"}]), output_dir=tmp)
            turn = TurnGenerator(backend, MotionProcessor(), Path(tmp) / "turn").generate(request(), "test_turn")
            self.assertIsNotNone(turn.document)
            self.assertTrue(Path(tmp, "final_trajectory.json").exists())


if __name__ == "__main__":
    unittest.main()
