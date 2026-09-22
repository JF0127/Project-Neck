"""Pure-software integration tests for the MotionPlan V2 backend."""
from __future__ import annotations

import json
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
from .deepseek_motion import DeepSeekMotionBackend, DeepSeekMotionError
from .generator import TurnGenerator
from .processor import MotionProcessor


class _Responses:
    def __init__(self, document: dict) -> None:
        self.document = document
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(output_text=json.dumps(self.document))


class _Client:
    def __init__(self, document: dict) -> None:
        self.responses = _Responses(document)


def _request(duration: float = 1.5, pcm_value: int = 0) -> MotionRequest:
    sample = int(pcm_value).to_bytes(2, "little", signed=True)
    speech = RobotSpeech(
        "普通说明。",
        sample * round(duration * 16_000),
        (),
        duration,
    )
    return MotionRequest.snapshot(
        TurnContext("v2", robot_text=speech.text, robot_speech=speech),
        SessionContext("test"),
        RobotState(head_rpy=[0.0, 0.0, 0.0], head_rpy_valid=True),
    )


class DeepSeekMotionV2Test(unittest.TestCase):
    def test_empty_plan_is_valid_base_flow_and_parameters_are_preserved(self) -> None:
        document = {"mode": "speaking", "duration_sec": 1.5, "segments": []}
        client = _Client(document)
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(output_dir=directory, client=client)
            output = backend.infer(_request())
            self.assertEqual(len(output.rpy_offset), 45)
            self.assertEqual(output.rpy_offset[0], (0.0, 0.0, 0.0))
            self.assertTrue(
                any(any(abs(component) > 0.0 for component in frame) for frame in output.rpy_offset[1:])
            )
            call = client.responses.calls[0]
            self.assertEqual(call["temperature"], 0.2)
            self.assertEqual(call["max_output_tokens"], 1024)
            self.assertEqual(call["reasoning"], {"effort": "none"})
            for filename in (
                "prosody.json",
                "deepseek_motion_plan_raw.json",
                "motion_plan.json",
                "raw_relative_trajectory.json",
            ):
                self.assertTrue(Path(directory, filename).is_file(), filename)

    def test_prosody_failure_falls_back_to_text_and_duration(self) -> None:
        def fail_prosody(*_: object, **__: object) -> object:
            raise ValueError("injected prosody failure")

        document = {"mode": "speaking", "duration_sec": 1.5, "segments": []}
        client = _Client(document)
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=client,
                prosody_extractor=fail_prosody,
            )
            backend.infer(_request(pcm_value=1000))
            payload = json.loads(client.responses.calls[0]["input"][1]["content"])
            self.assertEqual(set(payload), {"robot_text", "duration_sec"})
            artifact = json.loads(Path(directory, "prosody.json").read_text())
            self.assertIsNone(artifact["segments"])
            self.assertIn("injected prosody failure", artifact["error"])

    def test_invalid_action_axis_fails_the_whole_plan(self) -> None:
        document = {
            "mode": "speaking",
            "duration_sec": 1.5,
            "segments": [
                {
                    "start_sec": 0.2,
                    "end_sec": 0.9,
                    "action": "nod",
                    "primary_axis": "yaw",
                    "amplitude_deg": 2.0,
                    "reason": "invalid axis",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=_Client(document),
            )
            with self.assertRaises(DeepSeekMotionError):
                backend.infer(_request())
            raw = json.loads(
                Path(directory, "deepseek_motion_plan_raw.json").read_text()
            )
            self.assertIn("MotionPlanValidationError", raw["error"])
            self.assertFalse(Path(directory, "motion_plan.json").exists())

    def test_turn_generator_writes_final_trajectory_artifact(self) -> None:
        document = {
            "mode": "speaking",
            "duration_sec": 1.5,
            "segments": [
                {
                    "start_sec": 0.2,
                    "end_sec": 1.0,
                    "action": "turn",
                    "primary_axis": "yaw",
                    "amplitude_deg": 2.0,
                    "reason": "test",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = DeepSeekMotionBackend(
                output_dir=directory,
                client=_Client(document),
            )
            turn = TurnGenerator(
                backend,
                MotionProcessor(),
                Path(directory) / "turn",
            ).generate(_request(), "test_turn")
            self.assertIsNotNone(turn.document)
            artifact = json.loads(Path(directory, "final_trajectory.json").read_text())
            self.assertEqual(artifact["representation"], "absolute_rpy")
            self.assertEqual(len(artifact["trajectory"]), len(turn.document["trajectory"]))


if __name__ == "__main__":
    unittest.main()
