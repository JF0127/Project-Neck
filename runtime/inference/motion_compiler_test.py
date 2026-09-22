"""Pure-software regression checks for DeepSeek gesture compilation."""
from __future__ import annotations

import math
import unittest

from ..contracts import MotionOutput
from .motion_compiler import (
    _minimum_jerk,
    compile_motion_plan,
    parse_motion_plan_with_rejections,
)
from .motor_json import final_trajectory_to_motor_document
from .processor import MotionProcessor
from .trajectory_optimizer import (
    DEFAULT_MAX_ACCELERATION_DEG_S2,
    DEFAULT_MAX_VELOCITY_DEG_S,
    trajectory_metrics,
)


class GestureCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        plan = {
            "actions": [
                {
                    "start": 0.2,
                    "end": 1.2,
                    "roll": 0.0,
                    "pitch": 3.0,
                    "yaw": 0.0,
                },
                {
                    "start": 1.5,
                    "end": 2.5,
                    "roll": 0.0,
                    "pitch": 2.5,
                    "yaw": 0.0,
                },
            ]
        }
        self.actions, rejected = parse_motion_plan_with_rejections(plan, 3.0)
        self.assertFalse(rejected)
        self.output = compile_motion_plan(self.actions, 3.0)

    def test_gestures_do_not_accumulate_and_return_to_baseline(self) -> None:
        frames = self.output.rpy_offset
        pitch_deg = [math.degrees(frame[1]) for frame in frames]
        self.assertAlmostEqual(max(pitch_deg[:40]), 3.0)
        self.assertAlmostEqual(max(pitch_deg[40:]), 2.5)
        for action in self.actions:
            start = round(action.start * 30)
            end = min(len(frames) - 1, round(action.end * 30))
            self.assertEqual(frames[start], (0.0, 0.0, 0.0))
            self.assertEqual(frames[end], (0.0, 0.0, 0.0))
        self.assertEqual(frames[-1], (0.0, 0.0, 0.0))

    def test_neutral_hold_between_gestures(self) -> None:
        frames = self.output.rpy_offset
        for index in range(round(1.2 * 30), round(1.5 * 30) + 1):
            self.assertEqual(frames[index], (0.0, 0.0, 0.0))

    def test_minimum_jerk_endpoint_velocity_and_acceleration(self) -> None:
        step = 1e-5
        for endpoint, direction in ((0.0, 1.0), (1.0, -1.0)):
            value0 = _minimum_jerk(endpoint)
            value1 = _minimum_jerk(endpoint + direction * step)
            value2 = _minimum_jerk(endpoint + direction * 2 * step)
            velocity = (value1 - value0) / step
            acceleration = (value2 - 2 * value1 + value0) / (step * step)
            self.assertLess(abs(velocity), 1e-7)
            self.assertLess(abs(acceleration), 0.01)

    def test_invalid_actions_do_not_remove_valid_actions(self) -> None:
        plan = {
            "actions": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "roll": 0.0,
                    "pitch": 2.0,
                    "yaw": 0.0,
                },
                {
                    "start": 1.2,
                    "end": 2.0,
                    "roll": 8.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            ]
        }
        accepted, rejected = parse_motion_plan_with_rejections(plan, 2.5)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].index, 1)

    def test_motor_json_contract_and_optimized_tail(self) -> None:
        final = MotionProcessor().process(self.output, [0.0, 0.0, 0.0])
        document = final_trajectory_to_motor_document(final, "gesture_test")
        self.assertEqual(
            set(document),
            {"name", "fps", "unit", "order", "trajectory", "states"},
        )
        self.assertEqual(document["fps"], 30.0)
        self.assertEqual(document["unit"], "radian")
        self.assertEqual(document["order"], ["roll", "pitch", "yaw"])
        self.assertEqual(len(document["trajectory"]), len(document["states"]))

        nonneutral = MotionOutput(
            ((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)),
            30.0,
        )
        tailed = MotionProcessor().process(nonneutral, [0.0, 0.0, 0.0])
        self.assertEqual(len(tailed.rpy), 26)
        self.assertEqual(
            tailed.states,
            ("speaking", "speaking") + ("silent",) * 24,
        )
        self.assertEqual(tailed.rpy[0], (0.0, 0.0, 0.0))
        self.assertEqual(tailed.rpy[-1], (0.0, 0.0, 0.0))
        pitch_metrics = trajectory_metrics(tailed.rpy, 30.0)["pitch"]
        self.assertLessEqual(
            pitch_metrics["peak_velocity_deg_s"],
            DEFAULT_MAX_VELOCITY_DEG_S + 1e-5,
        )
        self.assertLessEqual(
            pitch_metrics["peak_acceleration_deg_s2"],
            DEFAULT_MAX_ACCELERATION_DEG_S2 + 1e-4,
        )


if __name__ == "__main__":
    unittest.main()
