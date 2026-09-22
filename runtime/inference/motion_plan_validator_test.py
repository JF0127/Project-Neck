"""Tests for MotionPlan V2 action and safety validation."""
from __future__ import annotations

import unittest

from .motion_plan import MotionPlan, MotionSegment
from .motion_plan_validator import MotionPlanValidationError, validate_motion_plan


def _plan(action: str, axis: str, amplitude: float = 2.0) -> MotionPlan:
    return MotionPlan(
        mode="speaking",
        duration_sec=2.0,
        segments=(
            MotionSegment(0.2, 1.0, action, axis, amplitude, "test"),
        ),
    )


class MotionPlanValidatorTest(unittest.TestCase):
    def test_action_axis_compatibility_passes(self) -> None:
        for action, axis in (
            ("nod", "pitch"),
            ("turn", "yaw"),
            ("tilt", "roll"),
            ("shake", "yaw"),
        ):
            with self.subTest(action=action):
                plan = _plan(action, axis)
                self.assertIs(validate_motion_plan(plan), plan)

    def test_empty_plan_passes(self) -> None:
        plan = MotionPlan("speaking", 1.5, ())
        self.assertIs(validate_motion_plan(plan), plan)

    def test_incompatible_axis_is_rejected(self) -> None:
        for action, axis in (("nod", "yaw"), ("tilt", "pitch")):
            with self.subTest(action=action), self.assertRaises(
                MotionPlanValidationError
            ):
                validate_motion_plan(_plan(action, axis))

    def test_amplitude_over_limit_is_rejected(self) -> None:
        with self.assertRaises(MotionPlanValidationError):
            validate_motion_plan(_plan("nod", "pitch", 5.01))

    def test_segment_outside_duration_is_rejected_by_structure(self) -> None:
        with self.assertRaises(ValueError):
            MotionPlan(
                "speaking",
                1.0,
                (MotionSegment(0.5, 1.1, "nod", "pitch", 2.0, "test"),),
            )

    def test_duration_mismatch_is_rejected(self) -> None:
        with self.assertRaises(MotionPlanValidationError):
            validate_motion_plan(
                _plan("nod", "pitch"),
                expected_duration_sec=2.1,
            )

    def test_overlap_is_rejected(self) -> None:
        plan = MotionPlan(
            "speaking",
            2.0,
            (
                MotionSegment(0.1, 1.0, "nod", "pitch", 2.0, "first"),
                MotionSegment(0.9, 1.5, "turn", "yaw", 1.5, "second"),
            ),
        )
        with self.assertRaises(MotionPlanValidationError):
            validate_motion_plan(plan)


if __name__ == "__main__":
    unittest.main()
