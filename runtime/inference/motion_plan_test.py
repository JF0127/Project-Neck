"""Tests for the standalone high-level Motion plan data structures."""
from __future__ import annotations

import math
import unittest

from .motion_plan import MotionPlan, MotionSegment


def _segment(**overrides: object) -> MotionSegment:
    values: dict[str, object] = {
        "start_sec": 0.4,
        "end_sec": 0.9,
        "action": "nod",
        "primary_axis": "pitch",
        "amplitude_deg": 3.0,
        "reason": "emphasis",
    }
    values.update(overrides)
    return MotionSegment(**values)  # type: ignore[arg-type]


class MotionPlanTest(unittest.TestCase):
    def test_valid_plan_construction(self) -> None:
        segment = _segment()
        plan = MotionPlan(
            mode="speaking",
            duration_sec=3.0,
            segments=(segment,),
        )

        self.assertEqual(plan.mode, "speaking")
        self.assertEqual(plan.duration_sec, 3.0)
        self.assertEqual(plan.segments, (segment,))

    def test_empty_segments_are_valid(self) -> None:
        plan = MotionPlan(mode="speaking", duration_sec=1.5, segments=())

        self.assertEqual(plan.segments, ())
        self.assertEqual(MotionPlan.from_dict(plan.to_dict()), plan)

    def test_dict_round_trip(self) -> None:
        original = MotionPlan(
            mode="speaking",
            duration_sec=3.0,
            segments=(_segment(),),
        )
        document = original.to_dict()

        self.assertEqual(
            document,
            {
                "mode": "speaking",
                "duration_sec": 3.0,
                "segments": [
                    {
                        "start_sec": 0.4,
                        "end_sec": 0.9,
                        "action": "nod",
                        "primary_axis": "pitch",
                        "amplitude_deg": 3.0,
                        "reason": "emphasis",
                    }
                ],
            },
        )
        self.assertEqual(MotionPlan.from_dict(document), original)

    def test_duration_must_be_positive(self) -> None:
        for duration in (0.0, -1.0):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                MotionPlan("speaking", duration, ())

    def test_segment_interval_validation(self) -> None:
        invalid_values = (
            {"start_sec": -0.1},
            {"start_sec": 0.4, "end_sec": 0.4},
            {"start_sec": 0.4, "end_sec": 0.3},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _segment(**values)

    def test_segment_end_must_not_exceed_plan_duration(self) -> None:
        with self.assertRaises(ValueError):
            MotionPlan("speaking", 0.8, (_segment(end_sec=0.9),))

    def test_primary_axis_validation(self) -> None:
        with self.assertRaises(ValueError):
            _segment(primary_axis="x")

    def test_numeric_fields_must_be_finite(self) -> None:
        invalid_segments = (
            {"start_sec": math.nan},
            {"end_sec": math.inf},
            {"amplitude_deg": -math.inf},
        )
        for values in invalid_segments:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _segment(**values)
        for duration in (math.nan, math.inf, -math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                MotionPlan("speaking", duration, ())

    def test_action_and_reason_must_not_be_empty(self) -> None:
        for field in ("action", "reason"):
            for value in ("", "   "):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    _segment(**{field: value})

    def test_mode_validation(self) -> None:
        for mode in ("listening", "silent", ""):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                MotionPlan(mode, 3.0, ())

    def test_design_rules_are_not_applied_by_data_objects(self) -> None:
        first = _segment(
            start_sec=0.0,
            end_sec=0.1,
            amplitude_deg=30.0,
        )
        overlapping = _segment(start_sec=0.05, end_sec=0.2)

        plan = MotionPlan("speaking", 1.0, (first, overlapping))

        self.assertEqual(plan.segments, (first, overlapping))


if __name__ == "__main__":
    unittest.main()
