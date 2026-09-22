"""Tests for Continuous Motion Generator V3."""
from __future__ import annotations

import math
import unittest

import numpy as np

from .motion_plan import MotionPlan, MotionSegment
from .processor import MotionProcessor
from .trajectory_generator import TrajectoryGenerator


FPS = 30.0


class TrajectoryGeneratorTest(unittest.TestCase):
    @staticmethod
    def _plan(
        action: str,
        axis: str,
        *,
        duration: float = 3.0,
        start: float = 0.8,
        end: float = 1.6,
        amplitude: float = 2.0,
    ) -> MotionPlan:
        return MotionPlan(
            mode="speaking",
            duration_sec=duration,
            segments=(
                MotionSegment(start, end, action, axis, amplitude, "test"),
            ),
        )

    @staticmethod
    def _frames(start: float, end: float, frame_count: int) -> tuple[int, int]:
        start_frame = min(frame_count - 2, max(0, round(start * FPS)))
        end_frame = min(frame_count - 1, max(start_frame + 1, round(end * FPS)))
        return start_frame, end_frame

    @staticmethod
    def _values(plan: MotionPlan) -> np.ndarray:
        return np.asarray(TrajectoryGenerator().generate(plan).rpy_offset)

    def test_empty_plan_generates_deterministic_low_frequency_base_flow(self) -> None:
        plan = MotionPlan("speaking", 3.0, ())
        first = self._values(plan)
        second = self._values(plan)

        self.assertEqual(first.shape, (90, 3))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], np.zeros(3))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.any(np.abs(first[1:]) > 0.0))
        self.assertLessEqual(float(np.max(np.abs(np.degrees(first)))), 0.5)
        self.assertLess(
            float(np.max(np.abs(np.degrees(np.diff(first, axis=0))))),
            0.1,
        )

    def test_single_gestures_are_local_modulations_over_the_base(self) -> None:
        base = self._values(MotionPlan("speaking", 3.0, ()))
        for action, axis_index, axis in (
            ("nod", 1, "pitch"),
            ("turn", 2, "yaw"),
            ("tilt", 0, "roll"),
        ):
            with self.subTest(action=action):
                plan = self._plan(action, axis)
                values = self._values(plan)
                modulation = values - base
                start, end = self._frames(0.8, 1.6, len(values))

                self.assertEqual(values.shape, (90, 3))
                np.testing.assert_array_equal(values[0], np.zeros(3))
                other_axes = [index for index in range(3) if index != axis_index]
                np.testing.assert_allclose(
                    values[:, other_axes], base[:, other_axes], atol=1e-14
                )
                np.testing.assert_allclose(modulation[:start], 0.0, atol=1e-14)
                np.testing.assert_allclose(modulation[end + 1 :], 0.0, atol=1e-14)
                self.assertAlmostEqual(modulation[start, axis_index], 0.0)
                self.assertAlmostEqual(modulation[end, axis_index], 0.0)
                self.assertGreater(
                    float(np.max(modulation[:, axis_index])), math.radians(1.9)
                )
                self.assertTrue(np.any(np.abs(values[end + 1 :]) > 0.0))

    def test_shake_is_local_yaw_oscillation_and_preserves_other_axes(self) -> None:
        base = self._values(MotionPlan("speaking", 3.0, ()))
        values = self._values(self._plan("shake", "yaw"))
        modulation = values - base
        start, end = self._frames(0.8, 1.6, len(values))

        np.testing.assert_allclose(values[:, :2], base[:, :2], atol=1e-14)
        self.assertGreater(float(np.max(modulation[:, 2])), math.radians(1.9))
        self.assertLess(float(np.min(modulation[:, 2])), -math.radians(1.4))
        self.assertAlmostEqual(modulation[start, 2], 0.0)
        self.assertAlmostEqual(modulation[end, 2], 0.0)
        np.testing.assert_allclose(modulation[:start], 0.0, atol=1e-14)
        np.testing.assert_allclose(modulation[end + 1 :], 0.0, atol=1e-14)

    def test_two_gestures_have_nonzero_continuous_carrier_in_the_gap(self) -> None:
        plan = MotionPlan(
            "speaking",
            4.0,
            (
                MotionSegment(0.5, 1.2, "nod", "pitch", 2.0, "first"),
                MotionSegment(2.4, 3.1, "turn", "yaw", 2.0, "second"),
            ),
        )
        values = self._values(plan)
        base = self._values(MotionPlan("speaking", 4.0, ()))
        _, first_end = self._frames(0.5, 1.2, len(values))
        second_start, _ = self._frames(2.4, 3.1, len(values))
        gap = values[first_end + 1 : second_start]

        self.assertTrue(len(gap) > 0)
        self.assertTrue(np.any(np.linalg.norm(gap, axis=1) > 0.0))
        np.testing.assert_allclose(gap, base[first_end + 1 : second_start])
        self.assertLess(
            float(np.max(np.abs(np.degrees(np.diff(values, axis=0))))),
            1.0,
        )

    def test_same_axis_gestures_do_not_accumulate_drift(self) -> None:
        plan = MotionPlan(
            "speaking",
            4.0,
            (
                MotionSegment(0.5, 1.2, "nod", "pitch", 2.0, "first"),
                MotionSegment(2.2, 2.9, "nod", "pitch", 2.0, "second"),
            ),
        )
        values = self._values(plan)
        base = self._values(MotionPlan("speaking", 4.0, ()))
        modulation = values - base
        _, first_end = self._frames(0.5, 1.2, len(values))
        second_start, second_end = self._frames(2.2, 2.9, len(values))

        np.testing.assert_allclose(
            modulation[first_end + 1 : second_start], 0.0, atol=1e-14
        )
        np.testing.assert_allclose(modulation[second_end + 1 :], 0.0, atol=1e-14)
        self.assertLessEqual(
            float(np.max(np.abs(np.degrees(modulation[:, 1])))),
            2.0 + 1e-10,
        )
        self.assertTrue(
            np.any(np.linalg.norm(values[first_end + 1 : second_start], axis=1) > 0.0)
        )

    def test_cross_axis_gestures_only_modulate_their_primary_axis(self) -> None:
        plan = MotionPlan(
            "speaking",
            5.0,
            (
                MotionSegment(0.5, 1.1, "nod", "pitch", 2.0, "nod"),
                MotionSegment(1.8, 2.5, "tilt", "roll", -1.8, "tilt"),
                MotionSegment(3.2, 4.0, "turn", "yaw", 2.2, "turn"),
            ),
        )
        values = self._values(plan)
        base = self._values(MotionPlan("speaking", 5.0, ()))
        modulation = values - base

        for start_sec, end_sec, axis in ((0.5, 1.1, 1), (1.8, 2.5, 0), (3.2, 4.0, 2)):
            start, end = self._frames(start_sec, end_sec, len(values))
            other_axes = [index for index in range(3) if index != axis]
            np.testing.assert_allclose(
                modulation[start : end + 1, other_axes], 0.0, atol=1e-14
            )
            self.assertTrue(np.any(np.abs(modulation[start : end + 1, axis]) > 0.0))

    def test_composed_offsets_are_bounded_by_five_degrees(self) -> None:
        values = self._values(
            self._plan(
                "turn",
                "yaw",
                duration=2.0,
                start=0.2,
                end=1.5,
                amplitude=5.0,
            )
        )
        values_deg = np.degrees(values)
        self.assertLessEqual(float(np.max(np.abs(values_deg))), 5.0 + 1e-12)
        self.assertTrue(np.any(np.isclose(values_deg[:, 2], 5.0, atol=1e-10)))

    def test_generator_contract_and_position_continuity(self) -> None:
        plan = self._plan("shake", "yaw", duration=2.0, start=0.3, end=1.5)
        output = TrajectoryGenerator().generate(plan)
        values = np.asarray(output.rpy_offset)

        self.assertEqual(output.fps, 30.0)
        self.assertEqual(output.unit, "radian")
        self.assertEqual(output.order, ("roll", "pitch", "yaw"))
        self.assertEqual(values.shape, (60, 3))
        np.testing.assert_array_equal(values[0], np.zeros(3))
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(
            float(np.max(np.abs(np.degrees(np.diff(values, axis=0))))),
            1.0,
        )

    def test_plan_generator_processor_optimizer_pipeline_returns_global_neutral(self) -> None:
        plan = MotionPlan(
            "speaking",
            5.0,
            (
                MotionSegment(0.8, 1.4, "nod", "pitch", 2.0, "nod"),
                MotionSegment(2.2, 3.0, "turn", "yaw", 2.5, "turn"),
                MotionSegment(3.6, 4.3, "tilt", "roll", -1.8, "tilt"),
            ),
        )
        output = TrajectoryGenerator().generate(plan)
        self.assertTrue(np.any(np.abs(np.asarray(output.rpy_offset)[-1]) > 0.0))

        final = MotionProcessor().process(output, (0.0, 0.0, 0.0))
        values = np.asarray(final.rpy)
        self.assertGreater(len(values), len(output.rpy_offset))
        self.assertEqual(len(final.states), len(values))
        self.assertEqual(final.states[: len(output.rpy_offset)], ("speaking",) * len(output.rpy_offset))
        self.assertTrue(all(state == "silent" for state in final.states[len(output.rpy_offset) :]))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[0], np.zeros(3))
        np.testing.assert_array_equal(values[-1], np.zeros(3))
        self.assertLess(
            float(np.max(np.abs(np.degrees(np.diff(values, axis=0))))),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
