"""Tests for Continuous Motion Generator V4."""
from __future__ import annotations

import math
import unittest

import numpy as np

from .motion_plan import MotionPlan, MotionSegment
from .processor import MotionProcessor
from .trajectory_generator import TrajectoryGenerator, _compose_v4
from .prosody import ProsodyAnalysis, SegmentProsody


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
        self.assertLessEqual(float(np.max(np.abs(np.degrees(first)))), 2.0)
        self.assertGreater(float(np.max(np.abs(np.degrees(first)))), 0.5)
        self.assertLess(
            float(np.max(np.abs(np.degrees(np.diff(first, axis=0))))),
            0.1,
        )

    @staticmethod
    def _prosody() -> ProsodyAnalysis:
        def segment(start: float, end: float, energy: float, level: str, peak: float) -> SegmentProsody:
            return SegmentProsody("测试", start, end, end - start, 0., 0., .1, .2,
                                  energy, level, 5., "normal", peak)
        return ProsodyAnalysis(6., 16000, (
            segment(0., 2., .8, "low", 1.),
            segment(2., 4., 1.7, "high", 2.9),
            segment(4., 6., .9, "medium", 5.),
        ), 0.)

    def test_postural_only_is_slow_asynchronous_with_holds(self) -> None:
        plan = MotionPlan("speaking", 28., ())
        layers = TrajectoryGenerator().generate_layers(plan, self._prosody(),
                                                         include_prosodic=False, include_semantic=False)
        values = np.degrees(np.asarray(layers.postural_flow))
        np.testing.assert_array_equal(values, np.degrees(layers.composed_raw))
        velocity = np.abs(np.diff(values, axis=0) * FPS)
        self.assertLess(float(np.max(velocity)), 4.)
        self.assertLess(float(np.mean(velocity)), 0.65)
        self.assertGreater(float(np.min(np.mean(velocity < 0.03, axis=0))), 0.15)
        self.assertGreater(float(np.mean(np.abs(values[30:]) > 0.1)), 0.6)
        # Axis transitions do not all start/end at the same timestamps.
        active = velocity > .1
        self.assertGreater(float(np.mean(np.sum(active, axis=1) == 1)), 0.1)
        self.assertFalse(np.array_equal(active[:, 0], active[:, 2]))

    def test_prosodic_accent_local_to_high_energy_peak(self) -> None:
        plan = MotionPlan("speaking", 6., ())
        layers = TrajectoryGenerator().generate_layers(plan, self._prosody())
        accent = np.degrees(np.asarray(layers.prosodic_accent))
        np.testing.assert_array_equal(accent[:60], 0.)
        np.testing.assert_array_equal(accent[120:], 0.)
        self.assertGreater(float(np.max(np.abs(accent[60:120, 1]))), .3)
        self.assertLessEqual(float(np.max(np.abs(accent))), .75 + 1e-10)
        np.testing.assert_array_equal(layers.semantic_gesture, np.zeros((180, 3)))
        np.testing.assert_allclose(np.asarray(layers.composed_raw),
                                   np.asarray(layers.postural_flow) + np.asarray(layers.prosodic_accent))
        self.assertGreater(np.linalg.norm(layers.composed_raw[-1]), 0.)

    def test_three_layers_are_separate_and_semantic_is_sparse(self) -> None:
        plan = MotionPlan("speaking", 6., (
            MotionSegment(2.4, 3.2, "nod", "pitch", 2.0, "emphasis"),
            MotionSegment(4.5, 5.3, "turn", "yaw", 2.0, "shift"),
        ))
        layers = TrajectoryGenerator().generate_layers(plan, self._prosody())
        post, accent, gesture, composed = (np.asarray(getattr(layers, key)) for key in
            ("postural_flow", "prosodic_accent", "semantic_gesture", "composed_raw"))
        np.testing.assert_allclose(composed, post + accent + gesture)
        self.assertGreater(np.max(np.abs(gesture[72:96, 1])), math.radians(1.9))
        self.assertGreater(np.max(np.abs(gesture[135:159, 2])), math.radians(1.9))
        np.testing.assert_array_equal(gesture[:72], 0.)
        np.testing.assert_array_equal(gesture[96:135], 0.)
        np.testing.assert_array_equal(gesture[159:], 0.)
        empty = TrajectoryGenerator().generate_layers(MotionPlan("speaking", 6., ()), self._prosody())
        np.testing.assert_array_equal(post, empty.postural_flow)
        np.testing.assert_array_equal(accent, empty.prosodic_accent)

    def test_safety_priority_postural_then_accent_then_semantic(self) -> None:
        deg = lambda values: [[math.radians(x) for x in values]]
        result = _compose_v4(deg([2, 0, 0]), deg([.6, 0, 0]), deg([4, 0, 0]), 5)
        self.assertAlmostEqual(math.degrees(result[0][0]), 5)
        # Carrier alone absorbs the overflow (1.6 deg), retaining the accent and gesture.
        self.assertAlmostEqual(2 + .6 + 4 - 1.6, math.degrees(result[0][0]))
        result = _compose_v4(deg([.2, 0, 0]), deg([1, 0, 0]), deg([5, 0, 0]), 5)
        self.assertAlmostEqual(math.degrees(result[0][0]), 5)
        result = _compose_v4(deg([-2, 0, 0]), deg([.2, 0, 0]), deg([5, 0, 0]), 5)
        self.assertAlmostEqual(math.degrees(result[0][0]), 3.2)
        # Even an out-of-contract excursion is limited only after the other layers.
        result = _compose_v4(deg([.5, 0, 0]), deg([1, 0, 0]), deg([6, 0, 0]), 5)
        self.assertAlmostEqual(math.degrees(result[0][0]), 5)

    def test_first_frame_with_zero_start_gesture_and_peak_is_zero(self) -> None:
        p = ProsodyAnalysis(1., 16000, (
            SegmentProsody("重", 0., 1., 1., 0., 0., .1, .1, 1.6, "high", 5., "normal", 0.),
        ), 0.)
        layers = TrajectoryGenerator().generate_layers(self._plan(
            "nod", "pitch", duration=1., start=0., end=.7), p)
        for field in ("postural_flow", "prosodic_accent", "semantic_gesture", "composed_raw"):
            np.testing.assert_array_equal(getattr(layers, field)[0], [0., 0., 0.])

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
