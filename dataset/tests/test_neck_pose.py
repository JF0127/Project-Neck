"""Synthetic tests for the formal Neck Pose V1 mathematics."""

import unittest

import numpy as np

from src.features.neck_pose import (
    filter_neck_rpy,
    interpolate_short_rotation_gaps,
    relative_rotations,
    rotation_to_zyx_rpy,
    semantic_candidate_to_human,
)


def rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class NeckPoseV1Tests(unittest.TestCase):
    def test_neutral_is_identity_and_zero_rpy(self) -> None:
        neutral = rz(0.4) @ ry(-0.2) @ rx(0.1)
        rotations = np.stack([neutral, neutral @ rx(0.2)])
        relative, index = relative_rotations(rotations, np.array([True, True]))
        self.assertEqual(index, 0)
        np.testing.assert_allclose(relative[0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation_to_zyx_rpy(relative[0]), [0, 0, 0], atol=1e-12)

    def test_relative_rotation_recovers_local_x_motion(self) -> None:
        neutral = rz(-0.3) @ ry(0.25)
        expected = rx(np.deg2rad(10.0))
        rotations = np.stack([neutral, neutral @ expected])
        relative, _ = relative_rotations(rotations, np.array([True, True]))
        np.testing.assert_allclose(relative[1], expected, atol=1e-12)

    def test_semantic_mapping(self) -> None:
        candidate = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(semantic_candidate_to_human(candidate), [3.0, 1.0, 2.0])

    def test_slerp_fills_short_internal_gap(self) -> None:
        rotations = np.full((3, 3, 3), np.nan)
        rotations[0] = np.eye(3)
        rotations[2] = rz(np.deg2rad(20.0))
        timestamps = np.array([0.0, 0.1, 0.2])
        output, valid, count, max_gap = interpolate_short_rotation_gaps(
            rotations, np.array([True, False, True]), timestamps, max_gap_seconds=0.20
        )
        self.assertEqual(count, 1)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(max_gap, 0.1)
        np.testing.assert_allclose(output[1], rz(np.deg2rad(10.0)), atol=1e-10)

    def test_long_gap_stays_invalid(self) -> None:
        rotations = np.full((4, 3, 3), np.nan)
        rotations[0] = np.eye(3)
        rotations[3] = rz(0.3)
        timestamps = np.array([0.0, 0.1, 0.2, 0.35])
        output, valid, count, max_gap = interpolate_short_rotation_gaps(
            rotations, np.array([True, False, False, True]), timestamps, max_gap_seconds=0.20
        )
        self.assertEqual(count, 0)
        self.assertFalse(valid[1:3].any())
        self.assertTrue(np.isnan(output[1:3]).all())
        self.assertAlmostEqual(max_gap, 0.25)

    def test_filter_preserves_low_and_attenuates_high_frequency(self) -> None:
        fs = 50.0
        time = np.arange(0.0, 30.0, 1.0 / fs)
        valid = np.ones(len(time), dtype=bool)
        low = np.sin(2.0 * np.pi * 0.3 * time)
        high = np.sin(2.0 * np.pi * 5.0 * time)
        low_track = np.repeat(low[:, None], 3, axis=1)
        high_track = np.repeat(high[:, None], 3, axis=1)
        low_output, _ = filter_neck_rpy(low_track, valid, fs)
        high_output, _ = filter_neck_rpy(high_track, valid, fs)
        center = slice(250, -250)
        low_ratio = np.sqrt(np.mean(low_output[center, 0] ** 2)) / np.sqrt(np.mean(low[center] ** 2))
        high_ratio = np.sqrt(np.mean(high_output[center, 0] ** 2)) / np.sqrt(np.mean(high[center] ** 2))
        self.assertGreater(low_ratio, 0.98)
        self.assertLess(high_ratio, 0.01)


if __name__ == "__main__":
    unittest.main()
