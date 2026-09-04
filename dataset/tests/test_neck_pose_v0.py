"""Synthetic math tests for the experimental candidate RPY convention."""

import unittest

import numpy as np

from src.features.neck_pose_v0 import project_to_so3, rotation_to_zyx_rpy


def rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class CandidateEulerTests(unittest.TestCase):
    def test_identity(self) -> None:
        np.testing.assert_allclose(rotation_to_zyx_rpy(np.eye(3)), [0, 0, 0], atol=1e-12)

    def test_pure_x_rotation(self) -> None:
        angle = np.deg2rad(30.0)
        np.testing.assert_allclose(rotation_to_zyx_rpy(rx(angle)), [angle, 0, 0], atol=1e-12)

    def test_pure_y_rotation(self) -> None:
        angle = np.deg2rad(20.0)
        np.testing.assert_allclose(rotation_to_zyx_rpy(ry(angle)), [0, angle, 0], atol=1e-12)

    def test_pure_z_rotation(self) -> None:
        angle = np.deg2rad(-15.0)
        np.testing.assert_allclose(rotation_to_zyx_rpy(rz(angle)), [0, 0, angle], atol=1e-12)

    def test_svd_projection(self) -> None:
        expected = rz(0.3) @ ry(-0.2) @ rx(0.1)
        projected, singular_values = project_to_so3(1.02 * expected)
        np.testing.assert_allclose(projected, expected, atol=1e-12)
        np.testing.assert_allclose(singular_values, [1.02, 1.02, 1.02], atol=1e-12)
        np.testing.assert_allclose(projected.T @ projected, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(projected)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
