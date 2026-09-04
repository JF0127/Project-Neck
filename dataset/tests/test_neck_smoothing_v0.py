"""Synthetic tests for the Neck Motion Smoothing V0 filters."""

import unittest

import numpy as np

from src.features.neck_smoothing_v0 import CUTOFFS_HZ, filter_rpy_comparison


class NeckSmoothingTests(unittest.TestCase):
    fs = 50.0

    def _three_channels(self, signal: np.ndarray) -> np.ndarray:
        return np.repeat(signal[:, None], 3, axis=1).astype(np.float32)

    def test_constant_signal_is_preserved(self) -> None:
        raw = self._three_channels(np.full(1000, np.deg2rad(10.0)))
        valid = np.ones(len(raw), dtype=bool)
        outputs, short = filter_rpy_comparison(raw, valid, self.fs)
        for cutoff in CUTOFFS_HZ:
            np.testing.assert_allclose(outputs[cutoff], raw, atol=1e-6)
            self.assertEqual(short[cutoff], 0)

    def test_low_frequency_signal_is_preserved(self) -> None:
        time = np.arange(0.0, 20.0, 1.0 / self.fs)
        signal = np.sin(2.0 * np.pi * 0.3 * time)
        raw = self._three_channels(signal)
        valid = np.ones(len(raw), dtype=bool)
        outputs, _ = filter_rpy_comparison(raw, valid, self.fs)
        center = slice(100, -100)
        input_rms = np.sqrt(np.mean(raw[center, 0] ** 2))
        for cutoff in CUTOFFS_HZ:
            output_rms = np.sqrt(np.mean(outputs[cutoff][center, 0] ** 2))
            self.assertGreater(output_rms / input_rms, 0.98)

    def test_high_frequency_attenuation_orders_cutoffs(self) -> None:
        time = np.arange(0.0, 40.0, 1.0 / self.fs)
        raw = self._three_channels(np.sin(2.0 * np.pi * 5.0 * time))
        valid = np.ones(len(raw), dtype=bool)
        outputs, _ = filter_rpy_comparison(raw, valid, self.fs)
        # Exclude filtfilt's finite-record edge transient when measuring attenuation.
        center = slice(300, -300)
        amplitudes = {
            cutoff: float(np.sqrt(np.mean(outputs[cutoff][center, 0] ** 2)))
            for cutoff in CUTOFFS_HZ
        }
        self.assertLess(amplitudes[1.0], amplitudes[1.5])
        self.assertLess(amplitudes[1.5], amplitudes[2.0])
        self.assertLess(amplitudes[2.0], 0.1)

    def test_nan_gap_is_preserved_and_segments_are_independent(self) -> None:
        time = np.arange(400) / self.fs
        raw = self._three_channels(np.sin(2.0 * np.pi * 0.5 * time))
        valid = np.ones(len(raw), dtype=bool)
        valid[180:220] = False
        raw[~valid] = np.nan
        outputs, _ = filter_rpy_comparison(raw, valid, self.fs)
        for cutoff in CUTOFFS_HZ:
            output = outputs[cutoff]
            self.assertEqual(output.shape, raw.shape)
            self.assertTrue(np.isnan(output[180:220]).all())
            self.assertTrue(np.isfinite(output[:180]).all())
            self.assertTrue(np.isfinite(output[220:]).all())

        changed_second = raw.copy()
        changed_second[220:] += 100.0
        changed_outputs, _ = filter_rpy_comparison(changed_second, valid, self.fs)
        for cutoff in CUTOFFS_HZ:
            np.testing.assert_array_equal(outputs[cutoff][:180], changed_outputs[cutoff][:180])


if __name__ == "__main__":
    unittest.main()
