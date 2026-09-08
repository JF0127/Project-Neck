from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from runtime.vad import SileroVAD


class SileroVADTests(unittest.TestCase):
    def test_buffers_20ms_frames_into_silero_windows_and_segments_speech(self) -> None:
        scores = iter([0.9, 0.9, 0.1])
        windows = []

        def score(waveform):
            windows.append(waveform.copy())
            return next(scores)

        vad = SileroVAD(
            "unused.jit",
            threshold=0.5,
            min_speech_ms=64,
            min_silence_ms=32,
            score_fn=score,
        )
        frame = b"\x01\x00" * 320
        segment = None
        for index in range(5):
            segment = vad.push(frame)
            if index == 3:
                self.assertTrue(vad.in_speech)

        self.assertIsNotNone(segment)
        self.assertFalse(vad.in_speech)
        self.assertEqual(len(windows), 3)
        self.assertEqual(len(segment.pcm_s16le), 3 * 512 * 2)
        self.assertEqual(segment.sample_rate, 16_000)
        self.assertEqual(segment.channels, 1)
        self.assertEqual(segment.duration_sec, 3 * 512 / 16_000)

    def test_loads_local_jit_once_and_sets_eval(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.eval_count = 0
                self.reset_count = 0

            def eval(self):
                self.eval_count += 1
                return self

            def reset_states(self):
                self.reset_count += 1

        model = FakeModel()
        load_calls = []
        fake_torch = SimpleNamespace(
            jit=SimpleNamespace(
                load=lambda path, map_location: load_calls.append((path, map_location)) or model
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "silero_vad.jit"
            model_path.write_bytes(b"local model fixture")
            with patch.dict(sys.modules, {"torch": fake_torch}):
                vad = SileroVAD(model_path)

        self.assertIs(vad.model, model)
        self.assertEqual(load_calls, [(str(model_path.resolve()), "cpu")])
        self.assertEqual(model.eval_count, 1)
        self.assertEqual(model.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
