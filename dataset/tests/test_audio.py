"""Unit tests for Speech V1 WAV and duration validation."""

import tempfile
import unittest
import wave
from pathlib import Path

from src.features.audio import AudioExtractionError, probe_wav, validate_audio_alignment


class AudioValidationTests(unittest.TestCase):
    def test_standard_wav_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(b"\x00\x00" * 16000)
            metadata = probe_wav(path)
            self.assertEqual(metadata["sample_rate"], 16000)
            self.assertEqual(metadata["channels"], 1)
            self.assertEqual(metadata["sample_width_bits"], 16)
            self.assertAlmostEqual(metadata["duration"], 1.0)

    def test_duration_alignment_boundary(self) -> None:
        self.assertAlmostEqual(validate_audio_alignment(10.0, 10.05), 0.05)
        with self.assertRaises(AudioExtractionError):
            validate_audio_alignment(10.0, 10.11)


if __name__ == "__main__":
    unittest.main()
