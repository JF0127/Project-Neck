from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from runtime.baseline_motion import BaselineSpeakerMotion, build_query_timestamps
from runtime.motion import SpeakerMotionPipeline
from runtime.neck_client import NeckClient


REPO_ROOT = Path(__file__).resolve().parents[2]


def real_checkpoint_path() -> Path:
    configured = os.getenv("BASELINE_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = sorted(
        (REPO_ROOT / "algorithm/outputs/baseline_audio_text").glob(
            "run_*/checkpoints/best.pt"
        )
    )
    if not candidates:
        raise unittest.SkipTest(
            "set BASELINE_CHECKPOINT to an Algorithm Baseline V1 best.pt"
        )
    return candidates[-1].resolve()


def synthetic_pcm(duration_sec: float = 1.0) -> bytes:
    sample_count = int(round(duration_sec * 16_000))
    time_axis = np.arange(sample_count, dtype=np.float64) / 16_000.0
    waveform = 0.05 * np.sin(2.0 * np.pi * 220.0 * time_axis)
    return np.rint(waveform * 32767.0).astype("<i2").tobytes()


class BaselineSpeakerMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = real_checkpoint_path()
        cls.adapter = BaselineSpeakerMotion(cls.checkpoint, device="cpu")
        cls.pcm = synthetic_pcm()
        cls.words = [
            {"text": "测试", "start_time": 0.10, "end_time": 0.50},
            {"text": "语音", "start_time": 0.50, "end_time": 0.90},
        ]
        captured_timestamps: list[torch.Tensor] = []

        def capture_feature_inputs(module, args):
            captured_timestamps.append(args[3].detach().cpu().clone())

        hook = cls.adapter.feature_encoder.register_forward_pre_hook(capture_feature_inputs)
        try:
            cls.prediction = cls.adapter.infer(cls.pcm, cls.words, 1.0)
        finally:
            hook.remove()
        cls.feature_timestamps = captured_timestamps[0]

        cls.pipeline = SpeakerMotionPipeline.__new__(SpeakerMotionPipeline)
        cls.pipeline.baseline = cls.adapter
        cls.pipeline.silent_tail_sec = 0.8
        cls.generations: list[tuple[str, np.ndarray, dict]] = []
        cls.turn = cls.pipeline.generate_turn(
            cls.pcm,
            cls.words,
            1.0,
            1,
            lambda role, trajectory, metadata: cls.generations.append(
                (role, trajectory.copy(), metadata)
            ),
        )

    def test_checkpoint_loads_complete_baseline(self):
        self.assertEqual(self.adapter.config["target"]["representation"], "rpy_offset")
        self.assertGreater(len(self.adapter.vocab), 3)
        self.assertFalse(self.adapter.feature_encoder.training)
        self.assertFalse(self.adapter.model.training)

    def test_query_timestamps_are_physical_30_hz_seconds(self):
        timestamps = build_query_timestamps(0.1)
        self.assertEqual(timestamps.shape, (1, 3))
        torch.testing.assert_close(
            timestamps[0],
            torch.tensor([0.0, 1.0 / 30.0, 2.0 / 30.0], dtype=torch.float64),
        )

    def test_physical_seconds_reach_feature_encoder(self):
        torch.testing.assert_close(
            self.feature_timestamps,
            build_query_timestamps(1.0),
        )
        prepared = self.adapter.prepare_inputs(self.pcm, self.words, 1.0)
        self.assertEqual(
            prepared["words"][0][0],
            {"text": "测试", "local_start": 0.10, "local_end": 0.50},
        )

    def test_real_checkpoint_speaker_inference_is_finite_and_zero_based(self):
        self.assertEqual(self.prediction.rpy.shape, (30, 3))
        self.assertTrue(np.isfinite(self.prediction.rpy).all())
        np.testing.assert_allclose(self.prediction.rpy[0], 0.0, rtol=0, atol=1e-6)

    def test_speaker_only_tail_and_motor_document(self):
        document = self.turn.document
        self.assertEqual(self.turn.speaking_start_frame, 0)
        self.assertEqual(self.turn.speaker_frames, 30)
        self.assertEqual(document["states"][:30], ["speaking"] * 30)
        self.assertEqual(document["states"][30:], ["silent"] * 24)
        self.assertNotIn("listening", document["states"])
        np.testing.assert_array_equal(
            np.asarray(document["trajectory"][:30], dtype=np.float32),
            self.generations[0][1],
        )
        np.testing.assert_array_equal(document["trajectory"][-1], [0.0, 0.0, 0.0])
        NeckClient.validate(document)
        json.dumps(document, allow_nan=False)

    def test_generation_is_baseline_speaker_with_query_grid(self):
        self.assertEqual(len(self.generations), 1)
        role, trajectory, metadata = self.generations[0]
        self.assertEqual(role, "speaker")
        self.assertEqual(metadata["model"], "baseline_v1")
        self.assertEqual(metadata["representation"], "rpy_offset")
        self.assertEqual(Path(metadata["checkpoint"]), self.checkpoint)
        self.assertEqual(len(metadata["query_timestamps_sec"]), len(trajectory))

    def test_motion_api_has_no_user_or_listener_inputs(self):
        parameters = inspect.signature(SpeakerMotionPipeline.generate_turn).parameters
        self.assertEqual(
            list(parameters),
            [
                "self", "robot_pcm_s16le", "robot_words", "robot_duration_sec",
                "turn_number", "on_generation",
            ],
        )

    def test_active_runtime_path_has_no_v3_imports(self):
        for relative in (
            "runtime/baseline_motion.py",
            "runtime/motion.py",
            "runtime/runtime.py",
            "runtime/__main__.py",
        ):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("motion_model", source, relative)
            self.assertNotIn("MultiCandidate", source, relative)
            self.assertNotIn("num_candidates", source, relative)


if __name__ == "__main__":
    unittest.main()
