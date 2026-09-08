"""Speaker-only Baseline V1 inference and Neck trajectory construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .baseline_motion import BaselineSpeakerMotion, FPS

FIRST_FRAME_TOLERANCE = 1e-6
GenerationCallback = Callable[[str, np.ndarray, dict], None]


@dataclass(frozen=True)
class MotionTurn:
    document: dict
    speaking_start_frame: int
    speaker_frames: int


class SpeakerMotionPipeline:
    """Run resident Baseline V1 inference and append a simple neutral tail."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        silent_tail_sec: float = 0.8,
    ) -> None:
        if silent_tail_sec <= 0.0:
            raise ValueError("silent_tail_sec must be greater than zero")
        self.baseline = BaselineSpeakerMotion(checkpoint, device=device)
        self.silent_tail_sec = silent_tail_sec

    @staticmethod
    def _validate_speaker(speaker_rpy: np.ndarray) -> np.ndarray:
        speaker = np.asarray(speaker_rpy, dtype=np.float32)
        if speaker.ndim != 2 or speaker.shape[1] != 3 or len(speaker) < 2:
            raise ValueError("speaker prediction must have shape [N,3] with N >= 2")
        if not np.isfinite(speaker).all():
            raise ValueError("speaker prediction contains NaN/Inf")
        if not np.allclose(
            speaker[0], np.zeros(3, dtype=np.float32),
            rtol=0.0, atol=FIRST_FRAME_TOLERANCE,
        ):
            raise ValueError(
                f"Baseline V1 first frame must be zero within "
                f"{FIRST_FRAME_TOLERANCE}: {speaker[0].tolist()}"
            )
        return speaker

    def _build_speaker_trajectory(
        self, speaker_rpy: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Preserve the raw speaker segment, then linearly return to neutral."""
        speaker = self._validate_speaker(speaker_rpy)
        tail_frames = max(1, int(round(self.silent_tail_sec * FPS)))
        # Exclude the duplicated start pose; the final sample is exactly neutral.
        tail = np.linspace(
            speaker[-1], np.zeros(3, dtype=np.float32),
            tail_frames + 1, dtype=np.float64,
        )[1:].astype(np.float32)
        trajectory = np.concatenate([speaker, tail], axis=0)
        states = ["speaking"] * len(speaker) + ["silent"] * len(tail)
        if not np.isfinite(trajectory).all() or len(trajectory) != len(states):
            raise RuntimeError("speaker trajectory validation failed")
        if not np.array_equal(trajectory[-1], np.zeros(3, dtype=np.float32)):
            raise RuntimeError("speaker trajectory does not end at neutral")
        return trajectory, states

    def generate_turn(
        self,
        robot_pcm_s16le: bytes,
        robot_words: list[dict],
        robot_duration_sec: float,
        turn_number: int,
        on_generation: GenerationCallback | None = None,
    ) -> MotionTurn:
        prediction = self.baseline.infer(
            robot_pcm_s16le, robot_words, robot_duration_sec
        )
        if on_generation is not None:
            on_generation(
                "speaker",
                prediction.rpy,
                {
                    "model": "baseline_v1",
                    "checkpoint": str(self.baseline.checkpoint_path),
                    "fps": FPS,
                    "query_timestamps_sec": prediction.query_timestamps_sec.tolist(),
                    "representation": "rpy_offset",
                },
            )

        trajectory, states = self._build_speaker_trajectory(prediction.rpy)
        document = {
            "name": f"algorithm_turn_{turn_number}",
            "fps": FPS,
            "unit": "radian",
            "order": ["roll", "pitch", "yaw"],
            "trajectory": trajectory.tolist(),
            "states": states,
        }
        return MotionTurn(
            document=document,
            speaking_start_frame=0,
            speaker_frames=len(prediction.rpy),
        )
