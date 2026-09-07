"""Resident v3 motion model and in-memory turn trajectory composition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .motion_model import build_model
from .motion_model.rotations import matrix_to_rpy, rpy_to_matrix
from .motion_model.text_features import START_OF_DIALOGUE, Vocab, encode_word_timestamps

SAMPLE_RATE = 16_000
FPS = 30.0
RAD2DEG = 180.0 / np.pi
ROLE_ID = {"speaker": 0, "listener": 1}
GenerationCallback = Callable[[str, np.ndarray, dict], None]


@dataclass(frozen=True)
class MotionTurn:
    document: dict
    speaking_start_frame: int
    listener_frames: int
    speaker_frames: int


def _candidate_energy(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(np.abs(np.diff(trajectory, axis=0)).mean()) * RAD2DEG * FPS


class ResidentMotionModel:
    """Load checkpoint/vocabulary/model once and serve repeated in-memory inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        num_candidates: int = 8,
        blend_sec: float = 0.3,
        thinking_sec: float = 0.3,
        silent_tail_sec: float = 0.8,
        seed: int = 42,
    ):
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        self.num_candidates = num_candidates
        self.blend_sec = blend_sec
        self.thinking_sec = thinking_sec
        self.silent_tail_sec = silent_tail_sec
        print(f"[runtime][motion] loading checkpoint once: {self.checkpoint_path}")
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint_data["config"]
        if self.config.get("model", {}).get("type") != "candidates":
            raise ValueError("Algorithm Runtime V1 requires model.type=candidates")
        vocab_path = Path(checkpoint_data["vocab_path"])
        if not vocab_path.exists():
            vocab_path = self.checkpoint_path.parent.parent / vocab_path.name
        self.vocab = Vocab.load(vocab_path)
        self.model = build_model(self.config, vocab_size=len(self.vocab)).to(self.device)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.eval()
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.inference_count = 0
        print(
            f"[runtime][motion] model ready: {type(self.model).__name__}, "
            f"device={self.device}, K={self.num_candidates}"
        )

    @staticmethod
    def _pcm_to_waveform(pcm_s16le: bytes) -> np.ndarray:
        if len(pcm_s16le) % 2:
            raise ValueError("pcm_s16le byte length must be even")
        return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0

    def _build_batch(
        self,
        pcm_s16le: bytes,
        words: list[dict],
        role: str,
        previous_text: str | None,
    ) -> dict:
        waveform = self._pcm_to_waveform(pcm_s16le)
        n_frames = max(2, int(round(len(waveform) / SAMPLE_RATE * FPS)))
        word_ids, starts, ends, _ = encode_word_timestamps(
            self.vocab, words, n_frames, FPS
        )
        previous_ids = (
            self.vocab.encode_text(previous_text)
            if previous_text and previous_text.strip()
            else [self.vocab.word2idx[START_OF_DIALOGUE]]
        )
        if not previous_ids:
            previous_ids = [self.vocab.word2idx[START_OF_DIALOGUE]]
        return {
            "audio": torch.from_numpy(waveform.copy()).view(1, 1, -1),
            "Ns": torch.tensor([n_frames], dtype=torch.long),
            "mask": torch.ones(1, n_frames, dtype=torch.bool),
            "roles": torch.tensor([ROLE_ID[role]], dtype=torch.long),
            "prev_token_ids": torch.tensor([previous_ids], dtype=torch.long),
            "prev_lens": torch.tensor([len(previous_ids)], dtype=torch.long),
            "word_ids": [word_ids],
            "word_starts": [starts],
            "word_ends": [ends],
        }

    def _infer_segment(
        self,
        pcm_s16le: bytes,
        words: list[dict],
        role: str,
        previous_text: str | None,
        expected_energy: float,
        on_generation: GenerationCallback | None = None,
    ) -> np.ndarray:
        batch = self._build_batch(pcm_s16le, words, role, previous_text)
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            candidates = self.model.sample(batch, n_samples=self.num_candidates)
        pool = candidates[0].detach().cpu().numpy().astype(np.float32)
        energies = np.asarray([_candidate_energy(candidate) for candidate in pool])
        selected = int(np.argmin(np.abs(energies - expected_energy)))
        self.inference_count += 1
        print(
            f"[runtime][motion] inference #{self.inference_count}: role={role}, "
            f"frames={pool.shape[1]}, candidate={selected}, "
            f"energy={energies[selected]:.2f} deg/s"
        )
        selected_trajectory = pool[selected]
        if on_generation is not None:
            on_generation(
                role,
                selected_trajectory,
                {
                    "fps": FPS,
                    "candidate_index": selected,
                    "energy_deg_per_s": float(energies[selected]),
                },
            )
        return selected_trajectory

    @staticmethod
    def _silent_segment(start_pose: np.ndarray, n_frames: int) -> np.ndarray:
        return np.linspace(start_pose, np.zeros(3), n_frames, dtype=np.float64).astype(np.float32)

    def _compose(self, listener: np.ndarray, speaker: np.ndarray) -> tuple[np.ndarray, list[str]]:
        segments: list[tuple[str, np.ndarray]] = []
        r_abs = np.eye(3, dtype=np.float64)
        r_origin = r_abs.copy()

        for state, relative in (
            ("listening", listener),
            ("silent", None),
            ("speaking", speaker),
            ("silent", None),
        ):
            if relative is not None:
                r_segment = r_abs @ rpy_to_matrix(relative)
                unified = matrix_to_rpy(r_origin.T @ r_segment).astype(np.float32)
                r_abs = r_segment[-1]
            else:
                duration = self.thinking_sec if len(segments) == 1 else self.silent_tail_sec
                count = max(1, int(round(duration * FPS)))
                start_pose = matrix_to_rpy(r_origin.T @ r_abs)
                unified = self._silent_segment(start_pose, count)
                r_abs = r_origin.copy()
            segments.append((state, unified))

        trajectories: list[np.ndarray] = []
        states: list[list[str]] = []
        blend_frames = int(round(self.blend_sec * FPS))
        for state, segment in segments:
            current = segment.copy()
            if trajectories and blend_frames > 1 and len(current) > 1:
                count = min(blend_frames, len(trajectories[-1]) // 2, len(current) // 2)
                if count > 1:
                    weight = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
                    current[:count] = trajectories[-1][-count:] * (1.0 - weight) + current[:count] * weight
                    trajectories[-1] = trajectories[-1][:-count]
                    states[-1] = states[-1][:-count]
            trajectories.append(current)
            states.append([state] * len(current))

        trajectory = np.concatenate(trajectories).astype(np.float32)
        flat_states = [state for part in states for state in part]
        if len(trajectory) != len(flat_states):
            raise RuntimeError("motion trajectory/state length mismatch")
        if not np.isfinite(trajectory).all():
            raise RuntimeError("motion trajectory contains NaN/Inf")
        trajectory[0] = 0.0
        trajectory[-1] = 0.0
        return trajectory, flat_states

    def generate_turn(
        self,
        user_pcm_s16le: bytes,
        user_words: list[dict],
        user_text: str,
        robot_pcm_s16le: bytes,
        robot_words: list[dict],
        robot_text: str,
        turn_number: int,
        on_generation: GenerationCallback | None = None,
    ) -> MotionTurn:
        listener = self._infer_segment(
            user_pcm_s16le, user_words, "listener", None, expected_energy=3.5,
            on_generation=on_generation,
        )
        speaker = self._infer_segment(
            robot_pcm_s16le, robot_words, "speaker", user_text, expected_energy=6.0,
            on_generation=on_generation,
        )
        trajectory, states = self._compose(listener, speaker)
        speaking_start = states.index("speaking")
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
            speaking_start_frame=speaking_start,
            listener_frames=len(listener),
            speaker_frames=len(speaker),
        )
