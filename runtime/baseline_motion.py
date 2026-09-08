"""Deployment adapter for Algorithm Baseline V1 speaker-motion inference."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from algorithm.features import CharacterVocabulary, FeatureEncoder
from algorithm.models.baseline import BaselineModel

SAMPLE_RATE = 16_000
FPS = 30.0


@dataclass(frozen=True)
class SpeakerPrediction:
    rpy: np.ndarray
    query_timestamps_sec: np.ndarray


def build_query_timestamps(duration_sec: float, fps: float = FPS) -> torch.Tensor:
    """Return a [1,N] physical-seconds grid beginning at zero."""
    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be finite and greater than zero")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and greater than zero")
    num_frames = max(2, int(round(duration_sec * fps)))
    return torch.arange(num_frames, dtype=torch.float64).unsqueeze(0) / fps


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("motion device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("motion device cuda was requested but CUDA is unavailable")
    return torch.device(requested)


class BaselineSpeakerMotion:
    """Load one Baseline V1 checkpoint and serve repeated speaker inference."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Baseline V1 checkpoint does not exist: {self.checkpoint_path}")
        self.device = _resolve_device(device)

        print(f"[runtime][motion] loading Baseline V1 once: {self.checkpoint_path}")
        checkpoint_data = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        required = {
            "feature_encoder_state_dict", "model_state_dict", "config", "vocab"
        }
        missing = required - checkpoint_data.keys()
        if missing:
            raise ValueError(
                f"Baseline V1 checkpoint missing fields: {sorted(missing)}"
            )

        self.config = checkpoint_data["config"]
        self._validate_contract(self.config)
        vocab_value = checkpoint_data["vocab"]
        if not isinstance(vocab_value, dict) or not isinstance(vocab_value.get("tokens"), list):
            raise ValueError("Baseline V1 checkpoint vocab.tokens must be a list")
        self.vocab = CharacterVocabulary(vocab_value["tokens"])

        feature_config = self.config["features"]
        model_config = self.config["model"]
        self.feature_encoder = FeatureEncoder(
            self.vocab,
            audio_feature_dim=int(feature_config["audio_dim"]),
            text_feature_dim=int(feature_config["text_dim"]),
        )
        self.model = BaselineModel(
            audio_dim=int(feature_config["audio_dim"]),
            text_dim=int(feature_config["text_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            num_layers=int(model_config["layers"]),
            num_heads=int(model_config["heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            dropout=float(model_config["dropout"]),
        )
        self.feature_encoder.load_state_dict(checkpoint_data["feature_encoder_state_dict"])
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.feature_encoder.to(self.device).eval()
        self.model.to(self.device).eval()
        self.inference_count = 0
        print(
            f"[runtime][motion] Baseline V1 ready: device={self.device}, "
            f"vocab={len(self.vocab)}"
        )

    @staticmethod
    def _validate_contract(config: dict) -> None:
        target = config.get("target", {})
        if target.get("representation") != "rpy_offset":
            raise ValueError("Baseline V1 target representation must be rpy_offset")
        if target.get("order") != ["roll", "pitch", "yaw"]:
            raise ValueError("Baseline V1 target order must be roll,pitch,yaw")
        if target.get("unit") not in {"rad", "radian"}:
            raise ValueError("Baseline V1 target unit must be radian")

        frontend = config.get("features", {}).get("audio_frontend", {})
        expected = {
            "sample_rate": SAMPLE_RATE,
            "n_fft": 400,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 80,
            "center": False,
            "boundary": "edge_hold",
        }
        mismatched = {
            key: (frontend.get(key), value)
            for key, value in expected.items()
            if frontend.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"unsupported Baseline V1 audio frontend contract: {mismatched}"
            )

    @staticmethod
    def _pcm_to_waveform(pcm_s16le: bytes) -> np.ndarray:
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ValueError("pcm_s16le must contain a non-empty even number of bytes")
        return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def _adapt_words(words: list[dict]) -> list[dict]:
        adapted: list[dict] = []
        for index, word in enumerate(words):
            try:
                text = str(word["text"])
                start = float(word["start_time"])
                end = float(word["end_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid robot word at index {index}") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start > end:
                raise ValueError(f"invalid robot word interval at index {index}")
            adapted.append({"text": text, "local_start": start, "local_end": end})
        return adapted

    def prepare_inputs(
        self,
        pcm_s16le: bytes,
        words: list[dict],
        duration_sec: float,
    ) -> dict:
        """Create the exact Algorithm FeatureEncoder inputs for one TTS stream."""
        waveform = self._pcm_to_waveform(pcm_s16le)
        pcm_duration = len(waveform) / SAMPLE_RATE
        if not math.isclose(duration_sec, pcm_duration, rel_tol=0.0, abs_tol=1.0 / SAMPLE_RATE):
            raise ValueError(
                f"TTS duration {duration_sec} does not match PCM duration {pcm_duration}"
            )
        query_timestamps = build_query_timestamps(duration_sec)
        sequence_mask = torch.ones(query_timestamps.shape, dtype=torch.bool)
        return {
            "audio": torch.from_numpy(waveform.copy()).unsqueeze(0),
            "audio_lengths": torch.tensor([len(waveform)], dtype=torch.long),
            "words": [self._adapt_words(words)],
            "neck_timestamps": query_timestamps,
            "sequence_mask": sequence_mask,
        }

    def infer(
        self,
        pcm_s16le: bytes,
        words: list[dict],
        duration_sec: float,
    ) -> SpeakerPrediction:
        batch = self.prepare_inputs(pcm_s16le, words, duration_sec)
        device_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            conditions = self.feature_encoder(
                device_batch["audio"],
                device_batch["audio_lengths"],
                device_batch["words"],
                device_batch["neck_timestamps"],
                device_batch["sequence_mask"],
            )
            prediction = self.model(
                conditions["aligned_audio"],
                conditions["aligned_text"],
                device_batch["neck_timestamps"],
                device_batch["sequence_mask"],
            )

        rpy = prediction[0].detach().cpu().numpy().astype(np.float32)
        timestamps = batch["neck_timestamps"][0].numpy().copy()
        if rpy.shape != (len(timestamps), 3):
            raise RuntimeError(
                f"Baseline V1 returned shape {rpy.shape}, expected {(len(timestamps), 3)}"
            )
        if not np.isfinite(rpy).all():
            raise RuntimeError("Baseline V1 prediction contains NaN/Inf")
        self.inference_count += 1
        print(
            f"[runtime][motion] Baseline V1 inference #{self.inference_count}: "
            f"frames={len(rpy)}, duration={duration_sec:.3f}s"
        )
        return SpeakerPrediction(rpy=rpy, query_timestamps_sec=timestamps)
