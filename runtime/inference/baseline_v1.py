"""TorchScript Baseline V1 deployment backend.

Loads the exported deployment package (``model.pt`` + ``vocab.json``) directly.
It never imports the Algorithm training package: the only cross-boundary
contract is the frozen tensor interface documented by the deployment package.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..contracts import (
    MotionOutput,
    MotionRequest,
    RobotSpeech,
    WordTimestamp,
)
from .base import MOTION_FPS, MotionBackend

SAMPLE_RATE = 16_000
SPECIAL_TOKENS = ("<pad>", "<unk>", "<no_word>")
NO_WORD_ID = 2


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "PyTorch is required to load the Baseline V1 deployment model"
        ) from exc
    return torch


class BaselineV1Backend(MotionBackend):
    """Build the frozen 9-tensor input contract and run one resident model."""

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.torch = _require_torch()
        self.model_path = Path(model_path).expanduser()
        self.vocab_path = Path(vocab_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Baseline V1 deployment model does not exist: {self.model_path}"
            )
        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"Baseline V1 vocabulary does not exist: {self.vocab_path}"
            )

        self.device = self._resolve_device(device)
        self.vocab = self._load_vocab(self.vocab_path)
        self.unk_id = self.vocab["<unk>"]

        print(f"[runtime][motion] loading Baseline V1 deployment: {self.model_path}")
        self.model = self.torch.jit.load(str(self.model_path), map_location="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.inference_count = 0
        print(
            f"[runtime][motion] Baseline V1 ready: device={self.device}, "
            f"vocab={len(self.vocab)}"
        )

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("motion device must be auto, cpu or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("motion device cuda was requested but CUDA is unavailable")
        return torch.device(requested)

    @staticmethod
    def _load_vocab(path: Path) -> dict[str, int]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Baseline V1 vocabulary: {path}") from exc
        tokens = value.get("tokens") if isinstance(value, dict) else value
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError("Baseline V1 vocabulary must be a list of strings")
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"vocabulary must start with {SPECIAL_TOKENS}")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary tokens must be unique")
        return {token: index for index, token in enumerate(tokens)}

    @staticmethod
    def _validate_speech(speech: object) -> RobotSpeech:
        if not isinstance(speech, RobotSpeech):
            raise ValueError("motion inference requires RobotSpeech")
        pcm = speech.pcm_s16le
        if not pcm or len(pcm) % 2:
            raise ValueError("robot pcm_s16le must contain an even non-zero byte count")
        duration = len(pcm) / 2 / SAMPLE_RATE
        if not math.isfinite(speech.duration_sec) or not math.isclose(
            speech.duration_sec, duration, rel_tol=0.0, abs_tol=1.0 / SAMPLE_RATE
        ):
            raise ValueError(
                f"robot duration {speech.duration_sec} does not match PCM {duration}"
            )
        return speech

    def _word_tensors(
        self, words: Sequence[WordTimestamp]
    ) -> tuple[Any, Any, Any, Any, Any]:
        torch = self.torch
        word_count = len(words)
        max_chars = max((len(word.text) for word in words), default=0)

        token_ids = torch.zeros((1, word_count, max_chars), dtype=torch.int64)
        token_mask = torch.zeros((1, word_count, max_chars), dtype=torch.bool)
        starts = torch.zeros((1, word_count), dtype=torch.float64)
        ends = torch.zeros((1, word_count), dtype=torch.float64)
        word_mask = torch.ones((1, word_count), dtype=torch.bool)

        for index, word in enumerate(words):
            if not math.isfinite(word.start_sec) or not math.isfinite(word.end_sec):
                raise ValueError(f"word {index} has a non-finite interval")
            if word.start_sec > word.end_sec:
                raise ValueError(f"word {index} interval start exceeds end")
            token_ids[0, index, : len(word.text)] = torch.tensor(
                [self.vocab.get(character, self.unk_id) for character in word.text],
                dtype=torch.int64,
            )
            if word.text:
                token_mask[0, index, : len(word.text)] = True
            starts[0, index] = word.start_sec
            ends[0, index] = word.end_sec
        return token_ids, token_mask, starts, ends, word_mask

    def _infer(self, request: MotionRequest) -> MotionOutput:
        speech = self._validate_speech(request.current_turn.robot_speech)
        torch = self.torch

        samples = np.frombuffer(speech.pcm_s16le, dtype="<i2").astype(np.float32)
        samples /= 32768.0
        audio = torch.from_numpy(samples.copy()).unsqueeze(0)
        audio_lengths = torch.tensor([len(samples)], dtype=torch.int64)

        token_ids, token_mask, starts, ends, word_mask = self._word_tensors(
            tuple(speech.words)
        )
        frame_count = max(2, int(round(speech.duration_sec * MOTION_FPS)))
        query_timestamps = (
            torch.arange(frame_count, dtype=torch.float64) / MOTION_FPS
        ).unsqueeze(0)
        sequence_mask = torch.ones((1, frame_count), dtype=torch.bool)

        inputs = (
            audio,
            audio_lengths,
            token_ids,
            token_mask,
            starts,
            ends,
            word_mask,
            query_timestamps,
            sequence_mask,
        )
        moved = tuple(value.to(self.device) for value in inputs)
        with torch.inference_mode():
            prediction = self.model(*moved)

        rpy = prediction[0].detach().to("cpu").numpy().astype(np.float32)
        if rpy.shape != (frame_count, 3):
            raise RuntimeError(
                f"Baseline V1 returned shape {rpy.shape}, expected {(frame_count, 3)}"
            )
        if not np.isfinite(rpy).all():
            raise RuntimeError("Baseline V1 prediction contains NaN/Inf")
        self.inference_count += 1
        return MotionOutput(
            rpy_offset=tuple(
                tuple(float(component) for component in frame) for frame in rpy
            ),
            fps=MOTION_FPS,
        )
