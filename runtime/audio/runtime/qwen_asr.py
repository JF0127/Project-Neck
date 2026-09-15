"""Minimal resident Qwen3-ASR runtime for local WAV transcription."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = Path(
    "/home/jhl/projects/Project-Neck/dataset/models/qwen"
)


class ASRRuntime:
    """Load Qwen3-ASR once and reuse the GPU-resident model."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR dependencies are missing; install runtime/audio/requirements.txt"
            ) from exc

        self._torch = torch
        self.model_path = model_path.expanduser().resolve()
        if not (self.model_path / "config.json").is_file():
            raise ValueError(f"invalid Qwen3-ASR model directory: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-ASR requires an available CUDA GPU")

        started = time.perf_counter()
        self._model = Qwen3ASRModel.from_pretrained(
            str(self.model_path),
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        self.model_load_sec = time.perf_counter() - started
        print(f"模型加载耗时: {self.model_load_sec:.3f} 秒", flush=True)

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe one WAV with the already-loaded model."""
        audio_path = wav_path.expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"WAV file not found: {audio_path}")
        if audio_path.suffix.lower() != ".wav":
            raise ValueError(f"input must be a WAV file: {audio_path}")

        self._torch.cuda.synchronize()
        started = time.perf_counter()
        result: Any = self._model.transcribe(
            audio=str(audio_path),
            language="Chinese",
        )
        self._torch.cuda.synchronize()
        inference_sec = time.perf_counter() - started

        if not isinstance(result, list) or len(result) != 1:
            raise RuntimeError(f"unexpected Qwen3-ASR result for: {audio_path}")
        text = str(result[0].text).strip()
        print(f"ASR 推理耗时: {inference_sec:.3f} 秒", flush=True)
        print(f"识别文本: {text}", flush=True)
        return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe WAV files with one resident local Qwen3-ASR model"
    )
    parser.add_argument(
        "wav",
        type=Path,
        nargs="+",
        help="one or more WAV paths; all files reuse the same model instance",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    runtime = ASRRuntime(args.model)
    for wav_path in args.wav:
        runtime.transcribe(wav_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
