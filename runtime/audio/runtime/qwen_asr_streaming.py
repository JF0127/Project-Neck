"""Minimal WAV-driven Qwen3-ASR streaming test using the vLLM backend."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np

MODEL_PATH = Path("/home/jhl/projects/Project-Neck/dataset/models/qwen")
AUDIO_PATH = Path(
    "/home/jhl/projects/Project-Neck/dataset/datasets/zhubo_shuo_lianbo/"
    "clean_v2/male/kanghui/n0Zvshs8wzk_a0421f2b63/"
    "n0Zvshs8wzk_a0421f2b63.wav"
)
SAMPLE_RATE = 16_000
INPUT_CHUNK_SEC = 0.1
STREAMING_CHUNK_SEC = 1.0
UNFIXED_CHUNK_NUM = 4
UNFIXED_TOKEN_NUM = 5


def load_waveform(wav_path: Path) -> np.ndarray:
    """Load a WAV as 16 kHz mono float32 samples."""
    path = wav_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    waveform, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def run_streaming_test(model_path: Path, wav_path: Path) -> None:
    from qwen_asr import Qwen3ASRModel

    resolved_model = model_path.expanduser().resolve()
    if not (resolved_model / "config.json").is_file():
        raise ValueError(f"invalid Qwen3-ASR model directory: {resolved_model}")

    waveform = load_waveform(wav_path)
    model = Qwen3ASRModel.LLM(
        model=str(resolved_model),
        gpu_memory_utilization=0.7,
        max_inference_batch_size=1,
        max_new_tokens=4096,
    )
    state_kwargs = {
        "language": "Chinese",
        "chunk_size_sec": STREAMING_CHUNK_SEC,
        "unfixed_chunk_num": UNFIXED_CHUNK_NUM,
        "unfixed_token_num": UNFIXED_TOKEN_NUM,
    }

    # Run one decode on a separate state so the measured stream does not include
    # vLLM's first-decode initialization cost.
    warmup_state = model.init_streaming_state(**state_kwargs)
    warmup_samples = min(len(waveform), int(SAMPLE_RATE * STREAMING_CHUNK_SEC))
    model.streaming_transcribe(waveform[:warmup_samples], warmup_state)
    model.finish_streaming_transcribe(warmup_state)

    state = model.init_streaming_state(**state_kwargs)
    input_chunk_samples = int(SAMPLE_RATE * INPUT_CHUNK_SEC)
    samples_sent = 0
    previous_text = state.text
    streaming_started = time.perf_counter()

    while samples_sent < len(waveform):
        chunk_end = min(samples_sent + input_chunk_samples, len(waveform))
        chunk = waveform[samples_sent:chunk_end]
        call_started = time.perf_counter()
        model.streaming_transcribe(chunk, state)
        call_sec = time.perf_counter() - call_started
        samples_sent = chunk_end

        if state.text != previous_text:
            elapsed_sec = time.perf_counter() - streaming_started
            input_sec = samples_sent / SAMPLE_RATE
            print(f"真实经过时间: {elapsed_sec:.3f} 秒", flush=True)
            print(f"已输入音频时长: {input_sec:.3f} 秒", flush=True)
            print(f"state.text: {state.text}", flush=True)
            print(f"本次调用耗时: {call_sec:.3f} 秒", flush=True)
            previous_text = state.text

        time.sleep(chunk.size / SAMPLE_RATE)

    model.finish_streaming_transcribe(state)
    total_sec = time.perf_counter() - streaming_started
    print(f"最终文本: {state.text}", flush=True)
    print(f"总真实耗时: {total_sec:.3f} 秒", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR vLLM streaming with a WAV")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--wav", type=Path, default=AUDIO_PATH)
    args = parser.parse_args()
    run_streaming_test(args.model, args.wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
