"""Pure-software checks for the production Qwen streaming Runtime entry."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .__main__ import DEFAULT_CONFIG, load_config
from .qwen_streaming_runtime import QwenStreamingRuntime, build_motion_components


class _FakeStreamingModel:
    def __init__(self) -> None:
        self.state_kwargs: dict | None = None

    def init_streaming_state(self, **kwargs: object) -> SimpleNamespace:
        self.state_kwargs = dict(kwargs)
        return SimpleNamespace(text="")

    def streaming_transcribe(self, *_: object) -> None:
        return None

    def finish_streaming_transcribe(self, *_: object) -> None:
        return None


class _FakeQwen3ASRModel:
    created: _FakeStreamingModel | None = None

    @classmethod
    def LLM(cls, **_: object) -> _FakeStreamingModel:
        cls.created = _FakeStreamingModel()
        return cls.created


class QwenStreamingRuntimeTest(unittest.TestCase):
    def test_runtime_config_selects_only_qwen_streaming(self) -> None:
        config, _ = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["asr"]["backend"], "qwen3_streaming")
        self.assertEqual(
            set(config["asr"]),
            {
                "backend",
                "model_path",
                "language",
                "chunk_size_sec",
                "unfixed_chunk_num",
                "unfixed_token_num",
                "gpu_memory_utilization",
                "max_inference_batch_size",
                "max_new_tokens",
            },
        )

    def test_motion_disabled_builds_no_motor_objects(self) -> None:
        config = {
            "dialogue": {},
            "motion": {"enabled": False},
            "motor": {"send_enabled": True},
        }
        state, generator, sender, monitor, offset, enabled = build_motion_components(
            config, DEFAULT_CONFIG
        )
        self.assertFalse(state.motor_available)
        self.assertIsNone(generator)
        self.assertIsNone(sender)
        self.assertIsNone(monitor)
        self.assertEqual(offset, 0)
        self.assertFalse(enabled)

    def test_qwen_runtime_initializes_streaming_state_from_config(self) -> None:
        fake_module = ModuleType("qwen_asr")
        fake_module.Qwen3ASRModel = _FakeQwen3ASRModel
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict(sys.modules, {"qwen_asr": fake_module}):
                runtime = QwenStreamingRuntime(
                    model_path=model_path,
                    vad_model_path=Path(
                        "runtime/models/silero_vad/silero_vad.jit"
                    ).resolve(),
                    dialogue=SimpleNamespace(),
                    tts=SimpleNamespace(),
                    motion_send_to_motor=False,
                    asr_language="Chinese",
                    asr_chunk_size_sec=1.25,
                    asr_unfixed_chunk_num=3,
                    asr_unfixed_token_num=6,
                )
                runtime._start_speech()
        self.assertEqual(
            _FakeQwen3ASRModel.created.state_kwargs,
            {
                "language": "Chinese",
                "chunk_size_sec": 1.25,
                "unfixed_chunk_num": 3,
                "unfixed_token_num": 6,
            },
        )

    def test_python_m_runtime_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "runtime", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Qwen3-ASR Streaming", result.stdout)


if __name__ == "__main__":
    unittest.main()
