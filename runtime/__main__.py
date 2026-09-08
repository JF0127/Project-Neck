"""CLI entry for the Stage 3 voice Runtime: ``python -m runtime``."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = RUNTIME_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project-Neck voice Runtime")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Runtime YAML config path",
    )
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _required(section: dict[str, Any], name: str, section_name: str) -> Any:
    if name not in section:
        raise ValueError(f"config field {section_name}.{name} is required")
    return section[name]


def _model_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read runtime/config.yaml") from exc

    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Runtime config root must be a mapping")
    return value, config_path


def build_runtime(config: dict[str, Any], config_path: Path):
    from .asr import WhisperASR
    from .dialogue import DeepSeekDialogue
    from .runtime import Runtime
    from .tts import EdgeTTS
    from .vad import SileroVAD

    audio = _section(config, "audio")
    vad_config = _section(config, "vad")
    asr_config = _section(config, "asr")
    dialogue_config = _section(config, "dialogue")
    tts_config = _section(config, "tts")
    runtime_config = _section(config, "runtime")

    if vad_config.get("backend") != "silero":
        raise ValueError("Stage 3 requires vad.backend=silero")
    if asr_config.get("backend") != "whisper":
        raise ValueError("Stage 3 requires asr.backend=whisper")
    if dialogue_config.get("backend") != "deepseek":
        raise ValueError("Stage 3 requires dialogue.backend=deepseek")
    if tts_config.get("backend") != "edge":
        raise ValueError("Stage 3 requires tts.backend=edge")

    vad = SileroVAD(
        model_path=_model_path(
            config_path, _required(vad_config, "model_path", "vad")
        ),
        threshold=float(vad_config.get("threshold", 0.5)),
        min_speech_ms=int(vad_config.get("min_speech_ms", 250)),
        min_silence_ms=int(vad_config.get("min_silence_ms", 500)),
    )
    language = asr_config.get("language", "en")
    asr = WhisperASR(
        model_path=str(
            _model_path(config_path, _required(asr_config, "model_path", "asr"))
        ),
        device=str(asr_config.get("device", "cpu")),
        language=None if language == "auto" else str(language),
    )
    dialogue = DeepSeekDialogue(
        model=str(_required(dialogue_config, "model", "dialogue")),
        base_url=str(_required(dialogue_config, "base_url", "dialogue")),
        timeout_sec=float(dialogue_config.get("timeout_sec", 30.0)),
        temperature=float(dialogue_config.get("temperature", 0.7)),
    )
    tts = EdgeTTS(voice=str(tts_config.get("voice", "en-US-GuyNeural")))
    runtime = Runtime(
        vad=vad,
        asr=asr,
        dialogue=dialogue,
        tts=tts,
        dialogue_fallback_text=str(
            _required(runtime_config, "dialogue_fallback_text", "runtime")
        ),
        cooldown_ms=int(runtime_config.get("cooldown_ms", 200)),
    )
    return runtime, str(audio.get("host", "0.0.0.0")), int(audio.get("port", 8765))


def main() -> None:
    args = parse_args()
    try:
        config, config_path = load_config(args.config)
        runtime, host, port = build_runtime(config, config_path)
    except Exception as exc:
        raise SystemExit(f"runtime: error: {exc}") from None

    from .audio_server import AudioWebSocketServer

    server = AudioWebSocketServer(runtime, host=host, port=port)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("[runtime] stopped")


if __name__ == "__main__":
    main()
