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


def build_motion_components(
    config: dict[str, Any], config_path: Path
):
    """Build the shared Motion/Motor objects without constructing an ASR Runtime."""
    from .contracts import RobotState
    from .feedback import MotorFeedbackMonitor

    dialogue_config = _section(config, "dialogue")
    motion_config = _section(config, "motion")
    motor_config = _section(config, "motor")
    robot_state = RobotState()
    output_dir = _model_path(
        config_path, str(motion_config.get("generated_dir", "generated"))
    )

    turn_generator = None
    if bool(motion_config.get("enabled", False)):
        from .inference import MotionProcessor, TurnGenerator

        backend_name = str(motion_config.get("backend", "baseline"))
        if backend_name == "baseline":
            from .inference.baseline_v1 import BaselineV1Backend

            backend = BaselineV1Backend(
                model_path=_model_path(
                    config_path, _required(motion_config, "model_path", "motion")
                ),
                vocab_path=_model_path(
                    config_path, _required(motion_config, "vocab_path", "motion")
                ),
                device=str(motion_config.get("device", "auto")),
            )
            model_label = str(backend.model_path)
        elif backend_name == "deepseek":
            from .inference.deepseek_motion import DeepSeekMotionBackend

            backend = DeepSeekMotionBackend(
                model=str(
                    motion_config.get(
                        "deepseek_model",
                        dialogue_config.get("model", "deepseek-flash"),
                    )
                ),
                base_url=str(
                    motion_config.get(
                        "deepseek_base_url",
                        dialogue_config.get("base_url", "https://api.deepseek.com"),
                    )
                ),
                timeout_sec=float(motion_config.get("deepseek_timeout_sec", 15.0)),
                temperature=float(motion_config.get("deepseek_temperature", 0.2)),
                max_output_tokens=int(
                    motion_config.get("deepseek_max_output_tokens", 1024)
                ),
                output_dir=output_dir,
            )
            model_label = f"deepseek:{backend.model}"
        else:
            raise ValueError("motion.backend must be baseline or deepseek")

        turn_generator = TurnGenerator(
            backend=backend,
            processor=MotionProcessor(),
            output_dir=output_dir,
            model_label=model_label,
        )

    send_to_motor = bool(
        turn_generator is not None
        and motion_config.get("send_to_motor", True)
        and motor_config.get("send_enabled", False)
    )
    neck_sender = None
    if send_to_motor:
        from .neck_client import NeckClient

        neck_sender = NeckClient(
            socket_path=str(motor_config.get("socket_path", "/tmp/neck_model.sock")),
            mock=bool(motor_config.get("mock", False)),
            measurement_socket_path=str(
                motor_config.get("measurement_socket", "/tmp/neck_measurement.sock")
            ),
        )

    monitor = None
    if send_to_motor and bool(motor_config.get("feedback_enabled", False)):
        monitor = MotorFeedbackMonitor(
            robot_state=robot_state,
            socket_path=str(
                motor_config.get("feedback_socket", "/tmp/neck_feedback.sock")
            ),
            stale_sec=float(motor_config.get("feedback_stale_sec", 0.2)),
        )
    return (
        robot_state,
        turn_generator,
        neck_sender,
        monitor,
        int(motion_config.get("sync_offset_ms", 0)),
        send_to_motor,
    )


def build_runtime(config: dict[str, Any], config_path: Path):
    from .asr import WhisperASR
    from .dialogue import DeepSeekDialogue
    from .runtime import Runtime
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
    if tts_config.get("backend") not in {"doubao", "edge"}:
        raise ValueError("tts.backend must be doubao or edge")

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
    if tts_config.get("backend") == "doubao":
        from .doubao_tts import DoubaoTTS

        tts = DoubaoTTS()
    else:
        from .tts import EdgeTTS

        tts = EdgeTTS(voice=str(tts_config.get("voice", "en-US-GuyNeural")))
    (
        robot_state,
        turn_generator,
        neck_sender,
        monitor,
        motion_sync_offset_ms,
        motion_send_to_motor,
    ) = build_motion_components(config, config_path)

    runtime = Runtime(
        vad=vad,
        asr=asr,
        dialogue=dialogue,
        tts=tts,
        dialogue_fallback_text=str(
            _required(runtime_config, "dialogue_fallback_text", "runtime")
        ),
        cooldown_ms=int(runtime_config.get("cooldown_ms", 200)),
        robot_state=robot_state,
        turn_generator=turn_generator,
        neck_sender=neck_sender,
        motion_sync_offset_ms=motion_sync_offset_ms,
        motion_send_to_motor=motion_send_to_motor,
    )
    return (
        runtime,
        monitor,
        str(audio.get("host", "0.0.0.0")),
        int(audio.get("port", 8765)),
    )


async def _serve(server, monitor) -> None:
    if monitor is not None:
        monitor.start()
    try:
        await server.serve_forever()
    finally:
        if monitor is not None:
            await monitor.stop()


def main() -> None:
    args = parse_args()
    try:
        config, config_path = load_config(args.config)
        runtime, monitor, host, port = build_runtime(config, config_path)
    except Exception as exc:
        raise SystemExit(f"runtime: error: {exc}") from None

    from .audio_server import AudioWebSocketServer

    server = AudioWebSocketServer(runtime, host=host, port=port)
    try:
        asyncio.run(_serve(server, monitor))
    except KeyboardInterrupt:
        print("[runtime] stopped")


if __name__ == "__main__":
    main()
