"""CLI entry: python -m runtime."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = RUNTIME_ROOT / "models/neck_motion_v3/checkpoints/best.pt"
DEFAULT_WHISPER = RUNTIME_ROOT / "models/whisper-base-ct2"
DEFAULT_EXPERIMENT_ROOT = RUNTIME_ROOT / "experiments/v0_trajectory"
TRAJECTORY_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project-Neck Robot Runtime V1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--motion-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--language", default="en", help="Whisper language; use 'auto' for detection")
    parser.add_argument("--tts-voice", default="en-US-GuyNeural")
    parser.add_argument("--dialogue", choices=["fixed", "echo"], default="fixed")
    parser.add_argument(
        "--fixed-reply",
        default="I heard you. Thank you for talking with me.",
    )
    parser.add_argument("--neck-socket", default="/tmp/neck_model.sock")
    parser.add_argument(
        "--neck-measurement-socket", default="/tmp/neck_measurement.sock"
    )
    parser.add_argument(
        "--mock-neck",
        action="store_true",
        help="validate and print Neck JSON without connecting to the motor process",
    )
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument(
        "--experiment-root",
        default=str(DEFAULT_EXPERIMENT_ROOT),
        help="v0 trajectory experiment root",
    )
    parser.add_argument(
        "--no-experiment-log",
        action="store_true",
        help="disable best-effort Session/Turn experiment records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep ``python -m runtime --help`` usable without loading deployment dependencies.
    from .audio_server import AudioWebSocketServer
    from .dialogue import EchoDialogue, FixedDialogue
    from .experiment_logger import ExperimentLogger
    from .runtime import AlgorithmRuntime

    dialogue = EchoDialogue() if args.dialogue == "echo" else FixedDialogue(args.fixed_reply)
    experiment_logger = None
    if not args.no_experiment_log:
        try:
            experiment_logger = ExperimentLogger(
                checkpoint=args.checkpoint,
                runtime_mode="mock-neck" if args.mock_neck else "motor-socket",
                trajectory_fps=TRAJECTORY_FPS,
                root=args.experiment_root,
            )
        except Exception as exc:
            print(
                f"[runtime][experiment][warning] cannot start session: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        runtime = AlgorithmRuntime(
            checkpoint=args.checkpoint,
            whisper_model=args.whisper_model,
            dialogue=dialogue,
            tts_voice=args.tts_voice,
            asr_device=args.asr_device,
            motion_device=args.motion_device,
            asr_language=None if args.language == "auto" else args.language,
            neck_socket=args.neck_socket,
            neck_measurement_socket=args.neck_measurement_socket,
            mock_neck=args.mock_neck,
            num_candidates=args.num_candidates,
            experiment_logger=experiment_logger,
        )
        server = AudioWebSocketServer(runtime, host=args.host, port=args.port)
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("[runtime] stopped")
    finally:
        if experiment_logger is not None:
            experiment_logger.end_session()


if __name__ == "__main__":
    main()
