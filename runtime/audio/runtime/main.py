"""Audio Module command-line entry point (phase 1: local capture validation)."""

import argparse
import asyncio
import queue
import time
import wave
from pathlib import Path

from . import config
from .audio_capture import MicrophoneCapture
from .websocket_client import run_conversation, run_duplex_test, stream_microphone


def validate_config() -> None:
    assert config.SAMPLE_RATE == 16_000
    assert config.CHANNELS == 1
    assert config.DTYPE == "int16"
    assert config.SAMPLES_PER_FRAME == 320
    assert config.BYTES_PER_FRAME == 640
    print("Audio format OK: 16000 Hz, mono, int16, 20 ms, 320 samples, 640 bytes/frame")


def capture_test(duration: float, output: Path) -> None:
    if duration <= 0:
        raise ValueError("duration must be greater than zero")

    validate_config()
    captured = 0
    invalid = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Capturing microphone for {duration:.1f} seconds... Press Ctrl+C to stop.")
    capture = MicrophoneCapture()
    started_at = time.monotonic()
    try:
        capture.start()
        deadline = started_at + duration
        first_frame_received = False
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(config.CHANNELS)
            wav_file.setsampwidth(config.SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(config.SAMPLE_RATE)
            while time.monotonic() < deadline:
                try:
                    frame = capture.read_frame(timeout=0.5)
                except queue.Empty:
                    continue
                if not first_frame_received:
                    # PulseAudio may need time to resume a suspended source.
                    # Measure the requested audio duration from the first frame,
                    # not from process startup.
                    started_at = time.monotonic()
                    deadline = started_at + duration
                    first_frame_received = True
                captured += 1
                if len(frame) != config.BYTES_PER_FRAME:
                    invalid += 1
                    continue
                wav_file.writeframesraw(frame)
    finally:
        capture.stop()

    elapsed = time.monotonic() - started_at
    print(
        f"Capture result: {captured} frames in {elapsed:.2f}s; "
        f"invalid={invalid + capture.invalid_frames}, "
        f"dropped={capture.dropped_frames}, statuses={capture.callback_status_count}"
    )
    if captured == 0:
        raise RuntimeError("No microphone frames were captured")
    if invalid or capture.invalid_frames:
        raise RuntimeError("Captured data contained invalid frame sizes")
    print("PASS: every received microphone frame is 640 bytes")
    print(f"Saved WAV: {output.resolve()}")


def websocket_test(
    duration: float, websocket_url: str, wait_for_robot: bool, robot_timeout: float
) -> None:
    validate_config()
    print(f"Connecting to {websocket_url}")
    result = asyncio.run(
        stream_microphone(duration, websocket_url, wait_for_robot, robot_timeout)
    )
    print(
        f"Stream result: stream_id={result.stream_id}, frames={result.frames_sent}, "
        f"bytes={result.total_bytes}, elapsed={result.elapsed:.2f}s"
    )
    print(
        f"Capture health: dropped={result.dropped_frames}, invalid={result.invalid_frames}, "
        f"max_queue_depth={result.max_queue_depth}"
    )
    if result.playback is not None:
        print(
            f"Playback result: received_frames={result.playback.received_frames}, "
            f"played_frames={result.playback.played_frames}, "
            f"underruns={result.playback.underruns}, "
            f"max_queue_depth={result.playback.max_queue_depth}, "
            f"statuses={result.playback.callback_statuses}"
        )


def conversation(websocket_url: str) -> None:
    validate_config()
    print("Natural half-duplex conversation. Press Ctrl+C to stop.")
    asyncio.run(run_conversation(websocket_url))


def asr_stream(websocket_url: str) -> None:
    validate_config()
    print("Continuous microphone/robot audio loop. Press Ctrl+C to stop.")
    asyncio.run(run_conversation(websocket_url))


def duplex_test(turns: int, duration: float, websocket_url: str, robot_timeout: float) -> None:
    validate_config()
    results = asyncio.run(run_duplex_test(turns, duration, websocket_url, robot_timeout))
    for result in results:
        print(f"turn number: {result.turn_number}")
        print(f"  user stream_id: {result.user.stream_id}")
        print(
            f"  user frames / bytes: {result.user.frames_sent} / {result.user.total_bytes}"
        )
        print(f"  robot stream_id: {result.robot.stream_id}")
        print(
            f"  robot frames / bytes: "
            f"{result.robot.frames_received} / {result.robot.total_bytes}"
        )
        print(f"  playback underruns: {result.robot.playback.underruns}")
        print(
            f"  connection maintained: "
            f"{'yes' if result.connection_maintained else 'no'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Head Audio Module V1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-config", help="validate the frozen audio constants")
    capture_parser = subparsers.add_parser("capture-test", help="test the local microphone")
    capture_parser.add_argument("--duration", type=float, default=5.0, help="test length in seconds")
    capture_parser.add_argument(
        "--output", type=Path, default=Path("capture.wav"), help="output WAV path"
    )
    stream_parser = subparsers.add_parser("stream-test", help="stream the microphone over WebSocket")
    stream_parser.add_argument("--duration", type=float, default=5.0, help="stream length in seconds")
    stream_parser.add_argument("--url", default=config.WEBSOCKET_URL, help="WebSocket server URL")
    stream_parser.add_argument(
        "--wait-for-robot", action="store_true", help="wait for and play one robot stream"
    )
    stream_parser.add_argument(
        "--robot-timeout", type=float, default=30.0, help="seconds to wait for robot stream"
    )
    conversation_parser = subparsers.add_parser(
        "conversation", help="run continuous natural half-duplex conversation"
    )
    conversation_parser.add_argument(
        "--url", default=config.WEBSOCKET_URL, help="WebSocket server URL"
    )
    asr_parser = subparsers.add_parser(
        "asr-stream", help="run continuous microphone and robot playback turns"
    )
    asr_parser.add_argument(
        "--url", default=config.WEBSOCKET_URL, help="WebSocket server URL"
    )
    duplex_parser = subparsers.add_parser(
        "duplex-test", help="run multiple fixed-duration diagnostic turns"
    )
    duplex_parser.add_argument("--turns", type=int, default=2, help="number of turns (minimum 2)")
    duplex_parser.add_argument(
        "--duration", type=float, default=5.0, help="microphone duration per turn"
    )
    duplex_parser.add_argument("--url", default=config.WEBSOCKET_URL, help="WebSocket server URL")
    duplex_parser.add_argument(
        "--robot-timeout", type=float, default=30.0, help="seconds to wait per robot stream"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "check-config":
            validate_config()
        elif args.command == "capture-test":
            capture_test(args.duration, args.output)
        elif args.command == "stream-test":
            websocket_test(args.duration, args.url, args.wait_for_robot, args.robot_timeout)
        elif args.command == "conversation":
            conversation(args.url)
        elif args.command == "asr-stream":
            asr_stream(args.url)
        elif args.command == "duplex-test":
            duplex_test(args.turns, args.duration, args.url, args.robot_timeout)
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
