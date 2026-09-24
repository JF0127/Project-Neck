"""Offline-only input adapter: saved XiaoZhi turn -> existing Motion V2 backend."""
from __future__ import annotations

import audioop
import json
from pathlib import Path
import wave

from .contracts import MotionRequest, RobotSpeech, RobotState, SessionContext, TurnContext
from .inference.deepseek_motion import DeepSeekMotionBackend
from .inference.trajectory_optimizer import TrajectoryOptimizer


def process_saved_turn(turn_dir: Path, motion_config: dict, *, client=None) -> Path:
    """Produce turn-local motion artifacts without touching feedback or Motor."""
    turn_dir = Path(turn_dir)
    metadata = json.loads((turn_dir / "metadata.json").read_text(encoding="utf-8"))
    text = metadata["robot_text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("saved XiaoZhi turn has no spoken robot_text")
    with wave.open(str(turn_dir / "robot.wav"), "rb") as wav:
        rate = wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE" or rate <= 0:
            raise ValueError("XiaoZhi robot.wav must be mono uncompressed PCM s16le")
        frames = wav.getnframes()
        if not frames:
            raise ValueError("XiaoZhi robot.wav is empty")
        pcm = wav.readframes(frames)
    duration = frames / rate  # Source WAV, not the resampled/rounded timeline.
    if rate != 16_000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
    speech = RobotSpeech(text=text, pcm_s16le=pcm, words=(), duration_sec=duration)
    request = MotionRequest.snapshot(
        TurnContext(str(metadata["turn_id"]), robot_text=text, robot_speech=speech),
        SessionContext("xiaozhi"), RobotState(),
    )
    backend = DeepSeekMotionBackend(
        model=motion_config.get("deepseek_model", "deepseek-flash"),
        timeout_sec=motion_config.get("deepseek_timeout_sec", 15.0),
        temperature=motion_config.get("deepseek_temperature", 0.2),
        max_output_tokens=motion_config.get("deepseek_max_output_tokens", 1024),
        output_dir=turn_dir, client=client,
    )
    output = backend.infer(request)
    optimized = TrajectoryOptimizer().optimize(output.rpy_offset, output.fps)
    destination = turn_dir / "optimized_relative_trajectory.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps({
        "robot_text": text, "duration_sec": duration, "fps": output.fps,
        "unit": "radian", "order": ["roll", "pitch", "yaw"],
        "representation": "rpy_offset", "trajectory": [list(frame) for frame in optimized],
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
