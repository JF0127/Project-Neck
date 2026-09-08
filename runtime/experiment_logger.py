"""Best-effort Session/Turn/Motion Generation records for v0 runtime tests."""
from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence
import wave

DEFAULT_ROOT = Path(__file__).resolve().parent / "experiments/v0_trajectory"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class ExperimentLogger:
    """Write lightweight side-channel records without becoming a runtime dependency."""

    def __init__(
        self,
        checkpoint: str | Path,
        runtime_mode: str,
        trajectory_fps: float,
        root: str | Path = DEFAULT_ROOT,
    ) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._started_monotonic = time.monotonic()
        self._session_timeline: list[dict[str, Any]] = []
        self._turns: dict[str, dict[str, Any]] = {}
        self._ended = False

        now = datetime.now().astimezone()
        self.session_dir, self.session_id = self._create_session_dir(now)
        self.turns_dir = self.session_dir / "turns"
        self.turns_dir.mkdir()
        self.session_config = {
            "session_id": self.session_id,
            "created_at": now.isoformat(timespec="milliseconds"),
            "trajectory_fps": trajectory_fps,
            "rpy_unit": "radian",
            "rpy_order": ["roll", "pitch", "yaw"],
            "checkpoint": str(checkpoint),
            "runtime_mode": runtime_mode,
        }
        self._write_json(self.session_dir / "session_config.json", self.session_config)
        self._append_session_event("session_start")
        print(f"[runtime][experiment] session: {self.session_dir}")

    def _create_session_dir(self, now: datetime) -> tuple[Path, str]:
        sessions = self.root / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        date = now.strftime("%Y%m%d")
        number = 1
        while True:
            session_id = f"session_{date}_{number:03d}"
            path = sessions / session_id
            try:
                path.mkdir()
                return path, session_id
            except FileExistsError:
                number += 1

    def _timestamp(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_sec": round(time.monotonic() - self._started_monotonic, 6),
        }

    def _write_json(self, path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)

    @staticmethod
    def _warning(operation: str, error: Exception) -> None:
        print(f"[runtime][experiment][warning] {operation}: {type(error).__name__}: {error}")

    def _append_session_event(self, event: str, **fields: Any) -> None:
        entry = {"event": event, **fields, **self._timestamp()}
        self._session_timeline.append(entry)
        self._write_json(self.session_dir / "session_timeline.json", self._session_timeline)

    def start_turn(self, turn_number: int, user_stream_id: str | None = None) -> str:
        turn_id = f"turn_{turn_number:03d}"
        try:
            with self._lock:
                turn_dir = self.turns_dir / turn_id
                turn_dir.mkdir(exist_ok=False)
                generations_dir = turn_dir / "model_generations"
                generations_dir.mkdir()
                self._turns[turn_id] = {
                    "dir": turn_dir,
                    "generations_dir": generations_dir,
                    "timeline": [],
                    "generation_count": 0,
                    "dialogue": {"turn_id": turn_id},
                    "turn_origin_unix_sec": time.time(),
                    "ended": False,
                }
                if user_stream_id is not None:
                    self._turns[turn_id]["dialogue"]["user_stream_id"] = user_stream_id
                self._append_session_event("turn_start", turn_id=turn_id)
                self.record_event(turn_id, "turn_start")
        except Exception as exc:
            self._warning(f"start {turn_id}", exc)
        return turn_id

    def measured_rpy_target(self, turn_id: str) -> tuple[Path, float]:
        with self._lock:
            turn = self._turns[turn_id]
            return (
                (turn["dir"] / "measured_rpy.json").resolve(),
                float(turn["turn_origin_unix_sec"]),
            )

    def record_event(self, turn_id: str, event: str, **fields: Any) -> None:
        try:
            with self._lock:
                turn = self._turns[turn_id]
                turn["timeline"].append({"event": event, **fields, **self._timestamp()})
                self._write_json(turn["dir"] / "timeline.json", turn["timeline"])
        except Exception as exc:
            self._warning(f"record event {event} for {turn_id}", exc)

    def record_dialogue(
        self,
        turn_id: str,
        user_text: str | None = None,
        robot_text: str | None = None,
        user_words: list[dict] | None = None,
        user_stream_id: str | None = None,
        robot_stream_id: str | None = None,
        dialogue_backend: str | None = None,
        dialogue_model: str | None = None,
        dialogue_latency_sec: float | None = None,
        dialogue_usage: dict[str, int] | None = None,
    ) -> None:
        try:
            with self._lock:
                dialogue = self._turns[turn_id]["dialogue"]
                values = {
                    "user_text": user_text,
                    "robot_text": robot_text,
                    "user_words": user_words,
                    "user_stream_id": user_stream_id,
                    "robot_stream_id": robot_stream_id,
                    "dialogue_backend": dialogue_backend,
                    "dialogue_model": dialogue_model,
                    "dialogue_latency_sec": dialogue_latency_sec,
                }
                dialogue.update({key: value for key, value in values.items() if value is not None})
                if dialogue_usage is not None:
                    usage = {
                        key: int(dialogue_usage[key])
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if isinstance(dialogue_usage.get(key), int)
                        and dialogue_usage[key] >= 0
                    }
                    if usage:
                        dialogue["dialogue_usage"] = usage
                self._write_json(self._turns[turn_id]["dir"] / "dialogue.json", dialogue)
        except Exception as exc:
            self._warning(f"record dialogue for {turn_id}", exc)

    @staticmethod
    def _trajectory_list(trajectory: Any) -> list[list[float]]:
        values = trajectory.tolist() if hasattr(trajectory, "tolist") else trajectory
        if not isinstance(values, list) or not values:
            raise ValueError("trajectory must be a non-empty [N,3] array")
        result: list[list[float]] = []
        for frame in values:
            if not isinstance(frame, (list, tuple)) or len(frame) != 3:
                raise ValueError("trajectory must be a non-empty [N,3] array")
            converted = [float(component) for component in frame]
            if not all(math.isfinite(component) for component in converted):
                raise ValueError("trajectory contains NaN or Inf")
            result.append(converted)
        return result

    def record_robot_motion_inputs(
        self,
        turn_id: str,
        pcm_s16le: bytes,
        words: list[dict],
        duration_sec: float,
    ) -> None:
        """Save the exact TTS inputs supplied to Baseline V1."""
        try:
            if not pcm_s16le or len(pcm_s16le) % 2:
                raise ValueError("robot PCM must contain complete int16 samples")
            duration = float(duration_sec)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("robot duration_sec must be finite and positive")
            with self._lock:
                turn_dir = self._turns[turn_id]["dir"]
                path = turn_dir / "robot_audio.wav"
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                with wave.open(str(temporary), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16_000)
                    output.writeframes(pcm_s16le)
                temporary.replace(path)
                self._write_json(turn_dir / "robot_words.json", list(words))
                self._write_json(
                    turn_dir / "robot_motion_input.json",
                    {
                        "audio_file": "robot_audio.wav",
                        "words_file": "robot_words.json",
                        "sample_rate": 16_000,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "pcm_bytes": len(pcm_s16le),
                        "num_samples": len(pcm_s16le) // 2,
                        "duration_sec": duration,
                    },
                )
                self.record_event(
                    turn_id,
                    "robot_motion_inputs_recorded",
                    pcm_bytes=len(pcm_s16le),
                    word_count=len(words),
                    duration_sec=duration,
                )
        except Exception as exc:
            self._warning(f"record robot motion inputs for {turn_id}", exc)

    def record_generation(
        self,
        turn_id: str,
        role: str,
        trajectory: Any,
        fps: float,
        model: str | None = None,
        checkpoint: str | None = None,
        query_timestamps_sec: list[float] | None = None,
        representation: str | None = None,
    ) -> str | None:
        try:
            frames = self._trajectory_list(trajectory)
            with self._lock:
                turn = self._turns[turn_id]
                turn["generation_count"] += 1
                generation_id = f"generation_{turn['generation_count']:03d}"
                safe_role = _SAFE_NAME.sub("_", role).strip("_") or "unknown"
                document: dict[str, Any] = {
                    "generation_id": generation_id,
                    "role": role,
                    "fps": fps,
                    "unit": "radian",
                    "order": ["roll", "pitch", "yaw"],
                    "num_frames": len(frames),
                    "trajectory": frames,
                }
                if model is not None:
                    document["model"] = model
                if checkpoint is not None:
                    document["checkpoint"] = checkpoint
                if representation is not None:
                    document["representation"] = representation
                if query_timestamps_sec is not None:
                    timestamps = [float(value) for value in query_timestamps_sec]
                    if len(timestamps) != len(frames) or not all(
                        math.isfinite(value) for value in timestamps
                    ):
                        raise ValueError(
                            "query_timestamps_sec must be finite and match trajectory length"
                        )
                    document["query_timestamps_sec"] = timestamps
                path = turn["generations_dir"] / f"{generation_id}_{safe_role}.json"
                self._write_json(path, document)
                self.record_event(
                    turn_id,
                    f"{safe_role}_generation",
                    generation_id=generation_id,
                    num_frames=len(frames),
                )
                return generation_id
        except Exception as exc:
            self._warning(f"record {role} generation for {turn_id}", exc)
            return None

    def record_final_neck_trajectory(self, turn_id: str, document: dict) -> None:
        """Copy the exact final document trajectory immediately before NeckClient.send()."""
        try:
            frames = self._trajectory_list(document["trajectory"])
            states = list(document["states"])
            fps = float(document["fps"])
            if len(states) != len(frames):
                raise ValueError("trajectory/states length mismatch")
            record = {
                "turn_id": turn_id,
                "fps": document["fps"],
                "unit": document["unit"],
                "order": list(document["order"]),
                "num_frames": len(frames),
                "states": states,
                "trajectory": frames,
            }
            with self._lock:
                turn_dir = self._turns[turn_id]["dir"]
                self._write_json(turn_dir / "neck_rpy.json", record)
                self._write_neck_csv(turn_dir / "neck_rpy.csv", frames, states, fps)
                self.record_event(turn_id, "neck_rpy_recorded", num_frames=len(frames))
        except Exception as exc:
            self._warning(f"record final neck trajectory for {turn_id}", exc)

    @staticmethod
    def _write_neck_csv(
        path: Path, frames: Sequence[Sequence[float]], states: Sequence[str], fps: float
    ) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "time_sec", "state", "roll", "pitch", "yaw"])
            for index, (state, frame) in enumerate(zip(states, frames)):
                writer.writerow([index, index / fps, state, frame[0], frame[1], frame[2]])
        temporary.replace(path)

    def end_turn(self, turn_id: str, status: str = "complete", error: str | None = None) -> None:
        try:
            with self._lock:
                turn = self._turns.get(turn_id)
                if turn is None or turn["ended"]:
                    return
                fields: dict[str, Any] = {"status": status}
                if error is not None:
                    fields["error"] = error
                self.record_event(turn_id, "turn_end", **fields)
                turn["ended"] = True
                self._append_session_event("turn_end", turn_id=turn_id, status=status)
        except Exception as exc:
            self._warning(f"end {turn_id}", exc)

    def end_session(self) -> None:
        try:
            with self._lock:
                if self._ended:
                    return
                for turn_id, turn in self._turns.items():
                    if not turn["ended"]:
                        self.end_turn(turn_id, status="interrupted")
                self._append_session_event(
                    "session_end",
                    turn_count=len(self._turns),
                )
                self._ended = True
        except Exception as exc:
            self._warning("end session", exc)
