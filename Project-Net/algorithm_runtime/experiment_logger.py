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

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "experiments/v0_trajectory"
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
                    "ended": False,
                }
                if user_stream_id is not None:
                    self._turns[turn_id]["dialogue"]["user_stream_id"] = user_stream_id
                self._append_session_event("turn_start", turn_id=turn_id)
                self.record_event(turn_id, "turn_start")
        except Exception as exc:
            self._warning(f"start {turn_id}", exc)
        return turn_id

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
                }
                dialogue.update({key: value for key, value in values.items() if value is not None})
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

    def record_generation(
        self,
        turn_id: str,
        role: str,
        trajectory: Any,
        fps: float,
        candidate_index: int | None = None,
        energy_deg_per_s: float | None = None,
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
                if candidate_index is not None:
                    document["candidate_index"] = candidate_index
                if energy_deg_per_s is not None:
                    document["energy_deg_per_s"] = energy_deg_per_s
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
