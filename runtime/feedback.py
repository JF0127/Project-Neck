"""Real-time head pose from the Motor NDJSON feedback socket.

The motor process pushes one ``motor_state`` JSON line per 33 ms to
``/tmp/neck_feedback.sock``. This module is a read-only client: it parses the
stream, converts the motor-native ``[pitch, roll, yaw]`` degrees into the
Runtime ``RobotState`` convention ``[roll, pitch, yaw]`` radians, and marks the
pose invalid when messages stop arriving. It never sends anything to the motor.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Callable

from .contracts import RobotState

MAX_LINE_BYTES = 64 * 1024
DEFAULT_SOCKET_PATH = "/tmp/neck_feedback.sock"
DEFAULT_STALE_SEC = 0.2
DEFAULT_RECONNECT_DELAY_SEC = 1.0
WATCHDOG_INTERVAL_SEC = 0.05

_MOTOR_ORDER = ["pitch", "roll", "yaw"]


class FeedbackProtocolError(ValueError):
    """A feedback line does not satisfy the frozen motor_state contract."""


@dataclass(frozen=True)
class MotorStateMessage:
    valid: bool
    timestamp_sec: float
    motion_executing: bool
    head_rpy_rad: tuple[float, float, float] | None
    reason: str = ""


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_vector(value: object, field: str) -> list[float]:
    if not (
        isinstance(value, list)
        and len(value) == 3
        and all(_is_finite_number(component) for component in value)
    ):
        raise FeedbackProtocolError(f"{field} must contain three finite numbers")
    return [float(component) for component in value]


def parse_motor_state(line: str) -> MotorStateMessage:
    """Validate one NDJSON line and normalize it into Runtime semantics."""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FeedbackProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeedbackProtocolError("message must be a JSON object")
    if payload.get("type") != "motor_state":
        raise FeedbackProtocolError("message type must be 'motor_state'")
    if payload.get("version") != 1:
        raise FeedbackProtocolError("unsupported motor_state version")

    valid = payload.get("valid")
    if not isinstance(valid, bool):
        raise FeedbackProtocolError("valid must be a boolean")
    timestamp = payload.get("timestamp")
    if not _is_finite_number(timestamp):
        raise FeedbackProtocolError("timestamp must be a finite number")
    motion_executing = payload.get("motion_executing")
    if not isinstance(motion_executing, bool):
        raise FeedbackProtocolError("motion_executing must be a boolean")

    if not valid:
        reason = payload.get("reason", "invalid")
        return MotorStateMessage(
            valid=False,
            timestamp_sec=float(timestamp),
            motion_executing=motion_executing,
            head_rpy_rad=None,
            reason=reason if isinstance(reason, str) else "invalid",
        )

    if payload.get("unit") != "degree":
        raise FeedbackProtocolError("valid motor_state must use degree")
    if payload.get("order") != _MOTOR_ORDER:
        raise FeedbackProtocolError(
            "valid motor_state order must be ['pitch', 'roll', 'yaw']"
        )
    pitch, roll, yaw = _require_vector(payload.get("head_rpy"), "head_rpy")
    for optional in ("motor_angles", "sequence"):
        if payload.get(optional) is not None:
            _require_vector(payload[optional], optional)

    return MotorStateMessage(
        valid=True,
        timestamp_sec=float(timestamp),
        motion_executing=motion_executing,
        # Motor order is pitch/roll/yaw; RobotState order is roll/pitch/yaw.
        head_rpy_rad=(math.radians(roll), math.radians(pitch), math.radians(yaw)),
    )


class MotorFeedbackMonitor:
    """Keep one ``RobotState`` updated from the Motor feedback socket."""

    def __init__(
        self,
        robot_state: RobotState,
        socket_path: str = DEFAULT_SOCKET_PATH,
        stale_sec: float = DEFAULT_STALE_SEC,
        reconnect_delay_sec: float = DEFAULT_RECONNECT_DELAY_SEC,
        watchdog_interval_sec: float = WATCHDOG_INTERVAL_SEC,
        clock: Callable[[], float] = time.monotonic,
        logger: Callable[[str], None] | None = print,
    ) -> None:
        if not math.isfinite(stale_sec) or stale_sec <= 0.0:
            raise ValueError("stale_sec must be a positive finite number")
        if not math.isfinite(reconnect_delay_sec) or reconnect_delay_sec <= 0.0:
            raise ValueError("reconnect_delay_sec must be a positive finite number")
        if not math.isfinite(watchdog_interval_sec) or watchdog_interval_sec <= 0.0:
            raise ValueError("watchdog_interval_sec must be a positive finite number")
        self.robot_state = robot_state
        self.socket_path = socket_path
        self.stale_sec = float(stale_sec)
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.watchdog_interval_sec = float(watchdog_interval_sec)
        self._clock = clock
        self._logger = logger

        self._task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._stopping = False
        self._connected = False
        self._wait_logged = False
        self._stale_logged = False
        self._last_message_monotonic: float | None = None
        self._interval_ema_sec: float | None = None
        self._last_valid: bool | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def message_age_sec(self) -> float | None:
        """Seconds since the last parsed message, or None before the first one."""
        if self._last_message_monotonic is None:
            return None
        return max(0.0, self._clock() - self._last_message_monotonic)

    @property
    def observed_rate_hz(self) -> float | None:
        """EMA of the observed NDJSON arrival rate, or None before two messages."""
        if self._interval_ema_sec is None or self._interval_ema_sec <= 0.0:
            return None
        return 1.0 / self._interval_ema_sec

    def start(self) -> None:
        """Start the reader and watchdog tasks on the running event loop."""
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        self._watchdog_task = asyncio.create_task(self._watchdog())

    async def stop(self) -> None:
        self._stopping = True
        tasks = [task for task in (self._task, self._watchdog_task) if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._task = None
        self._watchdog_task = None
        self._connected = False

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    def _set_connected(self, connected: bool, reason: str = "") -> None:
        if connected == self._connected:
            return
        self._connected = connected
        if connected:
            self._wait_logged = False
            self._log(f"[runtime][feedback] connected: {self.socket_path}")
        else:
            self._log(
                f"[runtime][feedback] disconnected: {reason or 'stream ended'} "
                f"(retrying every {self.reconnect_delay_sec:g}s)"
            )

    def apply_line(self, line: str) -> bool:
        """Apply one feedback line; malformed lines are dropped with a warning."""
        if len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
            self._log("[runtime][feedback] dropped message: line too long")
            return False
        try:
            message = parse_motor_state(line)
        except FeedbackProtocolError as exc:
            self._log(f"[runtime][feedback] dropped message: {exc}")
            return False

        arrival = self._clock()
        if self._last_message_monotonic is not None:
            interval = arrival - self._last_message_monotonic
            if 0.0 < interval < 10.0:
                self._interval_ema_sec = (
                    interval if self._interval_ema_sec is None
                    else 0.9 * self._interval_ema_sec + 0.1 * interval
                )
        self._last_message_monotonic = arrival
        self._stale_logged = False
        state = self.robot_state
        state.motion_executing = message.motion_executing

        if message.valid:
            assert message.head_rpy_rad is not None
            state.head_rpy = list(message.head_rpy_rad)
            state.head_rpy_timestamp = message.timestamp_sec
            state.head_rpy_valid = True
            state.motor_available = True
        else:
            state.head_rpy_valid = False
            state.motor_available = False

        if message.valid != self._last_valid:
            if message.valid:
                self._log(
                    "[runtime][feedback] pose valid: "
                    f"rpy_rad={[round(value, 5) for value in state.head_rpy]}"
                )
            else:
                self._log(
                    f"[runtime][feedback] pose invalid: {message.reason or 'invalid'}"
                )
        self._last_valid = message.valid
        return True

    def check_stale(self, now: float | None = None) -> bool:
        """Mark the pose invalid when no message arrived within stale_sec."""
        if self._last_message_monotonic is None:
            return False
        current = self._clock() if now is None else now
        if current - self._last_message_monotonic <= self.stale_sec:
            return False
        self.robot_state.head_rpy_valid = False
        self.robot_state.motor_available = False
        if not self._stale_logged:
            self._stale_logged = True
            self._log(
                f"[runtime][feedback] pose stale: no message for "
                f"{current - self._last_message_monotonic:.3f}s"
            )
        return True

    async def _watchdog(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.watchdog_interval_sec)
            self.check_stale()

    async def _run(self) -> None:
        while not self._stopping:
            writer = None
            try:
                reader, writer = await asyncio.open_unix_connection(
                    self.socket_path, limit=MAX_LINE_BYTES
                )
                self._set_connected(True)
                while True:
                    raw = await reader.readline()
                    if not raw:
                        raise ConnectionError("motor feedback stream ended")
                    self.apply_line(raw.decode("utf-8", errors="replace").strip())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_connected(False, f"{type(exc).__name__}: {exc}")
                if not self._wait_logged:
                    self._wait_logged = True
                    self._log(
                        f"[runtime][feedback] waiting for {self.socket_path}: {exc}"
                    )
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
            if not self._stopping:
                await asyncio.sleep(self.reconnect_delay_sec)
