"""Continuous shared-phase cyclic baseline motion in roll/pitch/yaw order."""
from __future__ import annotations

import math
import random
from typing import Callable


AXES = ("roll", "pitch", "yaw")
TAU = 2.0 * math.pi


def _smoothstep(value: float) -> float:
    u = min(1.0, max(0.0, value))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _lerp(start: float, end: float, weight: float) -> float:
    return start + (end - start) * weight


class GlobalMotionGenerator:
    """One unwrapped phase drives a smooth, gently changing three-axis cycle."""

    STATES = {"silent", "listening", "thinking", "speaking"}
    _INTEGRATION_STEP_SEC = 1.0 / 120.0

    def __init__(
        self,
        config: dict,
        seed: int | None = None,
        on_cycle: Callable[[int, dict], None] | None = None,
    ):
        self.config = config
        self._validate_config()
        self.rng = random.Random(0 if seed is None else seed)
        self.on_cycle = on_cycle
        self.state = "silent"
        self.phase = 0.0
        self.cycle_index = 0
        self.last_time: float | None = None
        self._cycle_start_time: float | None = None
        self.completed_cycle_durations: list[float] = []
        self.cycle_parameter_durations: list[float] = []
        self._current = self._base_parameters()
        self._next = self._vary_parameters()
        self._normalizers = self._wave_normalizers()
        self._style_start = self._style_target = self._configured_style("silent")
        self._transition_start = 0.0

    def _validate_config(self) -> None:
        for key in ("cycle_duration_sec", "state_transition_sec"):
            if not math.isfinite(float(self.config[key])) or float(self.config[key]) <= 0:
                raise ValueError(f"global_motion.{key} must be positive and finite")
        for key in ("cycle_duration_variation", "parameter_variation"):
            value = float(self.config[key])
            if not math.isfinite(value) or not 0 <= value <= 0.2:
                raise ValueError(f"global_motion.{key} must be in [0, 0.2]")
        for axis in AXES:
            amplitude = float(self.config[f"{axis}_amplitude_deg"])
            harmonic = float(self.config[f"{axis}_harmonic_ratio"])
            if not math.isfinite(amplitude) or amplitude <= 0:
                raise ValueError(f"invalid {axis} amplitude")
            if not math.isfinite(harmonic) or not 0 <= harmonic <= 0.25:
                raise ValueError(f"invalid {axis} harmonic ratio")
            for suffix in ("phase_deg", "harmonic_phase_deg"):
                if not math.isfinite(float(self.config[f"{axis}_{suffix}"])):
                    raise ValueError(f"invalid {axis} {suffix}")
        for state in self.STATES:
            for field in ("amplitude", "speed"):
                value = float(self.config["states"][state][field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {state} {field} multiplier")

    def _configured_style(self, state: str) -> tuple[float, float]:
        values = self.config["states"][state]
        return float(values["amplitude"]), float(values["speed"])

    def _style_at(self, timestamp: float) -> tuple[float, float]:
        progress = (timestamp - self._transition_start) / self.config["state_transition_sec"]
        weight = _smoothstep(progress)
        return tuple(_lerp(a, b, weight) for a, b in zip(self._style_start, self._style_target))

    def set_state(self, state: str) -> None:
        if state not in self.STATES:
            raise ValueError(f"invalid global motion state: {state}")
        if state == self.state:
            return
        timestamp = self.last_time if self.last_time is not None else 0.0
        self._style_start = self._style_at(timestamp)
        self._style_target = self._configured_style(state)
        self._transition_start = timestamp
        self.state = state

    def _base_parameters(self) -> dict:
        return {
            "duration": float(self.config["cycle_duration_sec"]),
            **{axis: float(self.config[f"{axis}_amplitude_deg"]) for axis in AXES},
            **{f"{axis}_harmonic": float(self.config[f"{axis}_harmonic_ratio"]) for axis in AXES},
        }

    def _vary_parameters(self) -> dict:
        base = self._base_parameters()
        result = {}
        for key, value in base.items():
            variation = self.config[
                "cycle_duration_variation" if key == "duration" else "parameter_variation"
            ]
            result[key] = value * (1.0 + self.rng.uniform(-variation, variation))
        return result

    def _parameters_at_phase(self) -> dict:
        weight = _smoothstep(self.phase / TAU)
        return {key: _lerp(self._current[key], self._next[key], weight) for key in self._current}

    def _wave_normalizers(self) -> dict[str, float]:
        result = {}
        variation = self.config["parameter_variation"]
        for axis in AXES:
            axis_phase = math.radians(self.config[f"{axis}_phase_deg"])
            harmonic_phase = math.radians(self.config[f"{axis}_harmonic_phase_deg"])
            nominal_h = self.config[f"{axis}_harmonic_ratio"]
            ratios = (nominal_h * (1 - variation), nominal_h * (1 + variation))
            result[axis] = max(
                abs(math.sin(phi + axis_phase) + h * math.sin(2 * phi + harmonic_phase))
                for h in ratios
                for phi in (TAU * index / 4096 for index in range(4096))
            )
        return result

    def _advance(self, timestamp: float) -> None:
        assert self.last_time is not None
        cursor = self.last_time
        while cursor < timestamp:
            step = min(self._INTEGRATION_STEP_SEC, timestamp - cursor)
            speed = self._style_at(cursor + step / 2)[1]
            duration = self._parameters_at_phase()["duration"]
            self.phase += TAU * step * speed / duration
            cursor += step
            if self.phase >= TAU:
                self.phase -= TAU
                self.cycle_index += 1
                if self._cycle_start_time is not None:
                    self.completed_cycle_durations.append(cursor - self._cycle_start_time)
                self._cycle_start_time = cursor
                self._current = self._next
                self._next = self._vary_parameters()
                self.cycle_parameter_durations.append(self._current["duration"])
                if self.on_cycle:
                    self.on_cycle(self.cycle_index, self._current.copy())

    def sample(self, timestamp: float) -> tuple[float, float, float]:
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp < self.last_time):
            raise ValueError("timestamps must be finite and nondecreasing")
        if self.last_time is None:
            self.last_time = timestamp
            self._cycle_start_time = timestamp
            self.cycle_parameter_durations.append(self._current["duration"])
            if self.on_cycle:
                self.on_cycle(self.cycle_index, self._current.copy())
        else:
            self._advance(timestamp)
            self.last_time = timestamp
        parameters = self._parameters_at_phase()
        amplitude_style = self._style_at(timestamp)[0]
        values = []
        for axis in AXES:
            axis_phase = math.radians(self.config[f"{axis}_phase_deg"])
            harmonic_phase = math.radians(self.config[f"{axis}_harmonic_phase_deg"])
            wave = (
                math.sin(self.phase + axis_phase)
                + parameters[f"{axis}_harmonic"] * math.sin(2 * self.phase + harmonic_phase)
            ) / self._normalizers[axis]
            values.append(math.radians(parameters[axis] * amplitude_style * wave))
        return tuple(values)
