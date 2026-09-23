"""Standalone XiaoZhi + always-running Global Motion mode."""
from __future__ import annotations

import asyncio
import csv
import math
from pathlib import Path
import time

from .contracts import FinalTrajectory, RobotState
from .feedback import MotorFeedbackMonitor
from .global_motion import GlobalMotionGenerator
from .inference.motor_json import final_trajectory_to_motor_document
from .inference.trajectory_optimizer import TrajectoryOptimizer, trajectory_metrics
from .neck_client import NeckClient
from .xiaozhi_adapter import XiaoZhiAdapter


class XiaoZhiMotionRuntime:
    def __init__(self, config: dict, *, dry_run: bool = False, seed: int | None = None):
        self.config = config
        self.dry_run = dry_run
        motion = config["global_motion"]
        self.fps = int(motion["fps"])
        if self.fps != 30:
            raise ValueError("Global Motion requires existing 30 fps Motor contract")
        self.chunk_frames = round(self.fps * motion["chunk_duration_sec"])
        if self.chunk_frames < 3:
            raise ValueError("global_motion chunk must contain at least 3 frames")
        self.generator = GlobalMotionGenerator(motion, seed=seed, on_cycle=self._on_cycle)
        self.transitions = [(0.0, "silent")]
        self.frames = []
        self.frame_states = []
        self.frame_cycles = []
        self.frame_index = 0
        self.motor_state = RobotState()
        motor = config.get("motor", {})
        self.feedback = MotorFeedbackMonitor(self.motor_state, socket_path=motor.get("feedback_socket", "/tmp/neck_feedback.sock"))
        self.neck = NeckClient(socket_path=motor.get("socket_path", "/tmp/neck_model.sock"))
        self.optimizer = TrajectoryOptimizer()
        self.adapter = XiaoZhiAdapter(self.set_state)
        self._start = None

    def _on_cycle(self, index, parameters):
        print(
            f"[GLOBAL] cycle={index} duration={parameters['duration']:.2f}s "
            f"yaw={parameters['yaw']:.2f} pitch={parameters['pitch']:.2f} "
            f"roll={parameters['roll']:.2f}",
            flush=True,
        )

    def set_state(self, state):
        if state != self.generator.state:
            self.generator.set_state(state)
            self.transitions.append((self.frame_index / self.fps, state))
            print(f"[GLOBAL] state={state} transition={self.config['global_motion']['state_transition_sec']:.1f}s", flush=True)

    def _next_frame(self):
        t = self.frame_index / self.fps
        baseline = self.generator.sample(t)
        # Future semantic motion can provide an offset here; V1 is identically zero.
        semantic_offset = (0.0, 0.0, 0.0)
        frame = tuple(a + b for a, b in zip(baseline, semantic_offset))
        self.frames.append(frame)
        self.frame_states.append(self.generator.state)
        self.frame_cycles.append(self.generator.cycle_index)
        self.frame_index += 1
        if not self.dry_run and len(self.frames) > self.chunk_frames * 3:
            del self.frames[:-self.chunk_frames * 3]
            del self.frame_states[:-self.chunk_frames * 3]
            del self.frame_cycles[:-self.chunk_frames * 3]
        return frame

    def _motor_state(self, state):
        return state if state in ("silent", "listening", "speaking") else "silent"

    def _document(self, frames, states, number):
        optimized = self.optimizer.optimize(frames, self.fps)
        trajectory = FinalTrajectory(
            rpy=optimized, fps=float(self.fps),
            states=tuple(self._motor_state(s) for s in states),
            duration_sec=len(optimized) / self.fps,
        )
        return final_trajectory_to_motor_document(trajectory, f"global_{number:06d}")

    async def _motor_loop(self):
        number = 0
        previous_end = None
        while True:
            # The Motor rejects an active trajectory; feedback is the only acceptance signal.
            if not self.feedback.connected:
                await asyncio.sleep(0.1)
                continue
            if previous_end is None and (not self.motor_state.motor_available or not self.motor_state.head_rpy_valid):
                await asyncio.sleep(0.1)
                continue
            if self.motor_state.motion_executing:
                await asyncio.sleep(1 / self.fps)
                continue
            number += 1
            if len(self.frames) < self.chunk_frames:
                await asyncio.sleep(0.1)
                continue
            frames = list(self.frames[-self.chunk_frames:])
            states = list(self.frame_states[-self.chunk_frames:])
            if previous_end is not None:
                frames[0] = previous_end
            else:
                measured = tuple(self.motor_state.head_rpy)
                if max(abs(math.degrees(v)) for v in measured) > 5.0:
                    print("[GLOBAL] warning: measured pose exceeds 5 deg; waiting for manual neutral", flush=True)
                    await asyncio.sleep(1.0)
                    continue
                target = frames[-1]
                for index in range(len(frames)):
                    u = index / (len(frames) - 1)
                    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
                    frames[index] = tuple(a + (b - a) * blend for a, b in zip(measured, target))
            document = self._document(frames, states, number)
            try:
                await asyncio.to_thread(self.neck.send, document)
            except (OSError, ValueError) as exc:
                print(f"[GLOBAL] warning: Motor send failed: {exc}", flush=True)
                await asyncio.sleep(0.5)
                continue
            candidate_end = tuple(document["trajectory"][-1])
            # Wait for start, then completion. No socket ACK exists.
            deadline = time.monotonic() + 1.0
            while not self.motor_state.motion_executing and time.monotonic() < deadline:
                await asyncio.sleep(1 / self.fps)
            if not self.motor_state.motion_executing:
                print("[GLOBAL] warning: Motor execution not confirmed; waiting for valid feedback", flush=True)
                await asyncio.sleep(0.5)
                continue
            previous_end = candidate_end
            while self.motor_state.motion_executing:
                await asyncio.sleep(1 / self.fps)

    async def _sample_loop(self, duration):
        start = time.monotonic()
        while duration is None or time.monotonic() - start < duration:
            self._next_frame()
            await asyncio.sleep(max(0.0, start + self.frame_index / self.fps - time.monotonic()))

    def summary(self, output: Path | None = None):
        if not self.frames:
            return {}
        metrics = trajectory_metrics(self.frames, self.fps)
        extrema = {}
        for index, axis in enumerate(("roll", "pitch", "yaw")):
            values = [math.degrees(frame[index]) for frame in self.frames]
            extrema[axis] = {"min_deg": min(values), "max_deg": max(values), **metrics[axis]}
        boundaries = []
        if self.dry_run:
            previous_end = None
            for start in range(0, len(self.frames), self.chunk_frames):
                frames = list(self.frames[start:start + self.chunk_frames])
                states = self.frame_states[start:start + self.chunk_frames]
                if not frames:
                    break
                if previous_end is not None:
                    frames[0] = previous_end
                document = self._document(frames, states, start // self.chunk_frames + 1)
                if previous_end is not None:
                    boundaries.append(max(
                        abs(math.degrees(a - b))
                        for a, b in zip(document["trajectory"][0], previous_end)
                    ))
                previous_end = tuple(document["trajectory"][-1])
        complete = self.generator.completed_cycle_durations
        result = {
            "frames": len(self.frames), "axes": extrema,
            "completed_cycles": self.generator.cycle_index,
            "cycle_duration_min_sec": min(complete) if complete else None,
            "cycle_duration_max_sec": max(complete) if complete else None,
            "configured_cycle_duration_min_sec": min(self.generator.cycle_parameter_durations),
            "configured_cycle_duration_max_sec": max(self.generator.cycle_parameter_durations),
            "chunk_boundary_position_error_deg": max(boundaries, default=0.0),
            "transitions": self.transitions,
        }
        print(f"[GLOBAL] summary: {result}", flush=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("time_sec", "roll_deg", "pitch_deg", "yaw_deg", "state", "cycle_index"))
                for index, (frame, state, cycle) in enumerate(zip(self.frames, self.frame_states, self.frame_cycles)):
                    writer.writerow((index / self.fps, *(math.degrees(v) for v in frame), state, cycle))
            print(f"[GLOBAL] CSV: {output}", flush=True)
        return result

    async def run(self, *, duration: float | None = None, output: Path | None = None):
        print(f"[GLOBAL] started fps={self.fps}", flush=True)
        motion = self.config["global_motion"]
        print(
            f"[GLOBAL] mode=cyclic cycle={motion['cycle_duration_sec']:.1f}s "
            f"yaw_amp={motion['yaw_amplitude_deg']:.1f}deg "
            f"pitch_amp={motion['pitch_amplitude_deg']:.1f}deg "
            f"roll_amp={motion['roll_amplitude_deg']:.1f}deg",
            flush=True,
        )
        print("[GLOBAL] state=silent", flush=True)
        tasks = [asyncio.create_task(self.adapter.serve(
            self.config["xiaozhi"]["host"], self.config["xiaozhi"]["port"]))]
        if self.dry_run and duration is None:
            duration = 60.0
        tasks.append(asyncio.create_task(self._sample_loop(duration)))
        if not self.dry_run:
            self.feedback.start()
            tasks.append(asyncio.create_task(self._motor_loop()))
        try:
            if self.dry_run:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
            else:
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not self.dry_run:
                await self.feedback.stop()
            self.summary(output)
