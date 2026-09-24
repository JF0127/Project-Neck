"""Per-turn motion trajectory builder for Ubuntu V1; no Global Motion."""
from __future__ import annotations

from .contracts import FinalTrajectory
from .inference.deepseek_motion import DeepSeekMotionBackend
from .inference.motor_json import final_trajectory_to_motor_document
from .inference.trajectory_generator import TrajectoryGenerator
from .inference.trajectory_optimizer import TrajectoryOptimizer

FPS = 30


class UbuntuV1Mixer:
    """Builds one complete turn trajectory from the text Motion Planner output."""

    def __init__(self) -> None:
        self.optimizer = TrajectoryOptimizer()

    def turn_document(self, text_plan, duration: float, base_pose, name: str) -> dict:
        """Text plan + estimated duration -> one complete absolute RPY document."""
        timed = DeepSeekMotionBackend._for_generator(text_plan, duration)
        layers = TrajectoryGenerator().generate_layers(timed)
        # V4 output is a relative offset; re-base it on the measured pose.
        frames = [
            tuple(base + offset for base, offset in zip(base_pose, frame))
            for frame in layers.composed_raw
        ]
        optimized = self.optimizer.optimize(frames, FPS)
        final = FinalTrajectory(
            optimized, FPS, ("silent",) * len(optimized), len(optimized) / FPS
        )
        return final_trajectory_to_motor_document(final, name)
