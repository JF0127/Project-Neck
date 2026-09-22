"""Model input construction, inference, output postprocessing and artifacts."""

from .artifacts import GenerationArtifacts, save_generation, write_wav_atomic
from .base import MOTION_FPS, MotionBackend, validate_motion_output
from .deepseek_motion import DeepSeekMotionBackend
from .default_motion import DEFAULT_GENERATION_FALLBACK_TEXT, default_motion_output
from .generator import GeneratedTurn, PoseUnavailableError, TurnGenerator
from .motion_plan import MotionPlan, MotionSegment
from .motion_plan_validator import validate_motion_plan
from .motor_json import final_trajectory_to_motor_document
from .processor import MotionProcessor
from .prosody import ProsodyAnalysis, SegmentProsody, extract_prosody
from .trajectory_generator import TrajectoryGenerator
from .trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerConfig,
    trajectory_metrics,
)

__all__ = [
    "DEFAULT_GENERATION_FALLBACK_TEXT",
    "DeepSeekMotionBackend",
    "GenerationArtifacts",
    "GeneratedTurn",
    "MOTION_FPS",
    "MotionBackend",
    "MotionPlan",
    "MotionProcessor",
    "MotionSegment",
    "PoseUnavailableError",
    "ProsodyAnalysis",
    "SegmentProsody",
    "TrajectoryGenerator",
    "TrajectoryOptimizer",
    "TrajectoryOptimizerConfig",
    "TurnGenerator",
    "default_motion_output",
    "extract_prosody",
    "final_trajectory_to_motor_document",
    "save_generation",
    "trajectory_metrics",
    "validate_motion_output",
    "validate_motion_plan",
    "write_wav_atomic",
]
