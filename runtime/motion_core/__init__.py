"""Motion contracts and common postprocessing for the new Runtime.

This package is temporarily named ``motion_core`` because the legacy Runtime
still owns the top-level ``motion.py`` module.
"""

from .base import MotionBackend, validate_motion_output
from .processor import MotionProcessor

__all__ = ["MotionBackend", "MotionProcessor", "validate_motion_output"]
