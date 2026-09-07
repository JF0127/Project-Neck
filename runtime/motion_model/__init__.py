"""Frozen inference implementation for the currently deployed V3 checkpoint.

This package is owned by Runtime and is intentionally independent of the new
training project in ``algorithm/``.
"""

from .candidates import build_multi_candidate_model


def build_model(config: dict, vocab_size: int):
    """Build the only model type supported by the current deployed runtime."""
    model_type = config.get("model", {}).get("type")
    if model_type != "candidates":
        raise ValueError(f"Runtime V1 requires model.type=candidates, got {model_type!r}")
    return build_multi_candidate_model(config, vocab_size)
