#!/usr/bin/env python3
"""Compatibility helper that delegates to the production ``python -m runtime`` entry."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
