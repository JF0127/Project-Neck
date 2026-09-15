"""Small timestamped console logging helper for Runtime diagnostics."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


def log(message: str) -> None:
    timestamp = datetime.now(CHINA_TIMEZONE).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)
