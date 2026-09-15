"""Local current date/time tool with an explicit China timezone."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .logging_utils import log

SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


class CurrentDateTime:
    """Return the current wall clock in Asia/Shanghai."""

    def get_current_datetime(self) -> dict[str, str]:
        log("[tool] get_current_datetime")
        current = datetime.now(SHANGHAI_TIMEZONE)
        offset = current.strftime("%z")
        utc_offset = f"{offset[:3]}:{offset[3:]}"
        value = {
            "timezone": "Asia/Shanghai",
            "utc_offset": utc_offset,
            "datetime": current.isoformat(timespec="seconds"),
            "date": current.strftime("%Y-%m-%d"),
            "time": current.strftime("%H:%M:%S"),
            "weekday": current.strftime("%A"),
        }
        log("[tool] timezone=Asia/Shanghai")
        log(f"[tool] datetime={value['datetime']}")
        return value
