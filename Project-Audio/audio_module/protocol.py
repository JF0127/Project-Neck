"""JSON control messages for PCM streams."""

import json
from typing import Any, Dict

from . import config


class ProtocolError(ValueError):
    pass


def stream_start_message(stream_id: str, source: str) -> str:
    return json.dumps(
        {
            "type": "stream_start",
            "stream_id": stream_id,
            "source": source,
            "sample_rate": config.SAMPLE_RATE,
            "channels": config.CHANNELS,
            "format": config.PCM_FORMAT,
        },
        separators=(",", ":"),
    )


def stream_end_message(stream_id: str) -> str:
    return json.dumps(
        {"type": "stream_end", "stream_id": stream_id},
        separators=(",", ":"),
    )


def parse_control_message(message: str) -> Dict[str, Any]:
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("invalid JSON control message") from exc
    if not isinstance(data, dict) or data.get("type") not in {"stream_start", "stream_end"}:
        raise ProtocolError("unsupported control message")
    if not isinstance(data.get("stream_id"), str) or not data["stream_id"]:
        raise ProtocolError("stream_id must be a non-empty string")
    return data
