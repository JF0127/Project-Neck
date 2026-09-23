"""Incremental parser for the existing BoardBridge JSON and Opus wire stream."""

import json
import struct

MAX_AUDIO_PAYLOAD = 8192
MAX_CONTROL_BYTES = 65536


class ProtocolError(ValueError):
    pass


class BoardBridgeParser:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        events = []
        while self.buffer:
            kind = self.buffer[0]
            if kind == 10:
                del self.buffer[0]
                continue
            if kind == 13:
                if len(self.buffer) < 2:
                    break
                if self.buffer[1] != 10:
                    raise ProtocolError("unexpected carriage return")
                del self.buffer[:2]
                continue
            if kind == ord("{"):
                end = self.buffer.find(b"\n")
                if end < 0:
                    if len(self.buffer) > MAX_CONTROL_BYTES:
                        raise ProtocolError("control message too large")
                    break
                if end > MAX_CONTROL_BYTES:
                    raise ProtocolError("control message too large")
                line = bytes(self.buffer[:end])
                del self.buffer[:end + 1]
                if line.strip():
                    try:
                        events.append(("message", json.loads(line)))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ProtocolError(f"invalid JSON: {exc}") from exc
                continue
            if kind == 1:
                if len(self.buffer) < 9:
                    break
                sequence, length = struct.unpack_from("!II", self.buffer, 1)
                if not 0 < length <= MAX_AUDIO_PAYLOAD:
                    raise ProtocolError(f"invalid user_audio length: {length}")
                if len(self.buffer) < 9 + length:
                    break
                payload = bytes(self.buffer[9:9 + length])
                del self.buffer[:9 + length]
                events.append(("user_audio", sequence, payload))
                continue
            if kind in (2, 3):
                header = 16 if kind == 2 else 20
                if len(self.buffer) < header:
                    break
                if kind == 2:
                    sequence, length, rate, duration, channels = struct.unpack_from("!IIIHB", self.buffer, 1)
                    turn_id = None
                else:
                    sequence, length, turn_id, rate, duration, channels = struct.unpack_from("!IIIIHB", self.buffer, 1)
                if not 0 < length <= MAX_AUDIO_PAYLOAD:
                    raise ProtocolError(f"invalid robot_audio length: {length}")
                if not rate or not duration or not channels:
                    raise ProtocolError("invalid robot_audio format")
                if len(self.buffer) < header + length:
                    break
                payload = bytes(self.buffer[header:header + length])
                del self.buffer[:header + length]
                events.append(("robot_audio", turn_id, sequence, payload, rate, duration, channels))
                continue
            raise ProtocolError(f"unknown frame type: 0x{kind:02x}")
        return events
