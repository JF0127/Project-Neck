"""Send the frozen 30 fps RPY JSON schema to the Neck Unix socket."""
from __future__ import annotations

import json
import socket


class NeckClient:
    def __init__(self, socket_path: str = "/tmp/neck_model.sock", mock: bool = False):
        self.socket_path = socket_path
        self.mock = mock
        self.send_count = 0

    @staticmethod
    def validate(document: dict) -> None:
        required = {"name", "fps", "unit", "order", "trajectory", "states"}
        missing = required - document.keys()
        if missing:
            raise ValueError(f"neck JSON missing fields: {sorted(missing)}")
        if document["fps"] != 30.0 or document["unit"] != "radian":
            raise ValueError("neck JSON must use 30 fps and radians")
        if document["order"] != ["roll", "pitch", "yaw"]:
            raise ValueError("neck JSON order must be roll,pitch,yaw")
        trajectory = document["trajectory"]
        states = document["states"]
        if not trajectory or len(trajectory) != len(states):
            raise ValueError("neck trajectory/states must be non-empty and equal length")
        if any(len(frame) != 3 for frame in trajectory):
            raise ValueError("every neck trajectory frame must have three values")
        if any(state not in {"speaking", "listening", "silent"} for state in states):
            raise ValueError("invalid neck behavior state")

    @staticmethod
    def print_summary(document: dict) -> None:
        states = document["states"]
        counts = {state: states.count(state) for state in ("listening", "speaking", "silent")}
        print(
            "[runtime][neck] JSON: "
            f"fps={document['fps']}, frames={len(document['trajectory'])}, "
            f"states={counts}, first={document['trajectory'][0]}, "
            f"last={document['trajectory'][-1]}"
        )

    def send(self, document: dict) -> None:
        self.validate(document)
        self.print_summary(document)
        payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_count += 1
        if self.mock:
            print(f"[runtime][neck] mock send #{self.send_count}: {len(payload)} JSON bytes")
            return

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10.0)
            client.connect(self.socket_path)
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            while client.recv(4096):
                pass
        print(f"[runtime][neck] send #{self.send_count} complete: {self.socket_path}")
