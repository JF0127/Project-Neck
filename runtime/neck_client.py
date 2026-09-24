"""Send the frozen 30 fps RPY JSON schema to the Neck Unix socket."""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket


class NeckClient:
    def __init__(
        self,
        socket_path: str = "/tmp/neck_model.sock",
        mock: bool = False,
        measurement_socket_path: str = "/tmp/neck_measurement.sock",
    ):
        self.socket_path = socket_path
        self.measurement_socket_path = measurement_socket_path
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

    def send_pose(self, roll_deg: float, pitch_deg: float, yaw_deg: float,
                  slave_id: int = 0) -> None:
        """Send one sparse neck pose to the Motor's shared NeckPoseSet core."""
        document = {
            "type": "neck_pose_set",
            "slave_id": int(slave_id),
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
        }
        if not all(math.isfinite(document[key])
                   for key in ("roll_deg", "pitch_deg", "yaw_deg")):
            raise ValueError("neck_pose_set angles must be finite")
        payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_count += 1
        if self.mock:
            print(f"[runtime][neck] mock send #{self.send_count}: {len(payload)} JSON bytes (neck_pose_set)")
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10.0)
            client.connect(self.socket_path)
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            while client.recv(4096):
                pass
        print(f"[runtime][neck] send #{self.send_count} complete: {self.socket_path} (neck_pose_set)")

    def _configure_measurement(
        self,
        trajectory_name: str,
        output_path: Path,
        turn_origin_unix_sec: float,
    ) -> None:
        values = (trajectory_name, str(output_path), f"{turn_origin_unix_sec:.9f}")
        if any("\n" in value or "\r" in value for value in values):
            raise ValueError("measurement metadata must not contain newlines")
        request = (
            f"NECK_MEASUREMENT_V1\n{trajectory_name}\n"
            f"{turn_origin_unix_sec:.9f}\n{output_path}\n"
        ).encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(self.measurement_socket_path)
            client.sendall(request)
            client.shutdown(socket.SHUT_WR)
            response = bytearray()
            while True:
                chunk = client.recv(1024)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > 4096:
                    raise RuntimeError("measurement socket response is too large")
        text = response.decode("utf-8", errors="replace").strip()
        if text != "OK":
            raise RuntimeError(f"measurement configuration rejected: {text or 'empty response'}")

    def send(
        self,
        document: dict,
        measured_output_path: Path | None = None,
        turn_origin_unix_sec: float | None = None,
    ) -> None:
        self.validate(document)
        self.print_summary(document)
        payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_count += 1
        if self.mock:
            print(f"[runtime][neck] mock send #{self.send_count}: {len(payload)} JSON bytes")
            return

        if measured_output_path is not None and turn_origin_unix_sec is not None:
            try:
                self._configure_measurement(
                    document["name"], measured_output_path, turn_origin_unix_sec
                )
            except Exception as exc:
                print(
                    f"[runtime][neck][warning] measured RPY disabled for this turn: "
                    f"{type(exc).__name__}: {exc}"
                )

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10.0)
            client.connect(self.socket_path)
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            while client.recv(4096):
                pass
        print(f"[runtime][neck] send #{self.send_count} complete: {self.socket_path}")
