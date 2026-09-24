#!/usr/bin/env python3
"""Loop global_01 as six sparse key poses through a running Motor.

Each key pose is sent once as a neck_pose_set message to the existing model
socket (/tmp/neck_model.sock); the Motor handles it through the same core as
the console NeckPoseSet command. No trajectory is constructed, and the script
never starts or owns the Motor process.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import time

DEFAULT_SOCKET = "/tmp/neck_model.sock"
# (duration_sec, roll_deg, pitch_deg, yaw_deg)
KEY_POSES = (
    (2.5, 0.5, -0.2, 2.0),
    (2.5, 1.0, 0.3, 4.0),
    (3.0, 0.2, 0.8, 0.8),
    (3.0, -0.8, 0.2, -3.2),
    (2.5, -0.4, -0.4, -1.4),
    (2.5, 0.0, 0.0, 0.0),
)
NEUTRAL = (0.0, 0.0, 0.0)


class MotorUnavailableError(RuntimeError):
    """The running Motor's model socket is missing or refuses connections."""


def pose_message(pose_deg: tuple[float, float, float]) -> bytes:
    """Serialize one key pose for the Motor's shared NeckPoseSet core."""
    roll, pitch, yaw = pose_deg
    return json.dumps(
        {
            "type": "neck_pose_set",
            "slave_id": 0,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "yaw_deg": yaw,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def send_pose(socket_path: str, pose_deg: tuple[float, float, float]) -> None:
    """Deliver one key pose; EOF means the Motor finished parsing this message."""
    if not Path(socket_path).exists():
        raise MotorUnavailableError(
            f"Motor 未运行或接口不存在: {socket_path} "
            "(请先手动启动 master_stack_test 并确认 ready)"
        )
    payload = pose_message(pose_deg)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(socket_path)
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            while client.recv(4096):
                pass
    except OSError as exc:
        raise MotorUnavailableError(
            f"无法连接 Motor model socket {socket_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def send_neutral(socket_path: str, attempts: int = 3) -> tuple[bool, str]:
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        try:
            send_pose(socket_path, NEUTRAL)
            return True, ""
        except MotorUnavailableError as exc:
            last_error = str(exc)
            time.sleep(0.3 * attempt)
    return False, last_error


def log_pose(cycle: int, key: int, pose, duration: float, action: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(
        f"{stamp} {action} cycle={cycle} key={key} "
        f"target_roll_pitch_yaw_deg={pose} hold_sec={duration:g}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Motor model socket")
    parser.add_argument("--cycles", type=int, help="number of full 16s cycles (default: forever)")
    parser.add_argument("--dry-run", action="store_true", help="print payloads without connecting")
    args = parser.parse_args()
    if args.cycles is not None and args.cycles <= 0:
        parser.error("--cycles must be a positive integer")

    sent_any = False
    exit_code = 0
    cycle = 0
    try:
        while args.cycles is None or cycle < args.cycles:
            cycle += 1
            for key, (duration, *pose) in enumerate(KEY_POSES, 1):
                target = tuple(pose)
                if args.dry_run:
                    payload = pose_message(target).decode("utf-8")
                    log_pose(cycle, key, target, duration, "dry-run")
                    print(f"  payload={payload}", flush=True)
                else:
                    send_pose(args.socket, target)
                    sent_any = True
                    log_pose(cycle, key, target, duration, "sent")
                time.sleep(duration)
    except KeyboardInterrupt:
        print("Ctrl+C: sending neutral (0,0,0)", flush=True)
    except MotorUnavailableError as exc:
        print(f"Motor 错误: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if sent_any:
            ok, error = send_neutral(args.socket)
            if ok:
                log_pose(cycle, 0, NEUTRAL, 0.0, "sent neutral")
            else:
                print(
                    f"WARNING: neutral 发送失败: {error}; "
                    "请人工确认姿态或使用急停",
                    file=sys.stderr,
                    flush=True,
                )
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
