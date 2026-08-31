#!/usr/bin/env python3
"""Send one complete trajectory JSON to the local neck model socket."""

import argparse
import socket
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        nargs="?",
        default="-",
        help="trajectory JSON file, or '-' to read standard input",
    )
    parser.add_argument("--socket", default="/tmp/neck_model.sock")
    args = parser.parse_args()

    if args.json_file == "-":
        payload = sys.stdin.buffer.read()
    else:
        with open(args.json_file, "rb") as json_file:
            payload = json_file.read()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(args.socket)
        client.sendall(payload)
        # EOF on the client's write side marks the end of this JSON message.
        client.shutdown(socket.SHUT_WR)
        while client.recv(1024):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
