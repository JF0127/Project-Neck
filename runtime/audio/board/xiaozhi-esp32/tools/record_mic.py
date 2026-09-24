#!/usr/bin/env python3
"""Standalone BoardBridge microphone check (no Runtime/Motor/ASR)."""
import argparse
import socket
import time
import wave
from pathlib import Path

from board_bridge_protocol import BoardBridgeParser, ProtocolError
from board_bridge_server import OpusDecoder, OpusDecodeError


RATE = 16000
CHANNELS = 1
WIDTH = 2


def record(output: Path, duration: float, host: str, port: int, timeout: float) -> None:
    if duration <= 0 or timeout <= 0:
        raise ValueError("duration and timeout must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    parser = BoardBridgeParser()
    decoder = OpusDecoder(RATE, CHANNELS)
    samples = packets = gaps = 0
    last_sequence = None
    started = time.monotonic()
    print(f"listening on {host}:{port}; output={output}; codec=Opus; WAV={RATE}Hz "
          f"{CHANNELS}ch PCM s16le; target={duration}s", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            server.settimeout(1)
            with wave.open(str(output), "wb") as wav:
                wav.setnchannels(CHANNELS)
                wav.setsampwidth(WIDTH)
                wav.setframerate(RATE)
                while samples < duration * RATE and time.monotonic() - started < timeout:
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        continue
                    print(f"board connected: {address}", flush=True)
                    parser = BoardBridgeParser()
                    client.settimeout(1)
                    with client:
                        while samples < duration * RATE and time.monotonic() - started < timeout:
                            try:
                                data = client.recv(4096)
                            except socket.timeout:
                                continue
                            if not data:
                                break
                            try:
                                events = parser.feed(data)
                            except ProtocolError as exc:
                                print(f"board protocol error: {exc}", flush=True)
                                break
                            for event in events:
                                if event[0] != "user_audio":
                                    continue
                                _, sequence, payload = event
                                if last_sequence is not None and sequence != (last_sequence + 1) & 0xFFFFFFFF:
                                    gaps += 1
                                    print(f"sequence gap: {last_sequence} -> {sequence}", flush=True)
                                last_sequence = sequence
                                try:
                                    pcm, count = decoder.decode(payload)
                                except OpusDecodeError as exc:
                                    print(f"decode failed seq={sequence}: {exc}", flush=True)
                                    continue
                                remaining = int(duration * RATE) - samples
                                wav.writeframesraw(pcm[:remaining * WIDTH])
                                samples += min(count, remaining)
                                packets += 1
                                if packets == 1 or packets % 50 == 0:
                                    print(f"packet={packets} seq={sequence} opus_bytes={len(payload)} "
                                          f"decoded_samples={count} pcm_bytes={len(pcm)} "
                                          f"recorded_sec={samples/RATE:.2f}", flush=True)
                                if samples >= duration * RATE:
                                    break
                    print("board disconnected or recording complete", flush=True)
    finally:
        decoder.close()
    print(f"saved {output}: {samples/RATE:.2f}s, packets={packets}, sequence_gaps={gaps}, "
          f"format=PCM s16le/{RATE}Hz/{CHANNELS}ch", flush=True)
    if samples < duration * RATE:
        raise RuntimeError("insufficient microphone audio; keep XiaoZhi in listening mode longer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/xiaozhi-mic.wav"))
    parser.add_argument("--duration", type=float, default=10.0, help="decoded audio seconds")
    parser.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    record(args.output, args.duration, args.host, args.port, args.timeout)


if __name__ == "__main__":
    main()
