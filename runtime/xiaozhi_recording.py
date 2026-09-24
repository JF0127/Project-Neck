"""Sidecar recorder for XiaoZhi's per-turn Opus robot audio."""
from __future__ import annotations

import array
import ctypes
import ctypes.util
import json
import sys
import wave
from pathlib import Path


class RobotTurnRecorder:
    def __init__(self, output_root: Path | None = None):
        self.output_root = output_root or Path(__file__).parent / "generated/xiaozhi"
        self.closed_turns = set()
        self.discard()

    def disconnect(self):
        self.discard()
        self.closed_turns.clear()

    def discard(self, turn=None):
        if turn is not None:
            self.closed_turns.add(turn)
            if turn != getattr(self, "turn", None):
                return
        self.turn = None
        self.texts = []
        self.packets = []
        self.audio_format = None

    def _select(self, turn):
        if (not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0
                or turn in self.closed_turns):
            return False
        if self.turn != turn:
            self.discard()
            self.turn = turn
        return True

    def text(self, turn, value, _timestamp):
        if self._select(turn):
            self.texts.append(value)

    def audio(self, turn, sequence, payload, rate, duration, channels):
        if not self._select(turn):
            return
        audio_format = (rate, duration, channels)
        if self.audio_format is not None and audio_format != self.audio_format:
            print(f"[XIAOZHI] warning: robot_audio format changed in turn={turn}; discarding", flush=True)
            self.discard(turn)
            return
        self.audio_format = audio_format
        # The wire payload is an Opus packet, NOT PCM. Preserve wire order and decoder state.
        self.packets.append(payload)

    @staticmethod
    def _decode(packets, rate, channels):
        name = ctypes.util.find_library("opus")
        if not name:
            raise RuntimeError("libopus was not found")
        lib = ctypes.CDLL(name)
        lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.opus_decoder_create.restype = ctypes.c_void_p
        lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int]
        lib.opus_decode.restype = ctypes.c_int
        error = ctypes.c_int()
        decoder = lib.opus_decoder_create(rate, channels, ctypes.byref(error))
        if not decoder or error.value:
            if decoder:
                lib.opus_decoder_destroy(decoder)
            raise RuntimeError(f"Opus decoder init failed: {error.value}")
        try:
            max_samples = rate * 120 // 1000
            pcm = (ctypes.c_int16 * (max_samples * channels))()
            output = bytearray()
            for packet in packets:
                encoded = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
                count = lib.opus_decode(decoder, encoded, len(packet), pcm, max_samples, 0)
                if count < 0:
                    raise RuntimeError(f"Opus decode failed: {count}")
                data = ctypes.string_at(pcm, count * channels * 2)
                if sys.byteorder == "big":
                    samples = array.array("h")
                    samples.frombytes(data)
                    samples.byteswap()
                    data = samples.tobytes()
                output.extend(data)
            return output
        finally:
            lib.opus_decoder_destroy(decoder)

    def detach(self, turn):
        """Hand completed turn to a worker without sharing mutable state with the next turn."""
        if turn != self.turn:
            return None
        snapshot = RobotTurnRecorder(self.output_root)
        snapshot.turn = turn
        snapshot.texts = self.texts
        snapshot.packets = self.packets
        snapshot.audio_format = self.audio_format
        self.discard(turn)
        return snapshot

    def finish(self, turn, _timestamp):
        if turn != self.turn:
            return
        try:
            if not self.packets or self.audio_format is None:
                return
            rate, duration, channels = self.audio_format
            pcm = self._decode(self.packets, rate, channels)
            directory = self.output_root / f"turn_{turn:04d}"
            if directory.exists():
                raise FileExistsError(f"refusing to overwrite {directory}")
            directory.mkdir(parents=True)
            try:
                with wave.open(str(directory / "robot.wav"), "wb") as wav:
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    wav.writeframes(pcm)
                (directory / "metadata.json").write_text(json.dumps({
                    "turn_id": turn, "robot_text": "".join(self.texts),
                    "robot_text_sentences": self.texts, "robot_audio": {
                        "codec_source": "opus", "sample_rate": rate, "channels": channels,
                        "frame_duration_ms": duration, "packets": len(self.packets),
                        "samples": len(pcm) // (2 * channels),
                    },
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                import shutil
                shutil.rmtree(directory)
                raise
            print(f"[XIAOZHI] saved turn={turn} to {directory}", flush=True)
            return directory
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[XIAOZHI] warning: robot recording failed turn={turn}: {exc}", flush=True)
        finally:
            self.discard(turn)
