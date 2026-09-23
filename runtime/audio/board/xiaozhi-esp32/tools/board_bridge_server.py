#!/usr/bin/env python3

import argparse
import array
import ctypes
import ctypes.util
import json
import math
import socket
import struct
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

from board_bridge_protocol import BoardBridgeParser, ProtocolError


HOST = "0.0.0.0"
PORT = 8766
RECV_SIZE = 4096
USER_AUDIO_HEADER_SIZE = 9
LEGACY_ROBOT_AUDIO_HEADER_SIZE = 16
ROBOT_AUDIO_V2_HEADER_SIZE = 20
MAX_AUDIO_PAYLOAD = 8192
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_DURATION_MS = 60
FINISH_FLUSH_SECONDS = 0.3
SOCKET_POLL_SECONDS = 0.1
DEFAULT_OUTPUT_DIR = Path("artifacts/board_bridge")


class OpusDecodeError(Exception):
    pass


class OpusDecoder:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        library_name = ctypes.util.find_library("opus")
        if not library_name:
            raise RuntimeError("libopus was not found")

        try:
            self.library = ctypes.CDLL(library_name)
        except OSError as error:
            raise RuntimeError(f"failed to load libopus: {error}") from error

        self.library.opus_decoder_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.opus_decoder_create.restype = ctypes.c_void_p
        self.library.opus_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.opus_decode.restype = ctypes.c_int
        self.library.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.library.opus_decoder_destroy.restype = None
        self.library.opus_strerror.argtypes = [ctypes.c_int]
        self.library.opus_strerror.restype = ctypes.c_char_p

        error = ctypes.c_int()
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_frame_samples = sample_rate * 120 // 1000
        self.decoder = self.library.opus_decoder_create(
            sample_rate, channels, ctypes.byref(error)
        )
        if not self.decoder or error.value != 0:
            message = self._error_message(error.value)
            if self.decoder:
                self.library.opus_decoder_destroy(self.decoder)
                self.decoder = None
            raise RuntimeError(f"failed to create Opus decoder: {message}")

    def _error_message(self, error_code):
        message = self.library.opus_strerror(error_code)
        return message.decode("utf-8", errors="replace") if message else str(error_code)

    def decode(self, payload):
        encoded = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        pcm = (ctypes.c_int16 * (self.max_frame_samples * self.channels))()
        sample_count = self.library.opus_decode(
            self.decoder,
            encoded,
            len(payload),
            pcm,
            self.max_frame_samples,
            0,
        )
        if sample_count < 0:
            raise OpusDecodeError(self._error_message(sample_count))

        pcm_bytes = ctypes.string_at(
            pcm, sample_count * SAMPLE_WIDTH * self.channels
        )
        if sys.byteorder == "big":
            samples = array.array("h")
            samples.frombytes(pcm_bytes)
            samples.byteswap()
            pcm_bytes = samples.tobytes()
        return pcm_bytes, sample_count

    def close(self):
        if self.decoder:
            self.library.opus_decoder_destroy(self.decoder)
            self.decoder = None


class EventProfiler:
    PROFILE_FIELDS = (
        "robot_text_rx",
        "first_audio_rx",
        "first_audio_packet_rx",
        "playback_start_rx",
        "audio_end_rx",
        "playback_end_rx",
    )

    def __init__(self):
        self.events = {}
        self.board_events_ms = {}

    def record(self, turn_id, event_name, host_rx_mono=None, first_only=False):
        if host_rx_mono is None:
            host_rx_mono = time.monotonic()
        if turn_id is None:
            return host_rx_mono, False
        turn_events = self.events.setdefault(turn_id, {})
        if first_only and event_name in turn_events:
            return host_rx_mono, False
        turn_events[event_name] = host_rx_mono
        return host_rx_mono, True

    def record_board(self, turn_id, event_name, timestamp_ms, first_only=False):
        if turn_id is None or timestamp_ms is None:
            return False
        turn_events = self.board_events_ms.setdefault(turn_id, {})
        if first_only and event_name in turn_events:
            return False
        turn_events[event_name] = timestamp_ms
        return True

    @staticmethod
    def _duration(events, start, end):
        if start not in events or end not in events:
            return None
        return events[end] - events[start]

    def print_summary(self, turn_id):
        if turn_id is None:
            return
        events = self.events.get(turn_id, {})
        values = " ".join(
            f"{name}={events[name]:.6f}" if name in events else f"{name}=NA"
            for name in self.PROFILE_FIELDS
        )
        print(f"[TURN {turn_id}] profile {values}", flush=True)
        board_events = self.board_events_ms.get(turn_id, {})
        metrics = {
            "text_to_first_audio_ms": self._duration(
                board_events, "robot_text", "first_audio"
            ),
            "text_to_playback_start_ms": self._duration(
                board_events, "robot_text", "playback_start"
            ),
            "first_audio_to_playback_start_ms": self._duration(
                board_events, "first_audio", "playback_start"
            ),
            "playback_duration_ms": self._duration(
                board_events, "playback_start", "playback_end"
            ),
        }
        metric_values = " ".join(
            f"{name}={value}" if value is not None else f"{name}=NA"
            for name, value in metrics.items()
        )
        print(f"[TURN {turn_id}] board_profile {metric_values}", flush=True)


def turn_label(turn_id):
    return str(turn_id) if turn_id is not None else "?"


class SessionOutput:
    def __init__(self, output_dir):
        base_dir = Path(output_dir)
        session_name = f"session_{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = base_dir / session_name
        suffix = 2
        while self.session_dir.exists():
            self.session_dir = base_dir / f"{session_name}_{suffix}"
            suffix += 1
        self.session_dir.mkdir(parents=True)
        self.next_turn_id = 1
        print(f"BoardBridge recording session: {self.session_dir}", flush=True)

    def create_turn(self):
        turn_id = self.next_turn_id
        self.next_turn_id += 1
        turn_dir = self.session_dir / f"turn_{turn_id:03d}"
        turn_dir.mkdir()
        return turn_id, turn_dir


class RobotAudioStats:
    def __init__(self):
        self.received_packets = 0
        self.audio_bytes = 0
        self.sequence_gaps = 0
        self.last_sequence = None
        self.audio_format = None

    def handle_robot_audio(
        self, turn_id, sequence, payload, sample_rate, frame_duration_ms, channels
    ):
        if self.last_sequence is not None:
            expected = (self.last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                self.sequence_gaps += 1
                print(
                    f"[{timestamp()}] robot_audio sequence gap: "
                    f"expected={expected} got={sequence}",
                    flush=True,
                )
        self.last_sequence = sequence

        audio_format = (sample_rate, frame_duration_ms, channels)
        if self.audio_format is not None and audio_format != self.audio_format:
            old_rate, old_frame, old_channels = self.audio_format
            print(
                f"[{timestamp()}] robot_audio format changed: "
                f"{old_rate}Hz/{old_frame}ms/ch{old_channels} -> "
                f"{sample_rate}Hz/{frame_duration_ms}ms/ch{channels}",
                flush=True,
            )
        self.audio_format = audio_format
        self.received_packets += 1
        self.audio_bytes += len(payload)

        if self.received_packets % 20 == 0:
            print(
                f"[TURN {turn_label(turn_id)}] robot_audio: packets={self.received_packets} "
                f"bytes={self.audio_bytes} last_seq={sequence} "
                f"rate={sample_rate}Hz frame={frame_duration_ms}ms ch={channels}",
                flush=True,
            )


class TurnRecorder:
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    FINISHING = "FINISHING"
    FINALIZED = "FINALIZED"

    def __init__(self, record, session_output, debug_audio=False, clock=time.monotonic):
        self.record = record
        self.session_output = session_output
        self.debug_audio = debug_audio
        self.clock = clock
        self.user_decoder = OpusDecoder() if record else None
        self.robot_decoder = None
        self._reset_turn_data()

    def _empty_audio_stats(self):
        return {
            "received_packets": 0,
            "audio_bytes": 0,
            "decoded_packets": 0,
            "decode_errors": 0,
            "sequence_gaps": 0,
            "samples": 0,
            "first_sequence": None,
            "last_sequence": None,
        }

    def _reset_turn_data(self):
        self.active = False
        self.turn_id = None
        self.turn_dir = None
        self.user_wav = None
        self.robot_wav = None
        self.user_text = None
        self.robot_text_sentences = []
        self.robot_turn_id = None
        self.user_state = self.IDLE
        self.robot_state = self.IDLE
        self.user_finish_deadline = None
        self.robot_finish_deadline = None
        self.user_stats = self._empty_audio_stats()
        self.robot_stats = self._empty_audio_stats()
        self.robot_format = None

    def _start_turn(self):
        self._reset_turn_data()
        self.active = True
        self.turn_id, self.turn_dir = self.session_output.create_turn()
        self.user_wav = wave.open(str(self.turn_dir / "user_audio.wav"), "wb")
        self.user_wav.setnchannels(CHANNELS)
        self.user_wav.setsampwidth(SAMPLE_WIDTH)
        self.user_wav.setframerate(SAMPLE_RATE)
        self.user_state = self.RECORDING
        print(f"User turn {self.turn_id:03d} started", flush=True)

    def _update_stats(self, stats, stream_name, sequence, payload_length):
        if stats["last_sequence"] is not None:
            expected = (stats["last_sequence"] + 1) & 0xFFFFFFFF
            if sequence != expected:
                stats["sequence_gaps"] += 1
                print(
                    f"[{timestamp()}] {stream_name} sequence gap: "
                    f"expected={expected} got={sequence}",
                    flush=True,
                )
        if stats["first_sequence"] is None:
            stats["first_sequence"] = sequence
        stats["last_sequence"] = sequence
        stats["received_packets"] += 1
        stats["audio_bytes"] += payload_length

    def _debug_pcm(self, stream_name, sequence, payload, sample_count, pcm, packet_count):
        if not self.debug_audio or not (packet_count <= 10 or packet_count % 20 == 0):
            return
        pcm_values = [value[0] for value in struct.iter_unpack("<h", pcm)]
        pcm_rms = math.sqrt(sum(value * value for value in pcm_values) / len(pcm_values))
        print(
            f"[{timestamp()}] {stream_name} debug: seq={sequence} "
            f"opus={len(payload)} decoded={sample_count} "
            f"min={min(pcm_values)} max={max(pcm_values)} rms={pcm_rms:.1f}",
            flush=True,
        )

    def handle_user_audio(self, sequence, payload):
        self.check_deadlines()
        if not self.record:
            self._update_stats(self.user_stats, "user_audio", sequence, len(payload))
            packet_count = self.user_stats["received_packets"]
            if packet_count == 1 or packet_count % 20 == 0:
                print(
                    f"[{timestamp()}] user_audio: packets={packet_count} "
                    f"bytes={self.user_stats['audio_bytes']} last_seq={sequence}",
                    flush=True,
                )
            return
        if self.active and self.user_state == self.FINALIZED:
            self.finalize_turn("missing_robot_audio_end")
        if not self.active:
            self._start_turn()

        self._update_stats(self.user_stats, "user_audio", sequence, len(payload))
        packet_count = self.user_stats["received_packets"]
        if packet_count == 1 or packet_count % 20 == 0:
            print(
                f"[{timestamp()}] user_audio: packets={packet_count} "
                f"bytes={self.user_stats['audio_bytes']} last_seq={sequence}",
                flush=True,
            )
        try:
            pcm, sample_count = self.user_decoder.decode(payload)
        except OpusDecodeError as error:
            self.user_stats["decode_errors"] += 1
            print(f"[{timestamp()}] user_audio decode error seq={sequence}: {error}", flush=True)
            return
        self._debug_pcm("user_audio", sequence, payload, sample_count, pcm, packet_count)
        self.user_wav.writeframesraw(pcm)
        self.user_stats["decoded_packets"] += 1
        self.user_stats["samples"] += sample_count

    def handle_user_text(self, text):
        self.check_deadlines()
        if not self.record:
            return
        if not self.active:
            print(f"[{timestamp()}] warning: user_text received without active turn", flush=True)
            return
        if self.user_text is not None:
            print(f"[{timestamp()}] warning: duplicate user_text ignored", flush=True)
            return
        self.user_text = text
        self.user_state = self.FINISHING
        self.user_finish_deadline = self.clock() + FINISH_FLUSH_SECONDS
        print(f"User turn {self.turn_id:03d} user audio finishing...", flush=True)

    def _finalize_user_audio(self):
        if self.user_wav is not None:
            self.user_wav.close()
            self.user_wav = None
        self.user_state = self.FINALIZED
        self.user_finish_deadline = None
        print(f"User turn {self.turn_id:03d} user audio phase finalized", flush=True)

    def _track_robot_turn(self, turn_id):
        if turn_id is None:
            return
        if self.robot_turn_id is None:
            self.robot_turn_id = turn_id
        elif self.robot_turn_id != turn_id:
            print(
                f"[{timestamp()}] warning: robot turn changed inside recorder: "
                f"{self.robot_turn_id} -> {turn_id}",
                flush=True,
            )

    def handle_robot_text(self, turn_id, text):
        self.check_deadlines()
        if not self.record:
            return
        if not self.active:
            print(f"[{timestamp()}] warning: robot_text received without active turn", flush=True)
            return
        self._track_robot_turn(turn_id)
        self.robot_text_sentences.append(text)

    def _start_robot_audio(self, sample_rate, frame_duration_ms, channels):
        try:
            self.robot_decoder = OpusDecoder(sample_rate, channels)
        except RuntimeError as error:
            print(f"[{timestamp()}] robot_audio decoder init failed: {error}", flush=True)
            return False
        self.robot_format = (sample_rate, frame_duration_ms, channels)
        self.robot_wav = wave.open(str(self.turn_dir / "robot_audio.wav"), "wb")
        self.robot_wav.setnchannels(channels)
        self.robot_wav.setsampwidth(SAMPLE_WIDTH)
        self.robot_wav.setframerate(sample_rate)
        self.robot_state = self.RECORDING
        return True

    def handle_robot_audio(
        self, turn_id, sequence, payload, sample_rate, frame_duration_ms, channels
    ):
        self.check_deadlines()
        if not self.record:
            return
        if not self.active:
            print(f"[{timestamp()}] warning: robot_audio received without active turn", flush=True)
            return

        self._track_robot_turn(turn_id)

        self._update_stats(self.robot_stats, "robot_audio", sequence, len(payload))
        packet_count = self.robot_stats["received_packets"]
        audio_format = (sample_rate, frame_duration_ms, channels)
        if self.robot_format is None:
            if not self._start_robot_audio(*audio_format):
                self.robot_stats["decode_errors"] += 1
                return
        elif audio_format != self.robot_format:
            self.robot_stats["decode_errors"] += 1
            print(
                f"[{timestamp()}] warning: robot_audio format changed; packet rejected: "
                f"{self.robot_format} -> {audio_format}",
                flush=True,
            )
            return

        if packet_count % 20 == 0:
            print(
                f"[TURN {turn_label(turn_id)}] robot_audio: packets={packet_count} "
                f"bytes={self.robot_stats['audio_bytes']} last_seq={sequence} "
                f"rate={sample_rate}Hz frame={frame_duration_ms}ms ch={channels}",
                flush=True,
            )
        try:
            pcm, sample_count = self.robot_decoder.decode(payload)
        except OpusDecodeError as error:
            self.robot_stats["decode_errors"] += 1
            print(f"[{timestamp()}] robot_audio decode error seq={sequence}: {error}", flush=True)
            return
        self._debug_pcm("robot_audio", sequence, payload, sample_count, pcm, packet_count)
        self.robot_wav.writeframesraw(pcm)
        self.robot_stats["decoded_packets"] += 1
        self.robot_stats["samples"] += sample_count

    def handle_robot_audio_end(self, turn_id):
        self.check_deadlines()
        if not self.record:
            return
        if not self.active:
            print(f"[{timestamp()}] warning: robot_audio_end received without active turn", flush=True)
            return
        self._track_robot_turn(turn_id)
        if self.robot_state == self.FINISHING:
            print(f"[{timestamp()}] warning: duplicate robot_audio_end ignored", flush=True)
            return
        self.robot_state = self.FINISHING
        self.robot_finish_deadline = self.clock() + FINISH_FLUSH_SECONDS
        print(f"User turn {self.turn_id:03d} robot audio finishing...", flush=True)

    def check_deadlines(self):
        now = self.clock()
        if self.user_state == self.FINISHING and now >= self.user_finish_deadline:
            self._finalize_user_audio()
        if self.robot_state == self.FINISHING and now >= self.robot_finish_deadline:
            self.finalize_turn("robot_audio_end")

    def _audio_metadata(self, stats, sample_rate, frame_duration_ms, channels):
        return {
            "codec_source": "opus",
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": SAMPLE_WIDTH,
            "frame_duration_ms": frame_duration_ms,
            "received_packets": stats["received_packets"],
            "decoded_packets": stats["decoded_packets"],
            "decode_errors": stats["decode_errors"],
            "sequence_gaps": stats["sequence_gaps"],
            "samples": stats["samples"],
            "duration_seconds": round(stats["samples"] / sample_rate, 6),
            "first_sequence": stats["first_sequence"],
            "last_sequence": stats["last_sequence"],
        }

    def finalize_turn(self, completed_by):
        if not self.active:
            return
        if self.user_wav is not None:
            self.user_wav.close()
            self.user_wav = None
        if self.robot_wav is not None:
            self.robot_wav.close()
            self.robot_wav = None
        if self.robot_decoder is not None:
            self.robot_decoder.close()
            self.robot_decoder = None

        robot_text = "".join(self.robot_text_sentences)
        (self.turn_dir / "user_text.txt").write_text(self.user_text or "", encoding="utf-8")
        (self.turn_dir / "robot_text.txt").write_text(
            "\n".join(self.robot_text_sentences), encoding="utf-8"
        )
        robot_audio = None
        robot_duration = 0.0
        if self.robot_format is not None:
            robot_audio = self._audio_metadata(self.robot_stats, *self.robot_format)
            robot_duration = robot_audio["duration_seconds"]
        metadata = {
            "turn_id": self.turn_id,
            "robot_turn_id": self.robot_turn_id,
            "user_text": self.user_text,
            "user_audio": self._audio_metadata(
                self.user_stats, SAMPLE_RATE, FRAME_DURATION_MS, CHANNELS
            ),
            "robot_text": robot_text,
            "robot_text_sentences": self.robot_text_sentences,
            "robot_audio": robot_audio,
            "completed_by": completed_by,
        }
        (self.turn_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        user_duration = self.user_stats["samples"] / SAMPLE_RATE
        print(f"Turn {self.turn_id:03d} saved:", flush=True)
        print(f"  user_audio: {user_duration:.2f} s", flush=True)
        print(f"  user_text: {self.user_text or '<none>'}", flush=True)
        print(f"  robot_audio: {robot_duration:.2f} s", flush=True)
        print(f"  robot_text: {robot_text or '<none>'}", flush=True)
        print(f"  completed_by: {completed_by}", flush=True)
        self._reset_turn_data()

    def close(self, completed_by):
        self.finalize_turn(completed_by)
        if self.user_decoder is not None:
            self.user_decoder.close()
            self.user_decoder = None
        if self.robot_decoder is not None:
            self.robot_decoder.close()
            self.robot_decoder = None


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def message_turn_id(message):
    turn_id = message.get("turn_id")
    if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 0:
        return None
    return turn_id


def message_timestamp_ms(message):
    timestamp_ms = message.get("timestamp_ms")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        return None
    return timestamp_ms


def handle_text_message(line, turn_recorder, profiler):
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        content = line.decode("utf-8", errors="replace")
        print(f"[{timestamp()}] invalid json: {content}", flush=True)
        return

    if isinstance(message, dict) and message.get("type") == "user_text":
        text = message.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        host_rx_mono = time.monotonic()
        print(
            f"[{timestamp()}] user_text: {text} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
        turn_recorder.handle_user_text(text)
    elif isinstance(message, dict) and message.get("type") == "robot_text":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        text = message.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        host_rx_mono, _ = profiler.record(
            turn_id, "robot_text_rx", first_only=True
        )
        profiler.record_board(
            turn_id, "robot_text", board_timestamp_ms, first_only=True
        )
        print(
            f"[TURN {turn_label(turn_id)}] robot_text: {text} "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
        turn_recorder.handle_robot_text(turn_id, text)
    elif isinstance(message, dict) and message.get("type") == "robot_first_audio":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        host_rx_mono, _ = profiler.record(
            turn_id, "first_audio_rx", first_only=True
        )
        profiler.record_board(
            turn_id, "first_audio", board_timestamp_ms, first_only=True
        )
        print(
            f"[TURN {turn_label(turn_id)}] robot_first_audio "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
    elif isinstance(message, dict) and message.get("type") == "robot_audio_end":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        host_rx_mono, _ = profiler.record(turn_id, "audio_end_rx")
        profiler.record_board(turn_id, "audio_end", board_timestamp_ms)
        print(
            f"[TURN {turn_label(turn_id)}] tts_stream_end "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
        turn_recorder.handle_robot_audio_end(turn_id)
    elif isinstance(message, dict) and message.get("type") == "robot_playback_start":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        host_rx_mono, _ = profiler.record(turn_id, "playback_start_rx")
        profiler.record_board(turn_id, "playback_start", board_timestamp_ms)
        print(
            f"[TURN {turn_label(turn_id)}] playback_start "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
    elif isinstance(message, dict) and message.get("type") == "robot_playback_end":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        host_rx_mono, _ = profiler.record(turn_id, "playback_end_rx")
        profiler.record_board(turn_id, "playback_end", board_timestamp_ms)
        print(
            f"[TURN {turn_label(turn_id)}] playback_end "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
        profiler.print_summary(turn_id)
    elif isinstance(message, dict) and message.get("type") == "robot_playback_abort":
        turn_id = message_turn_id(message)
        board_timestamp_ms = message_timestamp_ms(message)
        reason = message.get("reason", "")
        host_rx_mono, _ = profiler.record(turn_id, "playback_abort_rx")
        print(
            f"[TURN {turn_label(turn_id)}] playback_abort reason={reason} "
            f"board_ts={board_timestamp_ms} host_rx_mono={host_rx_mono:.6f}",
            flush=True,
        )
        profiler.print_summary(turn_id)
    else:
        content = json.dumps(message, ensure_ascii=False)
        print(f"[{timestamp()}] unknown: {content}", flush=True)


def receive_client(client, turn_recorder, robot_audio_stats, profiler):
    parser = BoardBridgeParser()
    client.settimeout(SOCKET_POLL_SECONDS)

    while True:
        try:
            data = client.recv(RECV_SIZE)
        except socket.timeout:
            turn_recorder.check_deadlines()
            continue
        if not data:
            return
        turn_recorder.check_deadlines()
        for event in parser.feed(data):
            if event[0] == "message":
                handle_text_message(json.dumps(event[1], ensure_ascii=False).encode("utf-8"), turn_recorder, profiler)
            elif event[0] == "user_audio":
                _, sequence, payload = event
                turn_recorder.handle_user_audio(sequence, payload)
            elif event[0] == "robot_audio":
                _, turn_id, sequence, payload, sample_rate, frame_duration_ms, channels = event
                host_rx_mono, is_first = profiler.record(
                    turn_id, "first_audio_packet_rx", first_only=True
                )
                if is_first:
                    print(
                        f"[TURN {turn_label(turn_id)}] first_robot_audio "
                        f"sr={sample_rate} frame={frame_duration_ms}ms "
                        f"host_rx_mono={host_rx_mono:.6f}",
                        flush=True,
                    )
                robot_audio_stats.handle_robot_audio(
                    turn_id,
                    sequence,
                    payload,
                    sample_rate,
                    frame_duration_ms,
                    channels,
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Receive Project-Neck BoardBridge data")
    parser.add_argument(
        "--record-turns",
        action="store_true",
        help="record complete user/robot turns",
    )
    parser.add_argument(
        "--record-user-audio",
        action="store_true",
        help="deprecated alias for --record-turns",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"session output parent directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="print bounded Opus/PCM diagnostics while recording",
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output:
        parser.error("--output is deprecated; use --output-dir")
    args.record_turns = args.record_turns or args.record_user_audio
    if args.debug_audio and not args.record_turns:
        parser.error("--debug-audio requires --record-turns")
    return args


def main():
    args = parse_args()
    session_output = None

    if args.record_user_audio:
        print("warning: --record-user-audio is deprecated; use --record-turns", flush=True)

    if args.record_turns:
        try:
            decoder = OpusDecoder()
            decoder.close()
        except RuntimeError as error:
            print(f"Failed to initialize Opus decoder: {error}", file=sys.stderr)
            return 1
        session_output = SessionOutput(args.output_dir)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"BoardBridge server listening on {HOST}:{PORT}", flush=True)

        try:
            while True:
                client, address = server.accept()
                peer = f"{address[0]}:{address[1]}"
                print(f"[{timestamp()}] connected: {peer}", flush=True)
                turn_recorder = TurnRecorder(
                    args.record_turns, session_output, args.debug_audio
                )
                robot_audio_stats = turn_recorder if args.record_turns else RobotAudioStats()
                profiler = EventProfiler()
                completed_by = "disconnect"
                try:
                    with client:
                        receive_client(client, turn_recorder, robot_audio_stats, profiler)
                except ProtocolError as error:
                    print(f"[{timestamp()}] protocol error: {error}", flush=True)
                except KeyboardInterrupt:
                    completed_by = "shutdown"
                    raise
                except (ConnectionError, OSError):
                    pass
                finally:
                    turn_recorder.close(completed_by)
                    print(f"[{timestamp()}] disconnected: {peer}", flush=True)
        except KeyboardInterrupt:
            pass

    print("Server stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
