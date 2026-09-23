"""Wi-Fi BoardBridge server and XiaoZhi behavior events."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

_PROTOCOL = Path(__file__).parent / "audio/board/xiaozhi-esp32/tools/board_bridge_protocol.py"
_SPEC = importlib.util.spec_from_file_location("project_neck_board_bridge_protocol", _PROTOCOL)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
BoardBridgeParser = _MODULE.BoardBridgeParser
ProtocolError = _MODULE.ProtocolError


def is_spoken_robot_text(text: str) -> bool:
    return bool(text.strip()) and not text.lstrip().startswith("% ")


class XiaoZhiAdapter:
    def __init__(self, on_state, on_user_text=None, on_robot_text=None,
                 on_robot_first_audio=None, on_robot_playback_start=None,
                 on_robot_playback_end=None, on_robot_playback_abort=None):
        self.on_state = on_state
        self.on_user_text = on_user_text
        self.on_robot_text = on_robot_text
        self.on_robot_first_audio = on_robot_first_audio
        self.on_robot_playback_start = on_robot_playback_start
        self.on_robot_playback_end = on_robot_playback_end
        self.on_robot_playback_abort = on_robot_playback_abort
        self.state = "silent"
        self.active_turn = None
        self.robot_audio_frames = 0
        self._connected = False

    def _state(self, value):
        if value != self.state:
            self.state = value
            self.on_state(value)

    def handle(self, event):
        kind = event[0]
        if kind == "user_audio":
            self._state("listening")
            return
        if kind == "robot_audio":
            self.robot_audio_frames += 1
            return
        if kind != "message" or not isinstance(event[1], dict):
            return
        message = event[1]
        kind = message.get("type")
        turn = message.get("turn_id")
        timestamp = message.get("timestamp_ms")
        if kind == "user_text":
            value = str(message.get("text", ""))
            print(f"[XIAOZHI] user_text: {value}", flush=True)
            self._state("listening")
            if self.on_user_text:
                self.on_user_text(value)
        elif kind == "robot_text":
            value = str(message.get("text", ""))
            if not is_spoken_robot_text(value):
                print(f"[XIAOZHI] robot_text ignored non-spoken: {value}", flush=True)
                return
            print(f"[XIAOZHI] robot_text turn={turn} text={json.dumps(value, ensure_ascii=False)}", flush=True)
            self.active_turn = turn
            self._state("thinking")
            if self.on_robot_text:
                self.on_robot_text(turn, value, timestamp)
        elif kind == "robot_first_audio":
            self.active_turn = turn
            self._state("thinking")
            if self.on_robot_first_audio:
                self.on_robot_first_audio(turn, timestamp)
        elif kind == "robot_playback_start":
            print(f"[XIAOZHI] playback_start turn={turn}", flush=True)
            self.active_turn = turn
            self._state("speaking")
            if self.on_robot_playback_start:
                self.on_robot_playback_start(turn, timestamp)
        elif kind in ("robot_playback_end", "robot_playback_abort"):
            if turn is not None and self.active_turn is not None and turn != self.active_turn:
                return
            if kind == "robot_playback_end":
                print(f"[XIAOZHI] playback_end turn={turn}", flush=True)
                if self.on_robot_playback_end:
                    self.on_robot_playback_end(turn, timestamp)
            else:
                reason = str(message.get("reason", ""))
                print(f"[XIAOZHI] playback_abort turn={turn} reason={reason}", flush=True)
                if self.on_robot_playback_abort:
                    self.on_robot_playback_abort(turn, reason, timestamp)
            self.active_turn = None
            self._state("silent")

    async def serve(self, host="0.0.0.0", port=8766):
        async def client(reader, writer):
            if self._connected:
                print("[XIAOZHI] warning: second board connection rejected", flush=True)
                writer.close()
                await writer.wait_closed()
                return
            self._connected = True
            peer = writer.get_extra_info("peername")
            print(f"[XIAOZHI] connected: {peer}", flush=True)
            parser = BoardBridgeParser()
            try:
                while data := await reader.read(4096):
                    for event in parser.feed(data):
                        self.handle(event)
            except (ProtocolError, ConnectionError) as exc:
                print(f"[XIAOZHI] warning: client error: {exc}", flush=True)
            finally:
                self._connected = False
                self.active_turn = None
                self._state("silent")
                writer.close()
                await writer.wait_closed()
                print("[XIAOZHI] warning: disconnected; waiting for board", flush=True)

        server = await asyncio.start_server(client, host, port)
        print(f"[XIAOZHI] listening on {host}:{port}", flush=True)
        async with server:
            await server.serve_forever()
