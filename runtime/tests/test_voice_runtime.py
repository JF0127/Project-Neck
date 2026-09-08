from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from runtime.audio_server import AudioWebSocketServer
from runtime.contracts import RobotSpeech, TurnSummary, UserAudio, UserSpeech
from runtime.dialogue import DialogueError
from runtime.runtime import Runtime, RuntimeState

FRAME = b"\x00\x00" * 320
USER_AUDIO = UserAudio(FRAME * 20, 16_000, 1, 0.4)
ROBOT_PCM = b"\x02\x00" * 320


class FakeVAD:
    def __init__(self) -> None:
        self.in_speech = False
        self.push_count = 0
        self._phase = 0

    def reset(self) -> None:
        self.in_speech = False
        self._phase = 0

    def push(self, frame: bytes):
        self.push_count += 1
        if self._phase == 0:
            self._phase = 1
            self.in_speech = True
            return None
        self._phase = 0
        self.in_speech = False
        return USER_AUDIO

    def end_stream(self):
        self.in_speech = False
        return None


class FakeASR:
    def __init__(self, text: str = "user text") -> None:
        self.text = text
        self.calls = []

    def transcribe(self, audio: UserAudio) -> UserSpeech:
        self.calls.append(audio)
        return UserSpeech(self.text, (), "en")


class FakeDialogue:
    def __init__(self, result="robot text") -> None:
        self.result = result
        self.requests = []

    def reply(self, request):
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeTTS:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts = []

    async def synthesize(self, text: str) -> RobotSpeech:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        return RobotSpeech(text, ROBOT_PCM, (), 0.02)


def make_runtime(
    *,
    asr: FakeASR | None = None,
    dialogue: FakeDialogue | None = None,
    tts: FakeTTS | None = None,
    cooldown_ms: int = 0,
) -> tuple[Runtime, FakeVAD, FakeASR, FakeDialogue, FakeTTS]:
    vad = FakeVAD()
    asr = asr or FakeASR()
    dialogue = dialogue or FakeDialogue()
    tts = tts or FakeTTS()
    runtime = Runtime(
        vad=vad,
        asr=asr,
        dialogue=dialogue,
        tts=tts,
        dialogue_fallback_text="fallback text",
        cooldown_ms=cooldown_ms,
    )
    return runtime, vad, asr, dialogue, tts


class RuntimeVoiceChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_chain_state_gating_history_and_next_turn(self) -> None:
        runtime, vad, asr, dialogue, tts = make_runtime(cooldown_ms=50)
        output_started = asyncio.Event()
        release_output = asyncio.Event()
        sent = []

        async def sender(speech: RobotSpeech) -> None:
            sent.append(speech)
            output_started.set()
            await release_output.wait()

        session = runtime.connection_opened(sender, "session")
        for number in range(10):
            session.add_turn(TurnSummary(str(number), f"u{number}", f"r{number}", "complete"))

        runtime.push_audio_frame(FRAME)
        self.assertIsNotNone(runtime.active_turn)
        self.assertEqual(runtime.active_turn.status, "listening")
        runtime.push_audio_frame(FRAME)
        self.assertEqual(runtime.state, RuntimeState.PROCESSING)

        pushes_after_segment = vad.push_count
        runtime.push_audio_frame(FRAME)
        self.assertEqual(vad.push_count, pushes_after_segment)

        await output_started.wait()
        self.assertEqual(runtime.state, RuntimeState.OUTPUTTING)
        runtime.push_audio_frame(FRAME)
        self.assertEqual(vad.push_count, pushes_after_segment)
        self.assertEqual(sent[0].pcm_s16le, ROBOT_PCM)

        release_output.set()
        while runtime.state != RuntimeState.COOLDOWN:
            await asyncio.sleep(0)
        runtime.push_audio_frame(FRAME)
        self.assertEqual(vad.push_count, pushes_after_segment)

        await runtime.wait_for_current_turn()
        self.assertEqual(runtime.state, RuntimeState.LISTENING)
        self.assertIsNone(runtime.active_turn)
        self.assertEqual(len(session.recent_turns), 10)
        self.assertEqual(session.recent_turns[-1].status, "complete")
        self.assertEqual(len(dialogue.requests[0].history), 10)
        self.assertEqual(asr.calls, [USER_AUDIO])
        self.assertEqual(tts.texts, ["robot text"])

        runtime.push_audio_frame(FRAME)
        self.assertIsNotNone(runtime.active_turn)
        await runtime.connection_closed()

    async def test_dialogue_failure_uses_runtime_fallback(self) -> None:
        error = DialogueError("network failed")
        runtime, _, _, dialogue, tts = make_runtime(dialogue=FakeDialogue(error))
        sent = []
        runtime.connection_opened(lambda speech: _append(sent, speech), "session")

        runtime.push_audio_frame(FRAME)
        runtime.push_audio_frame(FRAME)
        await runtime.wait_for_current_turn()

        self.assertEqual(tts.texts, ["fallback text"])
        self.assertEqual(sent[0].text, "fallback text")
        self.assertEqual(runtime.session.recent_turns[-1].status, "complete_fallback")
        self.assertIs(runtime.last_error, error)
        await runtime.connection_closed()

    async def test_empty_asr_skips_dialogue_and_audio(self) -> None:
        runtime, _, _, dialogue, tts = make_runtime(asr=FakeASR(""))
        sent = []
        runtime.connection_opened(lambda speech: _append(sent, speech), "session")

        runtime.push_audio_frame(FRAME)
        runtime.push_audio_frame(FRAME)
        await runtime.wait_for_current_turn()

        self.assertEqual(dialogue.requests, [])
        self.assertEqual(tts.texts, [])
        self.assertEqual(sent, [])
        self.assertEqual(runtime.session.recent_turns[-1].status, "empty_speech")
        await runtime.connection_closed()

    async def test_tts_failure_finishes_failed_turn_without_audio(self) -> None:
        error = RuntimeError("tts failed")
        runtime, _, _, _, _ = make_runtime(tts=FakeTTS(error))
        sent = []
        runtime.connection_opened(lambda speech: _append(sent, speech), "session")

        runtime.push_audio_frame(FRAME)
        runtime.push_audio_frame(FRAME)
        await runtime.wait_for_current_turn()

        self.assertEqual(sent, [])
        self.assertEqual(runtime.session.recent_turns[-1].status, "failed")
        self.assertIs(runtime.last_error, error)
        await runtime.connection_closed()


async def _append(values: list, value) -> None:
    values.append(value)


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send(self, message) -> None:
        self.messages.append(message)


class AudioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_robot_pcm_uses_frozen_frames_and_control_messages(self) -> None:
        runtime, _, _, _, _ = make_runtime()
        server = AudioWebSocketServer(runtime)
        websocket = FakeWebSocket()
        pcm = bytes(642)
        speech = RobotSpeech("robot", pcm, (), len(pcm) / 2 / 16_000)

        with patch("runtime.audio_server.asyncio.sleep", new=AsyncMock()):
            await server._send_robot_audio(websocket, speech)

        start = json.loads(websocket.messages[0])
        end = json.loads(websocket.messages[-1])
        frames = websocket.messages[1:-1]
        self.assertEqual(start["type"], "stream_start")
        self.assertEqual(start["source"], "robot")
        self.assertEqual(start["sample_rate"], 16_000)
        self.assertEqual(start["channels"], 1)
        self.assertEqual(start["format"], "pcm_s16le")
        self.assertEqual(end, {"type": "stream_end", "stream_id": start["stream_id"]})
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(len(frame) == 640 for frame in frames))
        self.assertEqual(frames[0], pcm[:640])
        self.assertEqual(frames[1][:2], pcm[640:])
        self.assertEqual(frames[1][2:], bytes(638))


if __name__ == "__main__":
    unittest.main()
