"""Software-only Ubuntu V1 turn-flow checks; no audio or Motor socket."""
import asyncio
import json
import os
import socket
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

from .contracts import RobotState
from .doubao_streaming_v1 import parse_frame
from .inference.text_motion_plan import TextMotionPlan
from .neck_client import NeckClient
from .ubuntu_v1_motion import UbuntuV1Mixer
from .ubuntu_v1_runtime import UbuntuV1Runtime


def _plan(text: str) -> TextMotionPlan:
    return TextMotionPlan.from_dict({'reply_text': text, 'actions': []}, text)


class UbuntuV1Test(unittest.TestCase):
    def test_turn_document_shape_and_base_pose(self):
        mixer = UbuntuV1Mixer()
        base = (0.01, -0.02, 0.03)
        document = mixer.turn_document(_plan('天气晴朗。'), 1.5, base, 'turn_0001')
        NeckClient.validate(document)
        self.assertEqual(document["fps"], 30.0)
        self.assertEqual(len(document["trajectory"]), 45)
        self.assertEqual(len(document["states"]), 45)
        self.assertEqual(document["trajectory"][0], list(base))

    def test_reply_flow_motion_ready_then_tts_then_single_trigger(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)
        runtime.mixer = UbuntuV1Mixer()
        runtime.planner = type('P', (), {
            'plan_text': lambda self, text: ('{}', _plan(text))
        })()
        runtime.state = RobotState()
        runtime.reply_number = 0
        runtime.playback_active = False
        yielded = []
        motion_starts = []

        class FakeStdin:
            def __init__(self):
                self.writes = []

            def write(self, chunk):
                self.writes.append(chunk)

            async def drain(self):
                await asyncio.sleep(0)

            def close(self):
                pass

            async def wait_closed(self):
                pass

        class FakeProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.returncode = None

            async def wait(self):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

        async def chunks(text):
            yielded.append(1)
            yield b'\0' * 640
            await asyncio.sleep(.001)
            yielded.append(2)
            yield b'\1' * 640

        real_sleep = asyncio.sleep
        messages = []

        async def fake_motion(self, document):
            motion_starts.append(len(yielded))
            await real_sleep(0.02)
            messages.append("motion_end")

        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def start_player(*args, **kwargs):
            return FakeProcess()

        def capture(message):
            messages.append(message)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.pcm_chunks', chunks), patch(
                'runtime.ubuntu_v1_runtime.asyncio.create_subprocess_exec', start_player
            ), patch('runtime.ubuntu_v1_runtime.asyncio.to_thread', immediate), patch(
                'runtime.ubuntu_v1_runtime.UbuntuV1Runtime._execute_motion', fake_motion
            ), patch('runtime.ubuntu_v1_runtime.log', capture):
                await runtime._output('你好。')

        asyncio.run(run())
        self.assertEqual(motion_starts, [1])
        expected = ["deepseek_reply", "motion_generation_start", "motion_plan",
                    "trajectory_ready", "tts_request", "tts_first_audio",
                    "playback_and_motion_start", "tts_playback_end", "motion_end",
                    "turn_complete"]
        positions = {}
        for name in expected:
            index = next(i for i, message in enumerate(messages) if message.startswith(name))
            positions[name] = index
        for before, after in zip(expected, expected[1:]):
            self.assertLess(positions[before], positions[after],
                            f"{before} should precede {after}: {messages}")

    def test_motion_generation_failure_skips_tts(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)

        class FailingPlanner:
            def plan_text(self, text):
                raise RuntimeError("planner down")

        runtime.planner = FailingPlanner()
        runtime.mixer = UbuntuV1Mixer()
        runtime.state = RobotState()
        runtime.reply_number = 0
        messages = []

        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.asyncio.to_thread', immediate), patch(
                'runtime.ubuntu_v1_runtime.log', lambda m: messages.append(m)
            ):
                await runtime._output('你好。')

        asyncio.run(run())
        self.assertFalse(any(m.startswith('tts_request') for m in messages))
        self.assertTrue(any(m.startswith('turn_complete status=motion_failed') for m in messages))

    def test_tts_failure_skips_motion(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)
        runtime.mixer = UbuntuV1Mixer()
        runtime.planner = type('P', (), {
            'plan_text': lambda self, text: ('{}', _plan(text))
        })()
        runtime.state = RobotState()
        runtime.reply_number = 0
        calls = []
        messages = []

        async def failing_chunks(text):
            raise RuntimeError("tts down")
            yield b""

        async def fake_motion(self, document):
            calls.append(document)

        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.pcm_chunks', failing_chunks), patch(
                'runtime.ubuntu_v1_runtime.asyncio.to_thread', immediate
            ), patch('runtime.ubuntu_v1_runtime.UbuntuV1Runtime._execute_motion', fake_motion), patch(
                'runtime.ubuntu_v1_runtime.log', lambda m: messages.append(m)
            ):
                await runtime._output('你好。')

        asyncio.run(run())
        self.assertEqual(calls, [])
        self.assertFalse(any(m.startswith('playback_and_motion_start') for m in messages))
        self.assertTrue(any(m.startswith('turn_complete status=tts_failed') for m in messages))

    def test_execute_motion_sends_trajectory_once(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)
        sent = []
        state = RobotState(head_rpy=[0.0, 0.0, 0.0],
                           head_rpy_valid=True, motor_available=True)

        class FakeNeck:
            def send(self, document):
                sent.append(document)
                state.motion_executing = True

        class FakeFeedback:
            connected = True
            stale_sec = 0.2

            def message_age_sec(self):
                return 0.005

            @property
            def observed_rate_hz(self):
                return 30.0

        runtime.neck = FakeNeck()
        runtime.feedback = FakeFeedback()
        runtime.state = state
        document = {"name": "t", "fps": 30.0, "unit": "radian",
                    "order": ["roll", "pitch", "yaw"],
                    "trajectory": [[0.0, 0.0, 0.0]] * 45,
                    "states": ["silent"] * 45}
        real_sleep = asyncio.sleep
        messages = []

        async def fake_sleep(_):
            if state.motion_executing:
                state.motion_executing = False
            await real_sleep(0)

        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.asyncio.sleep', fake_sleep), patch(
                'runtime.ubuntu_v1_runtime.asyncio.to_thread', immediate
            ), patch('runtime.ubuntu_v1_runtime.log', lambda m: messages.append(m)):
                await runtime._execute_motion(document)

        asyncio.run(run())
        self.assertEqual(sent, [document])
        self.assertTrue(any(m.startswith('motion_end') for m in messages))

    def test_startup_waits_for_first_valid_feedback(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)
        state = RobotState()
        runtime.state = state

        class FakeFeedback:
            connected = False
            stale_sec = 0.2

            def message_age_sec(self):
                return None

            @property
            def observed_rate_hz(self):
                return None

        runtime.feedback = FakeFeedback()
        real_sleep = asyncio.sleep
        messages = []
        calls = {"n": 0}

        async def fake_sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:
                runtime.feedback.connected = True
                state.head_rpy_valid = True
                state.motor_available = True
            await real_sleep(0)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.asyncio.sleep', fake_sleep), patch(
                'runtime.ubuntu_v1_runtime.log', lambda m: messages.append(m)
            ):
                await runtime._wait_for_motor_feedback()

        asyncio.run(run())
        self.assertTrue(any(m.startswith('[motion-gate] waiting_for_valid_feedback')
                            for m in messages))
        self.assertTrue(any(m.startswith('[ubuntu-v1] Motor feedback ready before listening')
                            for m in messages))

    def test_motion_gate_logs_specific_reason(self):
        runtime = UbuntuV1Runtime.__new__(UbuntuV1Runtime)
        state = RobotState(head_rpy_valid=False, motor_available=True)
        sent = []

        class FakeNeck:
            def send(self, document):
                sent.append(document)

        class FakeFeedback:
            connected = True
            stale_sec = 0.2

            def message_age_sec(self):
                return 0.012

            @property
            def observed_rate_hz(self):
                return 29.9

        runtime.neck = FakeNeck()
        runtime.feedback = FakeFeedback()
        runtime.state = state
        messages = []
        document = {"name": "t", "fps": 30.0, "unit": "radian",
                    "order": ["roll", "pitch", "yaw"],
                    "trajectory": [[0.0, 0.0, 0.0]] * 45,
                    "states": ["silent"] * 45}

        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def run():
            with patch('runtime.ubuntu_v1_runtime.asyncio.to_thread', immediate), patch(
                'runtime.ubuntu_v1_runtime.log', lambda m: messages.append(m)
            ):
                await runtime._execute_motion(document)

        asyncio.run(run())
        self.assertEqual(sent, [])
        gate = next(m for m in messages if m.startswith('[motion-gate]\nreason='))
        self.assertIn('reason=head_rpy_invalid', gate)
        self.assertIn('connected=True', gate)
        self.assertIn('motor_available=True', gate)
        self.assertIn('head_rpy_valid=False', gate)
        self.assertIn('feedback_age_ms=12.0', gate)
        self.assertIn('feedback_rate_hz=29.9', gate)

    def test_neck_client_sends_neck_pose_set_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.sock")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path)
            server.listen(1)
            received = []

            def serve():
                client, _ = server.accept()
                data = b""
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                client.close()
                received.append(json.loads(data))

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            NeckClient(socket_path=path).send_pose(0.5, -0.2, 2.0)
            thread.join(timeout=5)
            server.close()

        self.assertEqual(received, [{
            "type": "neck_pose_set", "slave_id": 0,
            "roll_deg": 0.5, "pitch_deg": -0.2, "yaw_deg": 2.0,
        }])

    def test_websocket_audio_event(self):
        sid = b'test'
        frame = (b'\x11\xb4\x00\x00' + struct.pack('>I', 352) +
                 struct.pack('>I', len(sid)) + sid + struct.pack('>I', 4) + b'\x00\x00\x01\x00')
        self.assertEqual(parse_frame(frame), (352, b'\x00\x00\x01\x00'))
        with self.assertRaises(ValueError):
            parse_frame(frame[:-1])


if __name__ == '__main__':
    unittest.main()
