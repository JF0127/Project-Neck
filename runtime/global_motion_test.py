"""Software checks for XiaoZhi Global Motion mode."""
import json
import math
import asyncio
import socket
import struct
import unittest
from types import SimpleNamespace

from .__main__ import DEFAULT_CONFIG, load_config
from .global_motion import GlobalMotionGenerator
from .xiaozhi_adapter import BoardBridgeParser, XiaoZhiAdapter, is_spoken_robot_text
from .xiaozhi_motion_runtime import XiaoZhiMotionRuntime


class GlobalMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)[0]["global_motion"]

    def test_sixty_seconds_continuous_and_bounded(self):
        cycles = []
        gen = GlobalMotionGenerator(self.config, seed=7, on_cycle=lambda *args: cycles.append(args))
        frames = []
        indices = []
        for i in range(1800):
            frames.append(tuple(math.degrees(x) for x in gen.sample(i / 30)))
            indices.append(gen.cycle_index)
        self.assertGreaterEqual(gen.cycle_index, 6)
        self.assertEqual(len(cycles), gen.cycle_index + 1)
        self.assertTrue(all(math.isfinite(x) for frame in frames for x in frame))
        for axis, name, visible in ((0, "roll", 0.8), (1, "pitch", 1.8), (2, "yaw", 3.5)):
            values = [row[axis] for row in frames]
            self.assertGreater(max(values), visible)
            self.assertLess(min(values), -visible)
            self.assertLessEqual(max(abs(value) for value in values),
                                 self.config[f"{name}_amplitude_deg"] * (1 + self.config["parameter_variation"]) + 1e-4)
        self.assertLess(max(abs(frames[i + 1][j] - frames[i][j]) * 30
                            for i in range(len(frames) - 1) for j in range(3)), 5.0)
        self.assertGreater(len(set(round(d, 3) for d in gen.cycle_parameter_durations)), 2)
        self.assertTrue(all(7.52 <= d <= 8.48 for d in gen.cycle_parameter_durations))
        self.assertTrue(all(duration > 7.0 for duration in gen.completed_cycle_durations))
        for index in range(1, len(frames)):
            if indices[index] != indices[index - 1]:
                self.assertLess(max(abs(frames[index][axis] - frames[index - 1][axis])
                                    for axis in range(3)), 0.2)

    def test_shared_phase_and_seed_are_reproducible(self):
        first = GlobalMotionGenerator(self.config, seed=13)
        second = GlobalMotionGenerator(self.config, seed=13)
        values_a = [first.sample(i / 30) for i in range(900)]
        values_b = [second.sample(i / 30) for i in range(900)]
        self.assertEqual(values_a, values_b)
        self.assertEqual(first.cycle_index, second.cycle_index)
        self.assertGreater(first.cycle_index, 2)
        # Harmonic distortion makes the waveform different from a pure sine.
        self.assertGreater(abs(values_a[0][2]), math.radians(0.1))

    def test_state_transitions_do_not_reset(self):
        gen = GlobalMotionGenerator(self.config, seed=19)
        frames = []
        for state, seconds in (("silent", 5), ("listening", 5), ("thinking", 5),
                               ("speaking", 10), ("silent", 5)):
            gen.set_state(state)
            for _ in range(seconds * 30):
                frames.append(gen.sample(len(frames) / 30))
        for index in (150, 300, 450, 750):
            self.assertLess(max(abs(frames[index][j] - frames[index - 1][j])
                                for j in range(3)), math.radians(0.2))
        self.assertNotEqual(frames[750], (0.0, 0.0, 0.0))

    def test_state_switch_preserves_phase_and_pose_at_transition(self):
        gen = GlobalMotionGenerator(self.config, seed=3)
        before = None
        for i in range(151):
            before = gen.sample(i / 30)
        phase = gen.phase
        gen.set_state("speaking")
        after = gen.sample(5.0)
        self.assertEqual(gen.phase, phase)
        self.assertEqual(after, before)
        self.assertAlmostEqual(gen._style_at(5.0)[0], self.config["states"]["silent"]["amplitude"])
        gen.sample(6.5)
        self.assertAlmostEqual(gen._style_at(6.5)[0], self.config["states"]["speaking"]["amplitude"])

    def test_chunk_uses_existing_motor_document_without_neutral_tail(self):
        config = load_config(DEFAULT_CONFIG)[0]
        runtime = XiaoZhiMotionRuntime(config, dry_run=True, seed=23)
        frames = [runtime._next_frame() for _ in range(30)]
        document = runtime._document(frames, ["thinking"] * 30, 1)
        self.assertEqual(document["fps"], 30.0)
        self.assertEqual(document["unit"], "radian")
        self.assertEqual(document["order"], ["roll", "pitch", "yaw"])
        self.assertEqual(document["states"], ["silent"] * 30)
        self.assertEqual(tuple(document["trajectory"][-1]), frames[-1])
        self.assertNotEqual(frames[-1], (0.0, 0.0, 0.0))

    def test_parser_and_filter(self):
        self.assertFalse(is_spoken_robot_text("% get_weather..."))
        self.assertTrue(is_spoken_robot_text("武汉现在23度，有雾。"))
        parser = BoardBridgeParser()
        messages = [
            {"type": "user_text", "text": "天气"},
            {"type": "robot_text", "turn_id": 1, "timestamp_ms": 10, "text": "% get_weather..."},
            {"type": "robot_text", "turn_id": 1, "timestamp_ms": 20, "text": "武汉现在23度，有雾。"},
            {"type": "robot_first_audio", "turn_id": 1, "timestamp_ms": 30},
            {"type": "robot_playback_start", "turn_id": 1, "timestamp_ms": 40},
            {"type": "robot_audio_end", "turn_id": 1, "timestamp_ms": 50},
            {"type": "robot_playback_end", "turn_id": 1, "timestamp_ms": 60},
            {"type": "robot_playback_abort", "turn_id": 2, "timestamp_ms": 70, "reason": "test"},
        ]
        audio = b"\x11\x22\x33"
        stream = b"\x01" + struct.pack("!II", 1, 2) + b"\x11\x22"
        stream += b"".join(json.dumps(m).encode() + b"\n" for m in messages[:5])
        stream += b"\x03" + struct.pack("!IIIIHB", 1, len(audio), 1, 16000, 60, 1) + audio
        stream += b"".join(json.dumps(m).encode() + b"\n" for m in messages[5:])
        events = []
        for i in range(0, len(stream), 7):
            events.extend(parser.feed(stream[i:i + 7]))
        self.assertEqual(len(events), 10)
        self.assertEqual(events[0][0], "user_audio")
        self.assertEqual(events[6][0], "robot_audio")
        states, spoken = [], []
        adapter = XiaoZhiAdapter(states.append, on_robot_text=lambda _id, text, _ts: spoken.append(text))
        for event in events:
            adapter.handle(event)
        self.assertEqual(spoken, ["武汉现在23度，有雾。"])
        self.assertEqual(adapter.robot_audio_frames, 1)
        self.assertEqual(states, ["listening", "thinking", "speaking", "silent"])

    def test_tcp_disconnect_keeps_adapter_alive(self):
        async def check():
            states = []
            adapter = XiaoZhiAdapter(states.append)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            task = asyncio.create_task(adapter.serve("127.0.0.1", port))
            try:
                for _ in range(30):
                    try:
                        reader, writer = await asyncio.open_connection("127.0.0.1", port)
                        break
                    except ConnectionRefusedError:
                        await asyncio.sleep(0.01)
                else:
                    self.fail("adapter failed to listen")
                writer.write(b'{"type":"user_text","text":"hello"}\n')
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                await asyncio.sleep(0.05)
                self.assertEqual(states, ["listening", "silent"])
                self.assertFalse(task.done())
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        asyncio.run(check())

    def test_motor_chunks_continue_after_pose_feedback_becomes_invalid(self):
        async def check():
            config = load_config(DEFAULT_CONFIG)[0]
            runtime = XiaoZhiMotionRuntime(config, dry_run=True, seed=5)
            for _ in range(90):
                runtime._next_frame()
            runtime.motor_state.motor_available = True
            runtime.motor_state.head_rpy_valid = True
            runtime.feedback = SimpleNamespace(connected=True)
            sent = []

            def send(document):
                sent.append(document)
                runtime.motor_state.motion_executing = True

            runtime.neck = SimpleNamespace(send=send)

            async def complete():
                while len(sent) < 2:
                    if runtime.motor_state.motion_executing:
                        await asyncio.sleep(0.06)
                        runtime.motor_state.motion_executing = False
                        runtime.motor_state.motor_available = False
                        runtime.motor_state.head_rpy_valid = False
                    await asyncio.sleep(0.01)

            motor = asyncio.create_task(runtime._motor_loop())
            finisher = asyncio.create_task(complete())
            try:
                await asyncio.wait_for(finisher, 2)
                self.assertGreaterEqual(len(sent), 2)
                self.assertEqual(sent[1]["trajectory"][0], sent[0]["trajectory"][-1])
            finally:
                motor.cancel()
                await asyncio.gather(motor, return_exceptions=True)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
