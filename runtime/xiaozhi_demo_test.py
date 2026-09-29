"""Software-only XiaoZhi demo checks (no board, speaker or Motor)."""
import asyncio
import ctypes
import ctypes.util
import struct
import unittest
from unittest.mock import patch

from . import xiaozhi_demo_runtime as demo
from .xiaozhi_adapter import BoardBridgeParser, XiaoZhiAdapter


class FakeDecoder:
    def __init__(self, rate, channels):
        self.closed = False

    def decode(self, packet):
        return b"pcm:" + packet

    def close(self):
        self.closed = True


class StreamingSpeakerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.played = []
        self.rates = []
        self.patch = patch.object(demo, "TRAILING_AUDIO_SEC", 0.03)
        self.patch.start()

        async def player(queue, rate):
            self.rates.append(rate)
            while (pcm := await queue.get()) is not None:
                self.played.append(pcm)

        self.speaker = demo.StreamingSpeaker(FakeDecoder, player)

    async def asyncTearDown(self):
        self.speaker.disconnect()
        await asyncio.sleep(0)
        self.patch.stop()

    async def test_streaming_and_trailing_audio_after_end(self):
        self.speaker.audio(1, 1, b"a", 24000, 60, 1)
        await asyncio.sleep(0)
        self.assertEqual(self.played, [b"pcm:a"])  # before playback_end
        self.speaker.end(1, None)
        self.speaker.audio(1, 2, b"b", 24000, 60, 1)
        await asyncio.sleep(0.06)
        self.assertEqual(self.played, [b"pcm:a", b"pcm:b"])
        self.assertEqual(self.rates, [24000])
        self.speaker.audio(1, 3, b"late", 24000, 60, 1)
        self.assertEqual(len(self.played), 2)

    async def test_end_overtakes_first_audio(self):
        self.speaker.end(2, None)
        self.speaker.audio(2, 1, b"audio", 16000, 60, 1)
        await asyncio.sleep(0.06)
        self.assertEqual(self.played, [b"pcm:audio"])
        self.assertIsNone(self.speaker.turn)

    async def test_abort_and_disconnect_drop_pending(self):
        self.speaker.audio(3, 1, b"first", 24000, 60, 1)
        await asyncio.sleep(0)
        self.speaker.abort(3, "wake_word_detected", None)
        self.speaker.audio(3, 2, b"ignored", 24000, 60, 1)
        await asyncio.sleep(0)
        self.assertEqual(self.played, [b"pcm:first"])
        self.speaker.disconnect()
        self.speaker.audio(3, 1, b"new session", 24000, 60, 1)
        await asyncio.sleep(0)
        self.assertEqual(self.played[-1], b"pcm:new session")

    async def test_fragmented_bridge_frames_reach_speaker(self):
        adapter = XiaoZhiAdapter(lambda _state: None, on_robot_audio=self.speaker.audio,
                                 on_robot_playback_end=self.speaker.end)
        wire = (struct.pack("!BIIIIHB", 3, 1, 1, 5, 24000, 60, 1) + b"a"
                + b'{"type":"robot_playback_end","turn_id":5}\n')
        parser = BoardBridgeParser()
        for byte in wire:
            for event in parser.feed(bytes([byte])):
                adapter.handle(event)
        await asyncio.sleep(0.06)
        self.assertEqual(self.played, [b"pcm:a"])

    async def test_format_change_stops_turn(self):
        self.speaker.audio(4, 1, b"a", 24000, 60, 1)
        self.speaker.audio(4, 2, b"b", 16000, 60, 1)
        self.assertIn(4, self.speaker.closed)
        self.assertIsNone(self.speaker.turn)


class OpusDecoderTest(unittest.TestCase):
    @unittest.skipUnless(ctypes.util.find_library("opus"), "libopus unavailable")
    def test_real_opus_packet_decodes_to_mono_pcm(self):
        lib = ctypes.CDLL(ctypes.util.find_library("opus"))
        lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int)]
        lib.opus_encoder_create.restype = ctypes.c_void_p
        lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16),
                                    ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
        lib.opus_encode.restype = ctypes.c_int
        lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        error = ctypes.c_int()
        handle = lib.opus_encoder_create(24000, 1, 2049, ctypes.byref(error))
        self.assertTrue(handle)
        self.assertEqual(error.value, 0)
        try:
            pcm = (ctypes.c_int16 * 1440)()
            output = (ctypes.c_ubyte * 8192)()
            length = lib.opus_encode(handle, pcm, 1440, output, len(output))
            self.assertGreater(length, 0)
            decoder = demo.OpusStreamDecoder(24000, 1)
            try:
                decoded = decoder.decode(bytes(output[:length]))
                self.assertEqual(len(decoded), 1440 * 2)
            finally:
                decoder.close()
        finally:
            lib.opus_encoder_destroy(handle)


class NaturalLoopTest(unittest.TestCase):
    def test_static_three_no_motor_clients(self):
        loop = demo.NaturalTrajectoryLoop()
        self.assertEqual([doc["name"] for doc in loop.documents], list(demo.TRAJECTORY_NAMES))
        self.assertEqual([len(doc["trajectory"]) for doc in loop.documents], [241, 271, 301])
        self.assertIsNone(loop.neck)
        self.assertIsNone(loop.feedback)

    def test_demo_constructs_without_cloud_or_motor(self):
        runtime = demo.XiaoZhiDemoRuntime({"xiaozhi": {"host": "127.0.0.1", "port": 0}})
        self.assertIsNone(runtime.trajectories.neck)
        self.assertIsNone(runtime.trajectories.feedback)
