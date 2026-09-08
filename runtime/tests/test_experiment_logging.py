from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from runtime.dialogue import FixedDialogue
from runtime.experiment_logger import ExperimentLogger
from runtime.motion import MotionTurn
from runtime.runtime import AlgorithmRuntime, RuntimeState


class FakeWord:
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end

    def as_dict(self) -> dict:
        return {"text": self.text, "start_time": self.start, "end_time": self.end}


class FakeASR:
    def transcribe_pcm(self, pcm: bytes):
        return type("Transcription", (), {
            "text": "hello",
            "words": [FakeWord("hello", 0.0, 0.2)],
        })()

    def transcribe_waveform(self, waveform):
        raise AssertionError("TTS words are available; fallback ASR should not run")


class FakeTTS:
    async def synthesize(self, text: str):
        return type("TTSResult", (), {
            "pcm_s16le": bytes(640),
            "waveform": np.zeros(320, dtype=np.float32),
            "words": [FakeWord("reply", 0.0, 0.02)],
            "duration_sec": 0.02,
        })()


class FakeMotion:
    def generate_turn(
        self, robot_pcm, robot_words, robot_duration, turn_number, on_generation,
    ):
        speaker = np.asarray(
            [[0.0, 0.0, 0.0], [-0.03, 0.02, 0.01]], dtype=np.float32
        )
        on_generation("speaker", speaker, {
            "model": "baseline_v1",
            "checkpoint": "/test/checkpoints/best.pt",
            "fps": 30.0,
            "query_timestamps_sec": [0.0, 1.0 / 30.0],
            "representation": "rpy_offset",
        })
        document = {
            "name": f"algorithm_turn_{turn_number}",
            "fps": 30.0,
            "unit": "radian",
            "order": ["roll", "pitch", "yaw"],
            "trajectory": [
                [0.0, 0.0, 0.0],
                [-0.03, 0.02, 0.01],
                [0.0, 0.0, 0.0],
            ],
            "states": ["speaking", "speaking", "silent"],
        }
        return MotionTurn(document, speaking_start_frame=0, speaker_frames=2)


class FakeNeck:
    mock = True

    def __init__(self):
        self.documents: list[dict] = []
        self.measurement_targets: list[tuple[Path | None, float | None]] = []

    def send(
        self,
        document: dict,
        measured_output_path: Path | None = None,
        turn_origin_unix_sec: float | None = None,
    ) -> None:
        self.documents.append(document)
        self.measurement_targets.append((measured_output_path, turn_origin_unix_sec))


def make_runtime(logger: ExperimentLogger) -> AlgorithmRuntime:
    runtime = AlgorithmRuntime.__new__(AlgorithmRuntime)
    runtime.state = RuntimeState.IDLE
    runtime.turn_count = 0
    runtime._turn_lock = asyncio.Lock()
    runtime.experiment_logger = logger
    runtime._active_experiment_turn_id = None
    runtime.asr = FakeASR()
    runtime.dialogue = FixedDialogue("robot reply")
    runtime.tts = FakeTTS()
    runtime.motion = FakeMotion()
    runtime.neck = FakeNeck()
    return runtime


async def run_fake_session(root: Path, turns: int = 2) -> tuple[ExperimentLogger, AlgorithmRuntime]:
    logger = ExperimentLogger(
        checkpoint="/test/checkpoints/best.pt",
        runtime_mode="mock-neck",
        trajectory_fps=30.0,
        root=root,
    )
    runtime = make_runtime(logger)
    for number in range(1, turns + 1):
        runtime.begin_user_stream()
        result = await runtime.process_user_stream(bytes(640), f"user_stream_{number}")
        robot_stream_id = f"robot_stream_{number}"
        runtime.record_robot_audio_start(result.experiment_turn_id, robot_stream_id)
        runtime.record_robot_audio_end(result.experiment_turn_id, robot_stream_id)
        runtime.finish_speaking(result.experiment_turn_id)
    logger.end_session()
    return logger, runtime


class ExperimentLoggingTest(unittest.TestCase):
    def test_session_ids_do_not_overwrite_same_day(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = ExperimentLogger("first.pt", "mock-neck", 30.0, root)
            second = ExperimentLogger("second.pt", "mock-neck", 30.0, root)
            try:
                self.assertNotEqual(first.session_id, second.session_id)
                self.assertTrue(first.session_dir.is_dir())
                self.assertTrue(second.session_dir.is_dir())
            finally:
                first.end_session()
                second.end_session()

    def test_speaker_only_records_match_sent_payload_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger, runtime = asyncio.run(run_fake_session(Path(temporary), turns=2))
            turns = sorted((logger.session_dir / "turns").iterdir())
            self.assertEqual([path.name for path in turns], ["turn_001", "turn_002"])
            self.assertEqual(len(runtime.neck.documents), 2)

            for index, turn_dir in enumerate(turns):
                for name in (
                    "dialogue.json", "timeline.json", "robot_audio.wav",
                    "robot_words.json", "robot_motion_input.json",
                    "neck_rpy.json", "neck_rpy.csv",
                ):
                    self.assertTrue((turn_dir / name).is_file(), name)

                generations = sorted((turn_dir / "model_generations").iterdir())
                self.assertEqual(
                    [path.name for path in generations],
                    ["generation_001_speaker.json"],
                )
                generation = json.loads(generations[0].read_text(encoding="utf-8"))
                self.assertEqual(generation["role"], "speaker")
                self.assertEqual(generation["model"], "baseline_v1")
                self.assertEqual(generation["representation"], "rpy_offset")
                self.assertEqual(generation["checkpoint"], "/test/checkpoints/best.pt")
                self.assertEqual(generation["query_timestamps_sec"], [0.0, 1.0 / 30.0])

                with wave.open(str(turn_dir / "robot_audio.wav"), "rb") as audio:
                    self.assertEqual(audio.getframerate(), 16_000)
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getsampwidth(), 2)
                    self.assertEqual(audio.readframes(audio.getnframes()), bytes(640))
                self.assertEqual(
                    json.loads((turn_dir / "robot_words.json").read_text(encoding="utf-8")),
                    [{"text": "reply", "start_time": 0.0, "end_time": 0.02}],
                )
                dialogue_record = json.loads(
                    (turn_dir / "dialogue.json").read_text(encoding="utf-8")
                )
                self.assertEqual(dialogue_record["dialogue_backend"], "fixed")
                self.assertNotIn("dialogue_model", dialogue_record)

                motion_input = json.loads(
                    (turn_dir / "robot_motion_input.json").read_text(encoding="utf-8")
                )
                self.assertEqual(motion_input["duration_sec"], 0.02)
                self.assertEqual(motion_input["sample_rate"], 16_000)
                self.assertEqual(motion_input["format"], "pcm_s16le")
                self.assertEqual(motion_input["num_samples"], 320)

                neck_record = json.loads((turn_dir / "neck_rpy.json").read_text(encoding="utf-8"))
                sent = runtime.neck.documents[index]
                measured_path, turn_origin = runtime.neck.measurement_targets[index]
                self.assertEqual(measured_path, turn_dir / "measured_rpy.json")
                self.assertIsInstance(turn_origin, float)
                self.assertEqual(neck_record["trajectory"], sent["trajectory"])
                self.assertEqual(set(neck_record["states"]), {"speaking", "silent"})
                self.assertNotIn("listening", neck_record["states"])

                with (turn_dir / "neck_rpy.csv").open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), neck_record["num_frames"])

                events = [
                    entry["event"]
                    for entry in json.loads((turn_dir / "timeline.json").read_text(encoding="utf-8"))
                ]
                for required in (
                    "user_audio_end", "asr_complete", "thinking_start",
                    "dialogue_start", "dialogue_complete", "tts_complete",
                    "robot_motion_inputs_recorded", "speaker_generation",
                    "neck_trajectory_ready", "neck_rpy_recorded",
                    "neck_trajectory_sent", "turn_end",
                ):
                    self.assertIn(required, events)
                self.assertNotIn("listener_generation", events)


if __name__ == "__main__":
    unittest.main()
