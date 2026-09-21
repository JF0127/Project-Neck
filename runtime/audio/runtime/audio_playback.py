"""Buffered pcm_s16le speaker playback."""

import queue
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import config


class PlaybackBufferFull(RuntimeError):
    pass


@dataclass
class PlaybackStats:
    received_frames: int
    played_frames: int
    underruns: int
    max_queue_depth: int
    callback_statuses: int
    first_playback_perf: Optional[float] = None


class AudioPlayback:
    """Buffer robot PCM and play it through the selected local backend."""

    def __init__(
        self,
        queue_size: int = config.PLAYBACK_QUEUE_MAX_FRAMES,
        prebuffer_frames: int = config.PLAYBACK_PREBUFFER_FRAMES,
    ) -> None:
        self.frames: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_size)
        self.prebuffer_frames = prebuffer_frames
        self.received_frames = 0
        self.played_frames = 0
        self.underruns = 0
        self.max_queue_depth = 0
        self.callback_statuses = 0
        self.last_callback_status = ""
        self.first_playback_perf: Optional[float] = None
        self._stream: Optional[Any] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._started = False
        self._receiving = True
        self._silence = bytes(config.BYTES_PER_FRAME)
        self._on_playback_start: Optional[Callable[[], None]] = None

    @staticmethod
    def _set_pulse_sink_port() -> None:
        if shutil.which("pactl") is None or shutil.which("paplay") is None:
            raise RuntimeError("Linux Pulse audio requires pactl and paplay")
        result = subprocess.run(
            [
                "pactl",
                "set-sink-port",
                config.PULSE_SINK,
                config.PULSE_SINK_PORT,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "could not select Pulse speaker port: "
                f"{result.stderr.strip()}"
            )

    def open(self) -> None:
        if config.LOCAL_AUDIO_BACKEND == "pulse":
            self._set_pulse_sink_port()
            return
        if config.LOCAL_AUDIO_BACKEND != "sounddevice":
            raise RuntimeError(
                f"unsupported local audio backend: {config.LOCAL_AUDIO_BACKEND}"
            )
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed; run: python3 -m pip install -r requirements.txt"
            ) from exc

        self._stream = sd.RawOutputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=config.SAMPLES_PER_FRAME,
            callback=self._callback,
        )

    def _callback(self, outdata: Any, frame_count: int, time_info: Any, status: Any) -> None:
        if status:
            self.callback_statuses += 1
            self.last_callback_status = str(status)
        if frame_count != config.SAMPLES_PER_FRAME:
            outdata[:] = bytes(len(outdata))
            if self._receiving:
                self.underruns += 1
            return
        try:
            frame = self.frames.get_nowait()
        except queue.Empty:
            outdata[:] = self._silence
            if self._receiving:
                self.underruns += 1
            return
        if self.first_playback_perf is None:
            self.first_playback_perf = time.perf_counter()
            callback, self._on_playback_start = self._on_playback_start, None
            if callback is not None:
                callback()
        outdata[:] = frame
        self.played_frames += 1

    def enqueue(self, frame: bytes) -> None:
        if len(frame) != config.BYTES_PER_FRAME:
            raise ValueError(f"invalid robot PCM frame: {len(frame)} bytes")
        try:
            self.frames.put_nowait(frame)
        except queue.Full as exc:
            raise PlaybackBufferFull("playback queue is full") from exc
        self.received_frames += 1
        self.max_queue_depth = max(self.max_queue_depth, self.frames.qsize())
        if not self._started and self.frames.qsize() >= self.prebuffer_frames:
            self._start_stream()

    def _start_stream(self) -> None:
        if config.LOCAL_AUDIO_BACKEND == "pulse":
            return
        if self._stream is None:
            self.open()
        if not self._started:
            self._stream.start()
            self._started = True

    def _finish_pulse(
        self, on_playback_start: Optional[Callable[[], None]] = None
    ) -> PlaybackStats:
        self._receiving = False
        pcm_frames: list[bytes] = []
        while True:
            try:
                pcm_frames.append(self.frames.get_nowait())
            except queue.Empty:
                break
        pcm = b"".join(pcm_frames)
        if not pcm:
            return self.stats()

        self._process = subprocess.Popen(
            [
                "paplay",
                f"--device={config.PULSE_SINK}",
                "--raw",
                "--format=s16le",
                f"--rate={config.SAMPLE_RATE}",
                f"--channels={config.CHANNELS}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        process = self._process
        self.first_playback_perf = time.perf_counter()
        if on_playback_start is not None:
            on_playback_start()
        _, stderr = process.communicate(input=pcm)
        self._process = None
        if process.returncode != 0:
            raise RuntimeError(
                "paplay failed: "
                + stderr.decode("utf-8", errors="replace").strip()
            )
        self.played_frames = self.received_frames
        return self.stats()

    def finish(
        self, on_playback_start: Optional[Callable[[], None]] = None
    ) -> PlaybackStats:
        """Play all queued frames, then stop and close the output device."""
        if config.LOCAL_AUDIO_BACKEND == "pulse":
            return self._finish_pulse(on_playback_start)
        self._receiving = False
        self._on_playback_start = on_playback_start
        if self.received_frames and not self._started:
            self._start_stream()

        timeout = time.monotonic() + self.frames.qsize() * config.FRAME_DURATION_MS / 1000 + 2.0
        while self.played_frames < self.received_frames and time.monotonic() < timeout:
            time.sleep(0.005)
        if self.played_frames < self.received_frames:
            raise RuntimeError("timed out while draining playback queue")

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._started = False
        return self.stats()

    def stats(self) -> PlaybackStats:
        return PlaybackStats(
            received_frames=self.received_frames,
            played_frames=self.played_frames,
            underruns=self.underruns,
            max_queue_depth=self.max_queue_depth,
            callback_statuses=self.callback_statuses,
            first_playback_perf=self.first_playback_perf,
        )

    def close(self) -> None:
        self._receiving = False
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

        stream, self._stream = self._stream, None
        if stream is not None:
            if self._started:
                stream.abort()
            stream.close()
        self._started = False
