"""Buffered pcm_s16le speaker playback."""

import queue
import time
from dataclasses import dataclass
from typing import Any, Optional

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


class AudioPlayback:
    """Connect an async network receiver to PortAudio through a bounded Queue."""

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
        self._stream: Optional[Any] = None
        self._started = False
        self._receiving = True
        self._silence = bytes(config.BYTES_PER_FRAME)

    def open(self) -> None:
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
        if self._stream is None:
            self.open()
        if not self._started:
            self._stream.start()
            self._started = True

    def finish(self) -> PlaybackStats:
        """Play all queued frames, then stop and close the output device."""
        self._receiving = False
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
        )

    def close(self) -> None:
        self._receiving = False
        stream, self._stream = self._stream, None
        if stream is not None:
            if self._started:
                stream.abort()
            stream.close()
        self._started = False
