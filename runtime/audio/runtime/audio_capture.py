"""macOS microphone capture as fixed-size PCM frames."""

import queue
from typing import Any, Optional

from . import config


class MicrophoneCapture:
    """Capture 20 ms pcm_s16le frames without doing I/O in the callback."""

    def __init__(self, queue_size: int = config.CAPTURE_QUEUE_MAX_FRAMES) -> None:
        self.frames: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_size)
        self._stream: Optional[Any] = None
        self.dropped_frames = 0
        self.invalid_frames = 0
        self.callback_status_count = 0
        self.last_callback_status = ""

    def _callback(self, indata: Any, frame_count: int, time_info: Any, status: Any) -> None:
        # This executes on PortAudio's callback thread: never block here.
        if status:
            self.callback_status_count += 1
            self.last_callback_status = str(status)

        frame = bytes(indata)
        if frame_count != config.SAMPLES_PER_FRAME or len(frame) != config.BYTES_PER_FRAME:
            self.invalid_frames += 1
            return

        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            # Preserve real-time behavior rather than blocking the audio callback.
            self.dropped_frames += 1

    def start(self) -> None:
        if self._stream is not None:
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed; run: python3 -m pip install -r requirements.txt"
            ) from exc

        self._stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=config.SAMPLES_PER_FRAME,
            callback=self._callback,
        )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def read_frame(self, timeout: Optional[float] = None) -> bytes:
        """Read one complete frame outside the audio callback thread."""
        return self.frames.get(timeout=timeout)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def __enter__(self) -> "MicrophoneCapture":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
