"""Local microphone capture as fixed-size PCM frames."""

import queue
import shutil
import subprocess
import threading
from typing import Any, Optional

from . import config


class MicrophoneCapture:
    """Capture 20 ms pcm_s16le frames without doing I/O in the callback."""

    def __init__(self, queue_size: int = config.CAPTURE_QUEUE_MAX_FRAMES) -> None:
        self.frames: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_size)
        self._stream: Optional[Any] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._reader_thread: Optional[threading.Thread] = None
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

    @staticmethod
    def _set_pulse_source_port() -> None:
        if shutil.which("pactl") is None or shutil.which("parec") is None:
            raise RuntimeError("Linux Pulse audio requires pactl and parec")
        result = subprocess.run(
            [
                "pactl",
                "set-source-port",
                config.PULSE_SOURCE,
                config.PULSE_SOURCE_PORT,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "could not select Pulse microphone port: "
                f"{result.stderr.strip()}"
            )

    def _read_pulse_frames(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        pending = bytearray()
        while True:
            chunk = process.stdout.read(config.BYTES_PER_FRAME * 16)
            if not chunk:
                break
            pending.extend(chunk)
            while len(pending) >= config.BYTES_PER_FRAME:
                frame = bytes(pending[: config.BYTES_PER_FRAME])
                del pending[: config.BYTES_PER_FRAME]
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    self.dropped_frames += 1
        # A short tail is expected when parec is terminated between PCM frames.
        # It is not an invalid frame because it is never exposed to consumers.

    def _start_pulse(self) -> None:
        if self._process is not None:
            return
        self._set_pulse_source_port()
        self._process = subprocess.Popen(
            [
                "parec",
                f"--device={config.PULSE_SOURCE}",
                "--format=s16le",
                f"--rate={config.SAMPLE_RATE}",
                f"--channels={config.CHANNELS}",
                f"--latency-msec={config.FRAME_DURATION_MS}",
                f"--process-time-msec={config.FRAME_DURATION_MS}",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(
            target=self._read_pulse_frames,
            name="pulse-microphone-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def start(self) -> None:
        if config.LOCAL_AUDIO_BACKEND == "pulse":
            self._start_pulse()
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
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        reader, self._reader_thread = self._reader_thread, None
        if reader is not None:
            reader.join(timeout=1.0)
        if self._process is process:
            self._process = None

        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def __enter__(self) -> "MicrophoneCapture":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
