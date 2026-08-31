# Robot Head Audio Module V1

The macOS Audio Module handles microphone capture, speaker playback, audio buffering,
audio-device management, and WebSocket transport only. It does not perform VAD, ASR,
TTS, text processing, robot motion, or motor control.

## Audio format

The wire format is fixed:

- 16,000 Hz, mono
- signed 16-bit little-endian PCM (`pcm_s16le`)
- 20 ms per frame
- 320 samples / 640 bytes per binary WebSocket frame

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If PortAudio is unavailable, install it with `brew install portaudio`. Grant microphone
permission to the terminal or Python host when macOS requests it.

## Local microphone validation

```bash
python3 -m audio_module.main check-config
python3 -m audio_module.main capture-test --duration 5 --output capture.wav
```

The second command saves a mono 16-bit/16 kHz WAV file. `--output` defaults to
`capture.wav`.

## PCM WebSocket test

On the Ubuntu computer (`10.255.0.35`), copy this repository, install the requirements,
and start the validation server:

```bash
python3 tools/pcm_ws_server.py --save-dir received_audio
```

It binds to `0.0.0.0:8765`, validates every control message and 640-byte PCM frame, prints
stream statistics, and optionally writes `received_<stream_id>.wav`.

To return a test WAV as real-time robot audio after each user stream:

```bash
python3 tools/pcm_ws_server.py \
  --save-dir received_audio \
  --reply-wav test_audio.wav
```

The WAV must already be uncompressed 16 kHz, mono, 16-bit PCM. The server sends one
640-byte binary frame every 20 ms and pads only a partial final frame with silence.

On the Mac, stream the microphone for five seconds without waiting for a reply:

```bash
python3 -m audio_module.main stream-test --duration 5
```

For the bidirectional playback test, use:

```bash
python3 -m audio_module.main stream-test --duration 5 --wait-for-robot
```

The client prebuffers five robot frames (100 ms), then plays through a bounded Queue and
prints received/played frame counts, underruns, callback statuses, and maximum Queue depth.

The default endpoint is `ws://10.255.0.35:8765`. Override it when testing locally:

```bash
python3 -m audio_module.main stream-test --duration 5 --url ws://127.0.0.1:8765 --wait-for-robot
```

It can also be changed with `AUDIO_MODULE_WS_URL`. Press Ctrl+C to stop early; the client
stops the microphone, attempts to send `stream_end`, and closes the WebSocket.
