"""Fixed configuration for Audio Module V1."""

import os

# The wire audio format is frozen by the system architecture.
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
PCM_FORMAT = "pcm_s16le"
SAMPLE_WIDTH_BYTES = 2
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1_000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH_BYTES

# Audio callbacks put frames here; network I/O must consume them elsewhere.
CAPTURE_QUEUE_MAX_FRAMES = 250  # 5 seconds
PLAYBACK_QUEUE_MAX_FRAMES = 250
PLAYBACK_PREBUFFER_FRAMES = 5  # 100 ms

# Used by the WebSocket phase. The audio format above is intentionally not configurable.
WEBSOCKET_URL = os.getenv("AUDIO_MODULE_WS_URL", "ws://10.255.0.35:8765")
