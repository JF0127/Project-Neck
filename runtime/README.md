# Robot Runtime

`runtime/` contains Project-Neck online integration:

```text
User Audio → Whisper ASR → user_text
                              ↓
                    DeepSeek Dialogue
                              ↓
                         robot_text
                              ↓
                          Edge TTS
                              ↓
                Baseline V1 speaker motion
                              ↓
                 speaker RPY + neutral tail
                              ↓
                         Neck Motor
```

The motion path is speaker-only. User audio is used by ASR and Dialogue but is not a
motion-model input. Runtime directly reuses `algorithm.features.FeatureEncoder` and
`algorithm.models.baseline.BaselineModel` so the first deployment version has exactly
the training feature semantics.

## Dialogue

DeepSeek is used only for `user_text → robot_text`; it does not receive or control
motion. The default backend remains `fixed` so offline and mock startup never require a
network or API key. Formal DeepSeek startup uses the OpenAI-compatible API:

```bash
export DEEPSEEK_API_KEY="your-key-here"
python3 -m runtime \
  --baseline-checkpoint /path/to/best.pt \
  --dialogue deepseek
```

Defaults are `https://api.deepseek.com`, model `deepseek-v4-flash`, non-thinking mode,
a 30-second timeout, at most 128 output tokens, and five successful conversation turns
of in-process history. The SDK performs no automatic retry. The key is read only from
`DEEPSEEK_API_KEY` and is never written to experiment records.

Useful dialogue overrides are `--deepseek-model`, `--deepseek-base-url`,
`--dialogue-history-turns`, and `--dialogue-timeout`. `--dialogue fixed` and
`--dialogue echo` remain available for offline tests.

## Safe startup

A Baseline V1 checkpoint is required; no timestamped training run is hard-coded:

```bash
python3 -m runtime \
  --mock-neck \
  --baseline-checkpoint /path/to/best.pt \
  --dialogue fixed
```

`--mock-neck` does not connect to `/tmp/neck_model.sock`. Initialization still loads the
local ASR and Baseline models and starts the PCM WebSocket server. A real Turn additionally
uses the configured Dialogue backend and Edge TTS network service.

Useful options:

```text
--motion-device auto|cpu|cuda   (default: auto)
--whisper-model PATH
--host HOST                     (default: 0.0.0.0)
--port PORT                     (default: 8765)
--no-experiment-log
```

## Fixed interfaces

Audio ↔ Runtime:

- Runtime WebSocket server, default `0.0.0.0:8765`;
- 16 kHz, mono, signed PCM s16le;
- 20 ms / 320 samples / 640 bytes per Binary frame;
- JSON `stream_start` and `stream_end` control frames.

Runtime → Motor:

- Unix stream socket `/tmp/neck_model.sock`;
- complete UTF-8 JSON delimited by client write EOF;
- 30 fps, radians, `[roll,pitch,yaw]`;
- speaker frames use `speaking`; the return-to-neutral tail uses `silent`.

The Motor protocol, parser, IK, limits, and trajectory postprocessor are unchanged.

## Baseline query grid

For TTS duration `duration_sec`:

```text
N = max(2, round(duration_sec * 30))
query_timestamps_sec = [0/30, 1/30, ..., (N-1)/30]
```

The raw Baseline `[N,3]` `rpy_offset` prediction is retained unchanged. Runtime only
validates it and appends a 0.8-second linear return-to-neutral tail.

## Experiment records

Unless disabled, each Turn records under:

```text
runtime/experiments/v0_trajectory/sessions/session_*/turns/turn_*/
```

including `dialogue.json`, `robot_audio.wav`, `robot_words.json`,
`robot_motion_input.json`, one speaker generation, the final `neck_rpy.json/csv`,
timeline events, and optional Motor-produced
`measured_rpy.json`. Dialogue records include the backend, model, request latency, and
available token usage, but never credentials or complete SDK response objects.

Hardware safety and frozen protocol details are in `runtime/AGENTS.md` and root
`AGENTS.md`.
