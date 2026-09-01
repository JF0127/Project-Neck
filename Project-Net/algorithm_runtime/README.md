# Algorithm Runtime V1

正式入口：

```bash
.venv/bin/python -m algorithm_runtime --mock-neck
```

去掉 `--mock-neck` 后，轨迹 JSON 会发送到冻结接口
`/tmp/neck_model.sock`。Runtime 默认监听 `0.0.0.0:8765`。

## 固定接口

Audio WebSocket：`stream_start` → 640-byte Binary PCM → `stream_end`；双向均为
16 kHz、mono、`pcm_s16le`、20 ms/帧。`stream_end` 是 V1 唯一 endpoint。

Neck JSON：顶层 `name/fps/unit/order/trajectory/states`，30 fps、radian、
`[roll,pitch,yaw]`；通过 `/tmp/neck_model.sock` 一次发送完整 JSON。

## 运行链路

```text
user PCM（内存）
  ├→ resident faster-whisper → text + word timestamps
  └→ resident MultiCandidateModel listener motion
text → FixedDialogue/EchoDialogue → edge-tts（内存）→ quantized PCM16
robot PCM（同一份量化数据）
  ├→ WebSocket 640-byte frames
  └→ resident MultiCandidateModel speaker motion
listener + thinking silent + speaker + silent return
  → rotation composition / boundary blend
  → 30 fps Neck JSON
```

Runtime 不依赖 WAV、word JSON、events JSON 或 NPY 中间文件。`--mock-neck` 只验证并
打印 Neck JSON，不连接或启动 Motor。

## v0 旁路实验记录

Runtime 默认在根目录 `experiments/v0_trajectory/sessions/` 创建一个 Session，并按 Turn
保存 dialogue、关键时间线、独立 listener/speaker 原始 generation，以及发送 Neck 前的
最终 `neck_rpy.json/csv`。记录失败只告警，不改变 WebSocket 或 Neck Socket 主链路。
可用 `--experiment-root` 改保存位置，纯临时运行可显式使用 `--no-experiment-log`。

WebSocket 依赖记录在 `requirements-runtime.txt`。V1 不实现 VAD/LLM/流式 ASR或
流式 motion。
