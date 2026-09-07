# Xmini C3 Board Interface

## Purpose

本目录是 Project-Neck 的 Xmini C3 / XiaoZhi 语音板卡适配模块。当前目标板为：

- Board：`xmini/c3`
- MCU：ESP32-C3
- Flash：16 MB
- Audio codec：ES8311
- 推荐 ESP-IDF：v6.0.2

Xmini C3 保留原 XiaoZhi Cloud 语音链路：

```text
Microphone → Opus → XiaoZhi Cloud ASR / LLM / TTS → Opus → Speaker
```

同时，Xmini C3 通过独立局域网 TCP `BoardBridge`，将 Project-Neck 需要的数据旁路发送到 Ubuntu：

1. `user_audio`
2. `user_text`
3. `robot_text`
4. `robot_audio`
5. 用于分段的 `robot_audio_end` control event

**BoardBridge 是旁路，不是 XiaoZhi 主链路。** Ubuntu 断线、处理过慢或旁路 queue overflow 时，必须优先丢弃旁路音频，不得反向阻塞或破坏原有 ASR、LLM、TTS、Speaker 行为。

## Architecture

```text
                    XiaoZhi Cloud
                         │
                         │ ASR / LLM / TTS
                         │
                     Xmini C3
                    /         \
             Mic/Speaker     BoardBridge
                                │
                                │ TCP :8766
                                ▼
                              Ubuntu
                                │
                   tools/board_bridge_server.py
                                │
                         Session / Turn Data
```

- Xmini C3：Wi-Fi client，同时作为 BoardBridge TCP client。
- Ubuntu：BoardBridge TCP server。
- 当前开发部署地址在 `main/boards/xmini/c3/board_bridge.cc` 中为 `192.168.110.234:8766`。
- IP 是 deployment-specific 配置，不是永久协议字段；更换 Ubuntu 主机或网络时需要更新。
- TCP port 和 framing semantics 属于 Frozen V1 Contract。

BoardBridge sender 只能在网络栈初始化后启动。当前正确生命周期是：

```cpp
WifiBoard::StartNetwork();
BoardBridge::GetInstance().Start();
```

不要将 `BoardBridge::Start()` 移回 board constructor。

## Quick Start

在 Ubuntu 上只监听和观察：

```bash
python3 tools/board_bridge_server.py
```

推荐录制完整 user/robot turns：

```bash
python3 tools/board_bridge_server.py \
    --record-turns \
    --output-dir artifacts/board_bridge
```

增加有界 Opus/PCM 诊断：

```bash
python3 tools/board_bridge_server.py \
    --record-turns \
    --debug-audio \
    --output-dir artifacts/board_bridge
```

`--record-user-audio` 仍是 `--record-turns` 的 deprecated alias。旧 `--output` 已弃用，使用 `--output-dir`。

Server 默认 bind `0.0.0.0:8766`。启动后让 Xmini C3 正常连接 Wi-Fi 和 XiaoZhi Cloud；BoardBridge 会独立连接 Ubuntu。

## Wire Protocol

同一个 TCP byte stream 混合承载 NDJSON text/control 和 binary audio。所有整数均为 network byte order（big-endian）。Audio payload 上限为 8192 bytes。

### `user_text`

来源：XiaoZhi incoming JSON：

```text
type == "stt"
```

含义：当前用户 utterance 的最终 ASR 文本。

Wire format：UTF-8 NDJSON。

```json
{"type":"user_text","text":"你好。"}
```

文档中的 JSON 行在线上以 `\n` 结束。

### `robot_text`

来源：XiaoZhi incoming JSON：

```text
type == "tts"
state == "sentence_start"
```

含义：机器人 TTS sentence 文本。一个机器人回复可以包含多个 `robot_text`，消费者必须保持到达顺序，不能假设每轮只有一条。

Wire format：UTF-8 NDJSON。

```json
{"type":"robot_text","text":"你好呀。"}
```

### `user_audio`

来源：ESP32 已完成 Opus 编码、即将发送给 XiaoZhi Cloud 的同一个 `AudioStreamPacket::payload`。Observer 位于原 packet `std::move()` 到 Cloud protocol 之前。

Payload 是单个 **raw Opus packet**，不是 PCM、Ogg 或 WebM。

固定音频合同：

- Codec：Opus
- Sample rate：16000 Hz
- Channels：1
- Frame duration：60 ms
- 理论解码输出：960 samples/channel per packet

Binary frame：

| Offset | Size | Field |
|---:|---:|---|
| 0 | 1 | type = `0x01` |
| 1 | 4 | sequence, uint32 big-endian |
| 5 | 4 | payload_length, uint32 big-endian |
| 9 | N | raw Opus payload |

Sequence 是 BoardBridge 本地 uint32 counter，在每次 mirror attempt 开始时分配，即使随后 drop 也会消耗。

### `robot_audio`

来源：XiaoZhi Cloud 下行 `AudioStreamPacket`，在原 packet 被 `std::move()` 到 `AudioService::PushPacketToDecodeQueue()` 之前旁路复制。原 packet 继续进入 Opus decode、playback queue 和 Speaker。

Payload 同样是单个 raw Opus packet。

当前真机格式为：

- Sample rate：24000 Hz
- Channels：1
- Frame duration：60 ms

但 sample rate 和 frame duration 来自 server hello，不能在 Ubuntu consumer 中永久假设为 24 kHz。每个 frame 都携带实际 format metadata。

Binary frame：

| Offset | Size | Field |
|---:|---:|---|
| 0 | 1 | type = `0x02` |
| 1 | 4 | sequence, uint32 big-endian |
| 5 | 4 | payload_length, uint32 big-endian |
| 9 | 4 | sample_rate, uint32 big-endian |
| 13 | 2 | frame_duration_ms, uint16 big-endian |
| 15 | 1 | channels |
| 16 | N | raw Opus payload |

`AudioStreamPacket` 本身没有 channels 字段。V1 使用 channels=1，依据是 XiaoZhi hello 声明 mono，且 `AudioService` 下行 Opus decoder 明确配置为 mono。

Robot sequence 使用独立的 `robot_audio_sequence_`，不与 user sequence 共用。

### `robot_audio_end`

来源：XiaoZhi incoming JSON：

```text
type == "tts"
state == "stop"
```

Wire format：UTF-8 NDJSON control event。

```json
{"type":"robot_audio_end"}
```

它表示当前 robot TTS audio phase 结束。不要使用 `robot_text` 作为 audio end：`sentence_start` 之后 robot audio 可以继续数秒。

### TCP stream framing

Parser 根据下一 frame 的首字节分流：

```text
0x7B "{" → NDJSON，读取到 '\n'
0x01     → user_audio，读取 9-byte header，再读取 payload_length bytes
0x02     → robot_audio，读取 16-byte header，再读取 payload_length bytes
```

TCP 是 stream，不保留应用消息边界。实现必须正确支持：

- partial header
- partial payload
- fragmentation
- coalescing
- multiple frames per `recv()`
- text/audio/control 任意合法混合

禁止假设一次 `recv()` 等于一个 frame。

## Queue and Realtime Invariants

当前 BoardBridge queue：

| Queue | Capacity |
|---|---:|
| `text_queue` | 12 |
| `user_audio_queue` | 4 packets |
| `robot_audio_queue` | 4 packets |

必须保持以下原则：

- Producer callback 不执行 TCP I/O。
- Audio queue 满时立即 drop，不等待。
- TCP 未真正连接时，audio 在复制 payload 前直接 drop。
- Wi-Fi/TCP 断开或 send failure 时清空 user/robot audio queue。
- 不为重连缓存陈旧 audio；text queue 保持当前独立策略。
- Sequence 在 mirror attempt 时先分配，Ubuntu 可通过 gap 观察旁路 drop。
- Text/control 优先发送。
- User/robot audio 使用简单 alternating fairness。
- 只有一个低优先级 sender task和一个 TCP socket。
- 禁止多个 task 并发 `send()`，避免 frame interleave。

**BoardBridge 丢包优先于阻塞 XiaoZhi 主音频链。**

## Turn Model

一个完整 turn 定义为：

```text
User utterance + Robot response
```

正常生命周期：

```text
USER PHASE
first user_audio
→ create turn
→ decode/write user_audio.wav

user_text
→ user FINISHING
→ accept trailing user_audio for 300 ms
→ close/finalize user_audio.wav

ROBOT PHASE
robot_text sentence(s)
→ append in arrival order

first robot_audio
→ initialize decoder from frame sample_rate/channels
→ decode/write robot_audio.wav

robot_audio_end
→ robot FINISHING
→ accept trailing robot_audio for 300 ms
→ close/finalize robot_audio.wav
→ write text/metadata
→ complete turn
```

User 和 robot 均保留 300 ms trailing flush，因为每个 audio queue 最多 4 × 60 ms = 240 ms，而高优先级 text/control 可能先于已排队的尾部 audio 到达。

实现使用 `monotonic()` deadline 和 100 ms socket timeout检查，不使用 `sleep(0.3)` 阻塞 TCP receive/parser。

Turn 关联基于当前串行交互模型：user → robot → user → robot，不在 wire protocol 中增加 turn ID。

异常规则：

- `robot_text` / `robot_audio` 无 active user turn：warning 并忽略，避免污染正常数据。
- `robot_audio_end` 无 active turn：warning。
- 同一 robot turn format 改变：warning，拒绝该 packet，不让 Server 崩溃。
- TTS stop但无 robot audio：正常完成 turn，不创建损坏 WAV，metadata 中 `robot_audio` 为 `null`。
- 新 user audio 到达但上一 turn 缺失 end：先以 `missing_robot_audio_end` 保存上一 turn。
- Disconnect/Ctrl+C：关闭所有打开的 WAV 并写出 metadata。

## Recorded Artifacts

默认目录：

```text
artifacts/board_bridge/
└── session_YYYYMMDD_HHMMSS/
    ├── turn_001/
    │   ├── user_audio.wav
    │   ├── user_text.txt
    │   ├── robot_audio.wav
    │   ├── robot_text.txt
    │   └── metadata.json
    └── turn_002/
        └── ...
```

文件语义：

- `user_audio.wav`：16 kHz、mono、signed 16-bit little-endian PCM。
- `user_text.txt`：用户最终 ASR 文本，UTF-8。
- `robot_audio.wav`：实际 robot stream sample rate/channels、signed 16-bit little-endian PCM；当前真机流为 24 kHz mono。
- `robot_text.txt`：所有 `sentence_start` 文本按到达顺序每句一行，UTF-8。
- `metadata.json`：两侧音频统计、文本和 turn completion 状态。

若本轮没有 robot audio，`robot_audio.wav` 不存在，这是合法情况。

### Metadata contract

结构模板：

```json
{
  "turn_id": 1,
  "user_text": "...",
  "user_audio": {
    "codec_source": "opus",
    "sample_rate": 16000,
    "channels": 1,
    "sample_width_bytes": 2,
    "frame_duration_ms": 60,
    "received_packets": 0,
    "decoded_packets": 0,
    "decode_errors": 0,
    "sequence_gaps": 0,
    "samples": 0,
    "duration_seconds": 0.0,
    "first_sequence": null,
    "last_sequence": null
  },
  "robot_text": "...",
  "robot_text_sentences": ["..."],
  "robot_audio": {
    "codec_source": "opus",
    "sample_rate": 24000,
    "channels": 1,
    "sample_width_bytes": 2,
    "frame_duration_ms": 60,
    "received_packets": 0,
    "decoded_packets": 0,
    "decode_errors": 0,
    "sequence_gaps": 0,
    "samples": 0,
    "duration_seconds": 0.0,
    "first_sequence": null,
    "last_sequence": null
  },
  "completed_by": "robot_audio_end"
}
```

`robot_audio` 在无 robot audio packet 时为 `null`。

正常 turn：

```text
completed_by = "robot_audio_end"
```

异常值包括：

- `disconnect`
- `shutdown`
- `missing_robot_audio_end`

Consumer 不得假设所有 turn 都完整正常；必须检查 `completed_by`、`decode_errors` 和 `sequence_gaps`。

## Runtime Integration

需要实时处理时，不要轮询 `artifacts/` 或把 WAV 当作模块间实时接口。应连接或复用 `tools/board_bridge_server.py` 的 mixed-stream parser/handler，直接消费：

- `user_audio`
- `user_text`
- `robot_audio`
- `robot_text`
- `robot_audio_end`

`artifacts/board_bridge/` 只用于：

- debug
- dataset collection
- experiment record
- offline inspection

它不是正式 Runtime IPC，也不得成为 Algorithm/Audio/Motor 模块间的数据总线。

未来 Algorithm Runtime 正式接管 BoardBridge server 时，应优先复用 Frozen V1 wire contract，并在 Ubuntu 侧替换或抽取 handler；不要为了 Runtime 接入随意修改 ESP32 sender。

## Build

当前硬件必须构建：

```text
xmini/c3
```

不要使用 `xmini/c3-v3`。

```bash
source ~/esp/esp-idf/export.sh
python3 scripts/build.py xmini/c3
```

输出：

```text
build/xiaozhi.bin
build/merged-binary.bin
```

Build script 会改变本地 `sdkconfig` 和 build state；不要假设 build 目录仍对应此前 target。不要手工编辑 `build/`、`managed_components/`、`components/`、`releases/` 或 `sdkconfig*`。

## Flash

只有用户明确要求时才烧录。当前硬件验证可靠的 download mode 流程：

1. Unplug USB。
2. Hold BOOT。
3. Plug USB。
4. Release BOOT。
5. 确认设备为 `/dev/ttyACM0`。

烧录：

```bash
python3 -m esptool --chip esp32c3 \
    --no-stub \
    -p /dev/ttyACM0 \
    write-flash \
    --flash-mode dio \
    --flash-freq 80m \
    --flash-size 16MB \
    0x0 build/merged-binary.bin
```

当前硬件读取/烧录已验证：`--no-stub` 是可靠方式。

烧录后：拔 USB，不按 BOOT，正常重新插入。

Monitor：

```bash
source ~/esp/esp-idf/export.sh
idf.py -p /dev/ttyACM0 monitor --no-reset
```

## Frozen V1 Contract

以下接口已经冻结，原则上不要随意修改：

- TCP port和 mixed-stream framing semantics
- `user_text` JSON type/semantics
- `robot_text` JSON type/semantics
- `user_audio` type `0x01` header
- `robot_audio` type `0x02` header
- `robot_audio_end` NDJSON control event
- `user_audio = raw Opus / 16 kHz / mono / 60 ms`
- Robot audio format metadata随每个 frame 携带
- User/robot 独立 uint32 sequence及 mirror-attempt-before-drop 语义
- `session_*/turn_NNN/` artifact naming
- Turn内 `user_audio.wav`、`user_text.txt`、`robot_audio.wav`、`robot_text.txt`、`metadata.json` 命名

Ubuntu IP 是 deployment-specific，不属于冻结协议字段。

确需改变 Frozen V1 Contract 时，必须同步：

1. 修改 ESP32 sender；
2. 修改 Ubuntu parser/recorder；
3. 更新本文件；
4. 更新 fragmentation/coalescing、sequence 和 turn state-machine tests；
5. 重新执行 Xmini C3 真机四路及原 XiaoZhi 交互回归。

## Validation Status

当前硬件事实：ES8311、Microphone、Speaker、XiaoZhi ASR、LLM/TTS 均正常。

已真机验证：

| Item | Status |
|---|---|
| `user_audio` transport | PASS |
| `user_text` transport | PASS |
| `robot_text` transport | PASS |
| `robot_audio` transport | PASS |
| User Opus decode / playable WAV | PASS |
| Original XiaoZhi interaction unaffected | PASS |

已实现并通过 host smoke test，但当前记录中尚未明确完成最终真机落盘回归：

| Item | Status |
|---|---|
| Robot Opus decode / `robot_audio.wav` | IMPLEMENTED / PENDING FINAL HARDWARE CHECK |
| User 300 ms phase segmentation | IMPLEMENTED / HOST TESTED |
| Robot TTS stop + 300 ms segmentation | IMPLEMENTED / PENDING FINAL HARDWARE CHECK |
| Complete multi-turn recording | IMPLEMENTED / PENDING FINAL HARDWARE CHECK |

不要把成功 build 或 host smoke test描述为硬件验证。

## Do Not Do

后续 Agent 不要：

- 把 BoardBridge TCP I/O 放进 audio callback。
- 让 Ubuntu backpressure 阻塞 XiaoZhi 音频链。
- 直接把 raw Opus payload 拼接后当作 Ogg/WebM 文件。
- 假设一次 TCP `recv()` 对应一个 frame。
- 假设 robot audio 永远固定为 24 kHz。
- 用 `robot_text` 作为 robot audio end。
- 用 `xmini/c3-v3` 构建当前板卡。
- 为 BoardBridge 随意修改 `main/audio/`、I2S、ES8311、Opus 参数或 XiaoZhi Protocol wire implementation。
- 修改其他 board 来支持 Xmini C3。
- 将 `artifacts/` 文件轮询当作正式实时接口。
- 自动烧录或把成功烧录等同于完整硬件验证。

保留未提交的其他工作，不回退、不覆盖、不顺手格式化无关文件。Application state mutation仍必须通过 `Application::Schedule()` 或既有 event/state-machine 路径完成。

## Troubleshooting

### `/dev/ttyACM0` busy

```bash
sudo lsof /dev/ttyACM0
sudo fuser -k /dev/ttyACM0
```

确认没有残留 monitor、esptool 或串口程序。

### Board 停在 `DOWNLOAD waiting`

设备仍处于 download mode。拔掉 USB，然后不按 BOOT 正常重新插入。

### BoardBridge 未连接 Ubuntu

优先检查：

1. `main/boards/xmini/c3/board_bridge.cc` 中 deployment IP；
2. Ubuntu 是否运行 `tools/board_bridge_server.py`；
3. TCP port 8766 是否监听/被防火墙允许；
4. Xmini C3 与 Ubuntu 是否在可路由的同一局域网；
5. XiaoZhi Wi-Fi 是否已进入 Connected 状态。

Wi-Fi 未 Connected 时 BoardBridge 不会调用 `socket/connect`。

### Audio sequence gap

- 少量 gap：通常表示 BoardBridge 为保护主链路主动 drop。
- 大量持续 gap：检查 Ubuntu server处理速度、局域网、TCP reconnect和 queue overflow。
- User 和 robot sequence 独立，不要交叉比较。

### WAV 无声音或时长异常

依次检查：

1. `received_packets` / `decoded_packets`；
2. `decode_errors`；
3. `sequence_gaps`；
4. sample rate和 channels；
5. WAV duration与 packet count × frame duration是否大致一致；
6. `--debug-audio` 的 `decoded`、PCM min/max/RMS；
7. Robot decoder 是否使用 frame header metadata，而不是固定 16 kHz。

## Key Files

- `main/application.h`, `main/application.cc`
  - XiaoZhi semantic/audio observer hooks。
  - User audio observer位于 Cloud send前。
  - Robot audio observer位于 Speaker decode queue `std::move()` 前。
  - `tts/stop` 产生 robot audio end observer。

- `main/boards/xmini/c3/xmini_c3_board.cc`
  - Xmini C3 BoardBridge callbacks注册。
  - `WifiBoard::StartNetwork()` 后启动 BoardBridge。

- `main/boards/xmini/c3/board_bridge.h`, `board_bridge.cc`
  - ESP32 → Ubuntu TCP client。
  - 有界 queues、sequence、framing、drop/reconnect和单 sender调度。

- `tools/board_bridge_server.py`
  - Ubuntu TCP server。
  - NDJSON + user/robot binary mixed-stream parser。
  - ctypes/libopus decode。
  - Complete turn recorder和 artifacts输出。

- `main/protocols/protocol.h`
  - `AudioStreamPacket` 定义。

- `main/audio/audio_service.h`, `main/audio/audio_service.cc`
  - XiaoZhi 原始 Opus encode/decode、queue和 Speaker链路；BoardBridge 不应侵入此层。

- `scripts/build.py`
  - Canonical board build entry。

## Repository Guardrails

- 一个 build 只能通过 `DECLARE_BOARD(...)` 导出一个 board factory。
- 不要修改现有 board pins来适配不同硬件；新硬件使用独立 board/variant identity。
- Core code依赖 `Board` interface，不依赖具体 board class或其 `config.h`。
- Callback 可能不在 main task；不要直接在 callback 中阻塞或改变 Application state。
- Protocol contract变更必须同时验证 WebSocket 与 MQTT/UDP受影响路径。
- 只格式化实际修改的 C/C++ 文件，避免 mass formatting。
- 所有验证报告必须区分：build、host test和 physical hardware test。
