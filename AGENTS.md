# Project-Neck 统一开发规则

本文件适用于仓库根目录及全部子目录。进入子模块后还必须阅读该模块自己的文档；若子目录 `AGENTS.md` 有更具体的规则，则在不违反本文件的模块边界、冻结接口与硬件安全要求的前提下遵守它。

## 1. 项目目标与当前阶段

Project-Neck 是一个**高拟人多模态交互机器人头部系统**。核心链路是：

```text
用户语音
  → modules/audio 采集并发送 PCM
  → modules/algorithm 完成识别、回复、TTS、状态调度和 Neck Motion
  → 机器人回复 PCM → modules/audio → Speaker
  → RPY trajectory → modules/motor → IK → Motors
```

当前处于 **v0 系统集成与优化阶段**。整体双向音频链路、Algorithm → Motor Socket 和基础 Motor/IK/EtherCAT 框架已在代码中接通；当前首要问题是模型 RPY 轨迹抖动、速度/加速度过大，Motor 只能拒绝不满足速度限制的轨迹，尚无正式的硬件可执行性 `TrajectoryPostprocessor`。

v0 优先保证可运行、可定位、可视化、可调试和模块边界清晰，不要把项目提前扩展成重型科研实验平台。

## 2. 模块结构与边界

### `modules/audio`：声音 I/O 与传输

只负责：

- 麦克风采集、扬声器播放；
- 有界 Audio Buffer / Queue 和音频设备管理；
- 固定格式 PCM 的 WebSocket 发送、接收与基础协议校验。

禁止放入：VAD、ASR/Whisper、word timestamps、TTS 语义生成、log-Mel/模型特征、Dialogue、Neck Motion Model、RPY 生成、IK 或电机控制。

> Audio 只负责声音输入、声音输出和传输。

### `modules/algorithm`：算法与系统 Runtime

负责：

- 接收用户 PCM；VAD/Endpoint、ASR、词时间戳、Dialogue/Reply、TTS；
- 音频/文本特征、speaker/listener/silent 状态调度；
- Neck Motion Model、候选选择、旋转合成、段间 blend、算法级轨迹后处理；
- 输出 30 fps 的 RPY trajectory；系统 Runtime 与轮次状态。

当前代码中的正式 Runtime 尚未实现 VAD：唯一 endpoint 是收到 `stream_end` 后整段处理；Dialogue 目前只是 fixed/echo 策略，不是 LLM。不要把规划能力写成已经实现。

当前动作实现是 `neck_motion/candidates.py` 的 `MultiCandidateModel`，由 `ConditionEncoder`（audio log-Mel、word timestamps/text、previous text context、role）+ 片段级 K 个 latent candidates + Transformer `SequenceDecoder` + speaker/listener heads 组成。Runtime 要求 checkpoint 内 `config.model.type == "candidates"`，在线按候选能量选择 listener/speaker 轨迹，再做四段 composition：listener → thinking silent → speaker → silent return。

**部署资产现状：**代码默认 checkpoint 为 `modules/algorithm/outputs/neck_motion_v3/checkpoints/best.pt`，文档称其为 v3、K=8；但 `outputs/` 被忽略且当前工作树中没有该 checkpoint、词表或默认 Whisper 目录 `Project-Neck/models/whisper-base-ct2/`。因此目前只能确认代码要求和默认路径，不能从本仓库验证实际部署 checkpoint 的内部配置。不要用 `neck_motion/config.yaml` 的 `model.type: regression` 推断线上模型；部署时必须检查实际 checkpoint 的 `config`、`vocab_path` 和权重。

Algorithm 不得知道 MotorId、电机中位/零点、EtherCAT、CAN、IK 内部公式或厂家参数。

### `modules/motor`：颈部执行与硬件

负责：

- 接收并严格解析 RPY JSON；轨迹、RPY 范围和相邻帧电机速度验证；
- 后续硬件侧 trajectory postprocessing；
- degree 内部表示、inverse kinematics、三电机目标生成与同步帧执行；
- EtherCAT/CAN、反馈、停止、限位与硬件安全。

Motor 不得处理 ASR、TTS、文本、Dialogue 或模型推理。`states` 仅是 metadata，不参与当前电机控制。

> Algorithm 只关心“头应该如何运动”；Motor 只关心“如何把 RPY 变成真实电机运动”。

### `dataset/`：离线数据集工程

`dataset/` 是 Project-Neck 顶层独立的离线数据集生产系统，负责视频发现与下载、原始数据管理、数据清洗、fragment 构建、ASR/audio/MediaPipe/neck-pose 特征提取、dataset split、数据质量分析，并为 Algorithm Module 提供训练/验证/测试数据。它不属于机器人实时 runtime；实时模块仍位于 `modules/audio/`、`modules/algorithm/` 和 `modules/motor/`。具体数据集输出位于 `dataset/datasets/`。

### `experiments`

`experiments/v0_trajectory/` 是旁路实验记录，不是 Runtime 数据总线。正式 Runtime 使用 Session → Turn → Motion Generation：Session 是一次完整测试，Turn 是一轮交互，Generation 是一次独立模型轨迹生成；旧 `runs/run_*/segment_*` 手动诊断数据继续保留。v0 轻量保存原始 generation、整轮 final RPY，以及后续 processed RPY、position/velocity/acceleration 图、对比图和简单参数即可。

任何模块不得依赖 `experiments/` 文件来交换运行时数据。主链路始终是 WebSocket + Unix Domain Socket；实验记录只能旁路复制。

## 3. 冻结接口：Audio ↔ Algorithm

这是跨机器局域网 WiFi/LAN 接口，未经明确的系统级决定不得修改。

### 角色与部署参数

- **Algorithm 是 WebSocket server**：`modules/algorithm/runtime/audio_server.py`，默认 bind `0.0.0.0:8765`，可由 `python -m runtime --host/--port` 配置。
- **Audio 是 WebSocket client**：默认 URL 在 `modules/audio/runtime/config.py`，当前为 `ws://10.255.0.35:8765`；可用 CLI `--url` 或环境变量 `AUDIO_MODULE_WS_URL` 覆盖。
- IP、bind host、port 是部署时可配置参数；帧格式和消息语义是固定协议。

### 固定 PCM

- 16000 Hz，mono，signed 16-bit little-endian，`pcm_s16le`；
- 20 ms/frame，320 samples/frame，640 bytes/Binary Frame；
- Binary Frame 是裸 PCM，无 WAV header；只有最后一帧的发送方 TTS 路径可补零到 640 bytes。

### 控制消息和时序

控制消息是 UTF-8 JSON Text Frame：

```json
{"type":"stream_start","stream_id":"user_<id>","source":"user","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
```

随后发送连续的 640-byte Binary Frames，最后：

```json
{"type":"stream_end","stream_id":"user_<id>"}
```

Robot Audio 反向使用相同协议，`source` 必须为 `robot`，通常使用新的 `robot_<id>`。`stream_end` 只含 `type` 和匹配的 `stream_id`。一条连接上同一方向一次只能有一个 active stream；当前 duplex test 会复用同一 WebSocket 完成多轮。

### 当前 buffer 与异常行为

- Audio capture/playback Queue 各最多 250 帧（5 s）；capture callback 满时丢帧并计数，playback Queue 满时报错；播放预缓冲 5 帧（100 ms）。
- Algorithm 将完整 user stream 保存在内存中，收到 `stream_end` 后才 ASR/推理；上限 5 分钟 PCM。当前没有流式 ASR、VAD、自动 endpoint、重连或断点续传。
- 非 640-byte Binary Frame、格式/来源/stream_id 错误会关闭连接；Algorithm 使用 WebSocket close code 1011，Audio 测试 server 的协议错误使用 1008。
- 修改协议、buffer、超时或异常策略时必须同时检查 `modules/audio/runtime/{protocol.py,websocket_client.py,audio_capture.py,audio_playback.py}`、`modules/algorithm/runtime/audio_server.py`、`tts.py` 和 `modules/audio/tools/pcm_ws_server.py`。

## 4. 冻结接口：Algorithm → Motor

同一 Ubuntu 主机使用 Unix Domain Stream Socket：

```text
/tmp/neck_model.sock
```

Algorithm 是 client，Motor 是 server。每次连接发送**一个完整 UTF-8 JSON 文档**，然后 client `shutdown(SHUT_WR)`；EOF 是消息边界。Motor 最大接收 16 MiB，解析/提交后关闭连接。目前没有应用层 ACK/错误响应内容，client 等待 EOF 只表示连接处理结束，不证明电机执行成功。

冻结 JSON：

```json
{
  "name": "model_output",
  "fps": 30,
  "unit": "radian",
  "order": ["roll", "pitch", "yaw"],
  "trajectory": [
    [0.0, 0.0, 0.0],
    [0.01, -0.02, 0.03]
  ],
  "states": ["speaking", "speaking"]
}
```

约束：

- 正式接口固定 `fps = 30`、`unit = "radian"`、`order = ["roll","pitch","yaw"]`；不得改单位、轴顺序或符号语义。
- `name`、`fps`、`unit`、`order`、非空 `trajectory` 是 Motor parser 的必需字段；Algorithm Runtime 还要求 `states` 必须存在且等长。
- Motor parser 允许省略 `states`；若存在，只允许 `speaking/listening/silent` 且必须与轨迹等长。该字段仅作 metadata。
- 每帧必须恰好三个 finite number。Motor 严格拒绝未知顶层字段。
- Motor 当前代码只检查 `fps > 0`，但这不是改变冻结 30 fps 合同的许可；Algorithm sender 明确要求 `30.0`。
- Motor 将输入 `[roll,pitch,yaw]` radians 显式转换为内部 `TrajectoryPoint{pitch_deg,roll_deg,yaw_deg}`，然后做 IK。

修改该接口时默认先停止并征得明确许可；确需修改时必须同时检查：

1. `modules/algorithm/runtime/{motion.py,neck_client.py}`；
2. `modules/motor/neck/{model_socket.cpp,trajectory_io.cpp,trajectory.h}`；
3. `modules/motor/tools/model_socket_client.py`、本地 trajectory JSON 和 `tools/trajectory_visualizer.py`；
4. 双方 README/HANDOFF 与本文件。

不要把旧 `mvp.py --export-json` 的嵌套 V1 格式（`format_version` + `trajectory.rpy`、`units`、数字 states）直接发给当前 Motor parser；它与正式扁平 Socket schema 不兼容。

## 5. 当前真实入口

所有命令默认从对应子项目根目录执行。

### Audio

```bash
cd modules/audio
python3 -m runtime.main check-config
python3 -m runtime.main capture-test --duration 5 --output capture.wav
python3 -m runtime.main stream-test --duration 5 --wait-for-robot
python3 -m runtime.main duplex-test --turns 2 --duration 5
```

主 CLI：`modules/audio/runtime/main.py`。WebSocket 验证 server：

```bash
python3 tools/pcm_ws_server.py --host 0.0.0.0 --port 8765 --save-dir received_audio
```

### Algorithm Runtime 与模型

正式 Runtime：

```bash
cd modules/algorithm
.venv/bin/python -m runtime --mock-neck
# 只有明确准备接 Motor 时才去掉 --mock-neck
```

入口是 `runtime/__main__.py`，编排在 `runtime.py`；ASR/TTS/motion 分别在 `asr.py`、`tts.py`、`motion.py`。默认 `0.0.0.0:8765`、checkpoint `outputs/neck_motion_v3/checkpoints/best.pt`、Whisper `../../models/whisper-base-ct2`、neck socket `/tmp/neck_model.sock`，均有 CLI 配置项。

模型入口：

```bash
python neck_motion/infer.py --checkpoint <best.pt> --input <fragment.json>
python neck_motion/mvp.py --checkpoint <best.pt> --events <events.json>
```

`infer.py` 是旧的单段/候选池文件输出入口；`mvp.py` 是离线三态调度与旧嵌套 JSON 导出入口；正式内存 Runtime 使用 `runtime/motion.py`，不要混淆三者的输出协议。训练/评估入口为 `train.py`、`eval.py`、`diagnose.py`，当前不是 v0 的优先工作。

### Motor

```bash
cd modules/motor
cmake -S . -B build
cmake --build build
sudo ./build/master_stack_test
```

`main.cpp` 无 mock 参数：启动时会先创建 `/tmp/neck_model.sock`，随后**无条件初始化真实 EtherCAT**（当前硬编码网卡 `enp4s0`）并启动 CLI。不要把历史文档中的 `--server`、`--no-ethercat`、`NeckTrajDryRun` 或 `/tmp/neck_ctl.sock` 当成当前入口。

Socket 发送工具：

```bash
python3 tools/model_socket_client.py <trajectory.json> --socket /tmp/neck_model.sock
```

该工具本身不启动 Motor，但若连接到运行中的真实 Motor，合法轨迹会立即提交硬件执行；它不是 dry-run 工具。人工 CLI 的 `NeckPoseSet`、`NeckSequence`、`NeckSequenceStop` 同样会操作硬件。

### 轨迹可视化与实验记录

```bash
python3 tools/trajectory_visualizer.py session_<id>
```

工具按 Session 批量分析各 Turn 的 position/velocity/acceleration，并写入对应 Session 的 `visualizations/turn_NNN/`。

## 6. 修改规则

1. 修改前先阅读根 `AGENTS.md`，再读相关模块的 README、AGENTS、HANDOFF 和实现；Motor 修改必须读 `modules/motor/AGENTS.md`，但以当前代码和本文件冻结边界为准。
2. 先确认工作树状态。当前仓库可能已有其他人的未提交修改；不得回退、覆盖、格式化或“顺手修复”不属于当前任务的内容。
3. 以当前代码行为为准；文档冲突时先记录冲突，不要为了匹配旧文档而回退新实现。
4. 优先最小、局部、可回滚修改。不做无关重构，不增加无意义 wrapper/helper，不大量添加解释性注释，不随意移动/重命名文件。
5. 不随意改变公共接口、默认音频格式、Socket 路径、RPY 单位/顺序、旋转约定或硬件标定参数。
6. 跨模块修改必须检查两端并提供兼容性验证；默认避免修改冻结接口。
7. Runtime 主链路保持内存/Socket 传输；不要引入 WAV、NPY、JSON 或 `experiments/` 文件作为模块间强依赖。
8. 新增 postprocessor 时先做纯软件、确定性接口，保留 raw 与 processed 轨迹用于对比；Algorithm 原始输出不得被静默覆盖。
9. 测试从最小静态/单元验证开始。Python 可先 `py_compile`；Motor 可编译但不得因测试算法而启动 executable。任何未执行的硬件测试必须明确说明。

## 7. Motor 与硬件安全铁律

- 未经用户明确要求，不运行 `sudo ./build/master_stack_test`，不调用 `NeckPoseSet`、`NeckSequence`、Socket 实发或任何 EtherCAT/CAN 命令。
- 不因测试 Algorithm、parser 或 postprocessor 而隐式启动 EtherCAT。优先 mock sender、纯 parser、离线 fixture 和可视化。
- 当前 Motor Socket 收到合法轨迹后直接调用 `executeTrajectory()`；没有 dry-run 模式或应用层确认。连接真实进程即可能运动。
- 保守处理 RPY 限位、电机位置、速度、加速度、jerk、奇异点、反馈和停止。不得放宽限制、修改电机中心/方向/IK 系数或 current/speed 参数来“让测试通过”。
- 当前仅有 RPY/电机位置验证和相邻帧 motor velocity 拒绝；尚无 acceleration/jerk postprocessor、重定时或完整反馈安全闭环。不要宣称已具备实机直接驱动资格。
- 后处理应先输出 diagnostics 并支持纯软件验证，再经低速、空载、急停与回中测试逐级进入实机。

## 8. 已验证的文档冲突与技术债务

- **checkpoint/config：**`config.yaml` 默认是 regression；Runtime 要求 candidates，文档指定 v3 checkpoint，但 checkpoint/词表在当前仓库缺失，无法核验真实部署 metadata。
- **Endpoint/VAD：**冻结职责把 VAD/Endpoint 放在 Algorithm；当前 Runtime `vad_filter=False` 且只以显式 `stream_end` 结束一轮，没有 VAD。
- **模型能力：**v3 代码具备 audio/text/context/role 条件结构，但 HANDOFF 的实验结论是当前 checkpoint 主要是 speaker/listener 角色动作先验，内容/语义/节奏响应没有稳定证据。
- **`mvp.py` 与正式 Socket schema：**`mvp.py --export-json` 仍输出嵌套旧格式；正式 Runtime 的 `runtime/motion.py` 才输出 Motor 可解析的扁平格式。
- **Motor 文档/配置：**`modules/motor/AGENTS.md` 示例称 motor3 center 为 177°，当前 `neck/neck_config.py` 是 145°；以配置和重新确认的实机标定为准，禁止猜改。该文档还强调简化旧安全体系，不能解释为可弱化本文件的保守硬件规则。
- **网卡配置未统一：**`neck_config.py` 有 `network_interface`，但 `main.cpp` 仍硬编码 `enp4s0`，启动时没有从配置读取。
- **Motor 后处理缺口：**当前 `precheckNeckTrajectory` 遇到速度超限直接拒绝，没有 raw→processed 平滑、限加速度/jerk、重采样/重定时与对比输出。这是当前 P0 工作，而不是放宽速度限制。
- **Socket 可观测性：**Motor 不返回 ACK/错误 JSON，Algorithm 等待 EOF 后会打印“send complete”，不能知道 parser、IK 或执行提交是否成功；服务端串行接收，新轨迹在已有轨迹执行时会被拒绝。
- **Runtime 稳定性：**整段 PCM 常驻内存、全局单 Runtime state、无自动重连/取消/打断；多 client 并发不是可靠支持场景。
- **依赖记录不完整：**`requirements-runtime.txt` 只新增 WebSocket 依赖；Runtime 实际还依赖环境中已有的 torch/numpy/scipy/soundfile/faster-whisper/edge-tts 等，且 TTS 是在线服务。
- **Motor README 过旧：**`modules/motor/README.MD` 主要描述 2022 SOEM demo，未覆盖当前 Socket、trajectory parser 和 Neck Motion 行为。

## 9. 当前开发优先级

### P0

- 保持 Audio ↔ Algorithm ↔ Motor 数据链路和冻结协议稳定；
- 用 `tools/trajectory_visualizer.py` 分析 raw RPY 的 position/velocity/acceleration 与异常帧；
- 在 Motor 边界实现可纯软件验证的硬件可执行性 trajectory postprocessor，保留 raw/processed 对比；
- 在不放宽安全限制的前提下补齐平滑、速度、加速度、jerk 与必要重定时策略。

### P1

- Runtime 断连、超时、busy/error 恢复和 Socket 结果可观测性；
- 轻量日志与 `run_id`/`segment_id` 实验记录；
- postprocessor 参数、诊断和软件回归 fixture；
- 补齐部署资产/依赖说明并消除旧入口歧义。

### P2

- VAD/自动 endpoint、正式 Dialogue/LLM 和更完整流式能力；
- 模型条件能力与候选策略优化、训练和数据采集；
- 正式论文实验与更复杂 experiment tracking。

除非用户明确改变阶段，不要越过 P0/P1 直接进行大规模模型重训、科研平台建设或跨模块重构。
