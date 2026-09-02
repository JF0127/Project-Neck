# Project-Neck 全链路运行与测试手册

Project-Neck 是机器头语音交互、颈部动作生成和三电机执行的统一工程。本文只说明**当前代码真实可用**的启动入口、测试顺序和安全边界。

> **硬件警告：**`modules/motor/build/master_stack_test` 没有 mock/no-EtherCAT 模式。运行它会初始化真实 EtherCAT；向其 `/tmp/neck_model.sock` 发送合法轨迹可能立即驱动机器头。首次测试必须先走纯软件链路。

## 1. 项目结构

```text
Project-Neck/
├── modules/
│   ├── audio/       # 麦克风、扬声器、PCM WebSocket
│   ├── algorithm/   # Algorithm Runtime、ASR、Dialogue、TTS、Neck Motion
│   └── motor/       # trajectory parser、IK、电机轨迹、EtherCAT/CAN
├── models/
│   └── whisper-base-ct2/  # faster-whisper 模型
├── experiments/     # 旁路实验记录
├── tools/           # 项目级实验分析工具
├── AGENTS.md        # 全项目开发与安全约束
└── README.md        # 本运行手册
```

| 模块 | 当前职责 |
|---|---|
| `modules/audio` | 麦克风采集、扬声器播放、有界 Audio Queue、16 kHz PCM WebSocket 收发 |
| `modules/algorithm` | Algorithm Runtime、整段 ASR、fixed/echo Dialogue、TTS、listener/speaker motion generation、最终 RPY trajectory |
| `modules/motor` | 扁平 trajectory JSON parser、radian→degree、IK、位置/速度 precheck、三电机轨迹执行、EtherCAT/CAN |

## 2. 当前整体链路

```text
Microphone
   ↓
modules/audio
   ↓ WebSocket：JSON control + Binary PCM
modules/algorithm / Algorithm Runtime
   ├──→ TTS PCM → WebSocket → modules/audio → Speaker
   │
   └──→ final 30 fps RPY trajectory
              ↓
         /tmp/neck_model.sock
              ↓
         modules/motor
              ↓
         parser / IK / Motors
```

- Audio ↔ Algorithm：跨机器局域网 WebSocket；Algorithm 是 server，Audio 是 client。
- Algorithm ↔ Motor：同一 Ubuntu 主机 Unix Domain Stream Socket。
- 主链路只使用内存、WebSocket 和 Unix Socket；`experiments/` 不参与数据传输。

### 冻结传输格式

Audio PCM 固定为 16000 Hz、mono、`pcm_s16le`、20 ms/frame、320 samples、640 bytes/Binary Frame。控制帧使用 `stream_start` / `stream_end`。

Algorithm → Motor 的正式 JSON 固定为 30 fps、radian、`[roll,pitch,yaw]`：

```json
{
  "name": "model_output",
  "fps": 30,
  "unit": "radian",
  "order": ["roll", "pitch", "yaw"],
  "trajectory": [[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]],
  "states": ["speaking", "speaking"]
}
```

不要将 `neck_motion/mvp.py --export-json` 的旧嵌套 JSON 直接发给当前 Motor parser。

## 3. 机器与环境

### Audio Computer

当前 `modules/audio` 面向 macOS：

```bash
cd ~/projects/Project-Neck/modules/audio
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

需要 PortAudio 和麦克风权限；macOS 缺 PortAudio 时可使用 `brew install portaudio`。

Audio 默认连接地址定义在 `runtime/config.py`：

```text
ws://10.255.0.35:8765
```

这是当前部署默认值，不是永久协议。可按部署覆盖：

```bash
python3 -m runtime.main stream-test --url ws://<algorithm-ip>:8765 ...
# 或
export AUDIO_MODULE_WS_URL=ws://<algorithm-ip>:8765
```

### Algorithm / Motor Computer

当前两者运行在同一 Ubuntu 主机：

- Algorithm 默认 bind：`0.0.0.0:8765`
- Motor Socket：`/tmp/neck_model.sock`
- `modules/algorithm/.venv` 当前文档环境为 Python 3.10
- Motor 文档环境为 Ubuntu 20.04、CMake、C++17、Boost、Readline，SOEM 位于仓库 third-party

Algorithm Runtime 还需要环境中存在：

- v3 candidates checkpoint，默认：`modules/algorithm/outputs/neck_motion_v3/checkpoints/best.pt`
- checkpoint 对应词表（由 checkpoint 的 `vocab_path` 指定）
- faster-whisper 模型，默认：`Project-Neck/models/whisper-base-ct2/`
- torch、numpy、scipy、soundfile、faster-whisper、edge-tts 等依赖
- edge-tts 可访问的网络

当前仓库忽略 `modules/algorithm/outputs/` 和 `Project-Neck/models/whisper-base-ct2/`，干净 checkout 不包含这些部署资产。`requirements-runtime.txt` 目前也只补充 WebSocket 依赖，不能作为完整环境安装清单。

Motor 当前 `main.cpp` 实际硬编码 EtherCAT 网卡为 `enp4s0`；`neck/neck_config.py` 中的 `network_interface` 尚未被入口读取。部署到其他主机时必须核对源码、网卡和重新编译，不能只改配置后假设生效。

## 4. 推荐测试顺序

不要第一次就启动真实 Motor。

```text
Level 1  Audio 单模块检查
Level 2  Audio PCM WebSocket 传输验证
Level 3  Algorithm Runtime + mock Neck
Level 4  Algorithm → Motor Socket（仅在硬件安全检查后）
Level 5  Audio + Algorithm + Motor 完整实机链路
```

## 5. Level 1 — Audio 单模块

```bash
cd ~/projects/Project-Neck/modules/audio
```

### 固定格式检查（纯软件安全）

```bash
python3 -m runtime.main check-config
```

预期输出确认 16000 Hz、mono、int16、20 ms、320 samples、640 bytes/frame。

### 麦克风采集（连接真实麦克风，不连接网络/Motor）

```bash
python3 -m runtime.main capture-test --duration 5 --output capture.wav
```

输出 WAV 应为 16 kHz、mono、16-bit。当前没有独立的 playback-only CLI；扬声器播放通过收到 robot PCM 的双向测试验证。

## 6. Level 2 — Audio PCM WebSocket 传输

若只想检查局域网、PCM 帧和 Audio 播放，不加载 ASR/模型，可在接收端使用验证 server。

### Terminal A — 接收端验证 server（纯软件，不连接 Motor）

接收用户 PCM：

```bash
cd ~/projects/Project-Neck/modules/audio
python3 tools/pcm_ws_server.py --host 0.0.0.0 --port 8765 --save-dir received_audio
```

若要回传一段 robot PCM 以测试扬声器：

```bash
python3 tools/pcm_ws_server.py \
  --host 0.0.0.0 --port 8765 \
  --save-dir received_audio \
  --reply-wav recordings/test.wav
```

仓库当前的 `recordings/test.wav` 已是未压缩 16 kHz、mono、16-bit PCM；替换文件时必须保持该格式。

### Terminal B — Audio Computer

只上传麦克风：

```bash
cd ~/projects/Project-Neck/modules/audio
python3 -m runtime.main stream-test \
  --duration 5 --url ws://<server-ip>:8765
```

等待并播放验证 server 返回的 robot audio：

```bash
python3 -m runtime.main stream-test \
  --duration 5 --url ws://<server-ip>:8765 --wait-for-robot
```

该级别连接真实麦克风/扬声器，但不运行 ASR、模型或 Motor。

## 7. Level 3 — Algorithm Runtime + mock Neck

### 启动 Algorithm

```bash
cd ~/projects/Project-Neck/modules/algorithm
.venv/bin/python -m runtime --mock-neck
```

`--mock-neck` 模式仍会真实执行：

- 接收 Audio PCM；
- faster-whisper ASR 和 word timestamps；
- fixed/echo Dialogue；
- edge-tts；
- listener/speaker Neck Motion；
- composition、boundary blend、silent return；
- 最终 RPY schema 校验与旁路实验记录。

但它**不会连接 `/tmp/neck_model.sock`，不会启动 Motor**。

常用部署覆盖：

```bash
.venv/bin/python -m runtime \
  --mock-neck \
  --host 0.0.0.0 --port 8765 \
  --checkpoint /path/to/best.pt \
  --whisper-model /path/to/faster-whisper-model
```

当前 Runtime 没有 VAD：只在 Audio 发出 `stream_end` 后处理完整用户 PCM。

### Audio ↔ Algorithm 两终端测试

**Terminal A — Algorithm Computer**

```bash
cd ~/projects/Project-Neck/modules/algorithm
.venv/bin/python -m runtime --mock-neck
```

**Terminal B — Audio Computer**

```bash
cd ~/projects/Project-Neck/modules/audio
python3 -m runtime.main duplex-test \
  --turns 2 --duration 5 --url ws://<algorithm-ip>:8765
```

预期过程：

```text
用户说话 → PCM 上传 → stream_end
→ ASR → Dialogue → TTS
→ robot PCM 返回 → Audio 扬声器播放
```

Algorithm 同时生成完整 Neck RPY，但 mock 模式只校验/打印，不驱动 Motor。`duplex-test` 要求至少两轮并复用同一 WebSocket；单轮可用：

```bash
python3 -m runtime.main stream-test \
  --duration 5 --url ws://<algorithm-ip>:8765 --wait-for-robot
```

当前 Audio CLI 是有限时长测试入口，没有无限持续运行 daemon 命令。

### 不使用麦克风的 Runtime 记录回归测试（纯软件安全）

```bash
cd ~/projects/Project-Neck/modules/algorithm
.venv/bin/python -m unittest tests.test_experiment_logging -v
```

该测试使用 fake ASR/TTS/Motion 和 mock Neck，不连接 Audio、网络或 Motor。

## 8. Motor 编译与单独检查

### 编译（纯软件安全，不启动电机）

```bash
cd ~/projects/Project-Neck/modules/motor
cmake -S . -B build
cmake --build build
```

编译不会启动 EtherCAT 或电机。

### 真实 Motor 入口（危险）

```bash
cd ~/projects/Project-Neck/modules/motor
sudo ./build/master_stack_test
```

> **危险：该命令会创建 `/tmp/neck_model.sock`，随后无条件调用 `EtherCAT_Init("enp4s0")` 并启动真实 EtherCAT Runtime。当前不存在 `--no-ethercat`、mock Motor server 或 Motor dry-run server。**

只有完成网卡、从站、电机中心、机械限位、急停和低速空载检查后才能运行。

## 9. Level 4 — Algorithm → Motor Socket

Motor 是 `/tmp/neck_model.sock` server，Algorithm 和发送工具是 client。Socket 使用“一次连接发送一个完整 JSON，以 client 写侧 EOF 结束”的方式。

当前发送工具：

```bash
cd ~/projects/Project-Neck/modules/motor
python3 tools/model_socket_client.py <trajectory.json> \
  --socket /tmp/neck_model.sock
```

> **危险：如果该 Socket 后面是正在运行的真实 `master_stack_test`，合法 trajectory 会立刻进入 `executeTrajectory()`，可能直接驱动三台电机。此工具不是 dry-run。**

### 协议验证与真实执行的区别

- `modules/algorithm -m runtime --mock-neck`：只在 Algorithm 侧验证最终 JSON，不连接 Motor，安全。
- `cmake --build build`：只编译，安全。
- `tools/model_socket_client.py` 连接真实 Motor：可能执行硬件，不是纯协议检查。
- 当前 Motor 没有独立 parser-only CLI、`--no-ethercat` 或 dry-run server。
- Socket server 不返回应用层 ACK/错误 JSON；client 正常结束只代表连接关闭，必须查看 Motor 终端日志确认 parser、precheck、IK 和提交结果。

## 10. Level 5 — 完整全链路实机测试

只有在 Level 1–3 全部通过、硬件安全检查完成后执行。

### Step 1 — Algorithm/Motor Ubuntu 主机：启动 Motor

```bash
cd ~/projects/Project-Neck/modules/motor
sudo ./build/master_stack_test
```

**此步骤会启动真实 EtherCAT/Motor。确认 Motor 终端成功找到从站，并出现：**

```text
[ModelSocket] listening on /tmp/neck_model.sock
```

### Step 2 — 同一 Ubuntu 主机另一个终端：启动 Algorithm

```bash
cd ~/projects/Project-Neck/modules/algorithm
.venv/bin/python -m runtime
```

这里故意不使用 `--mock-neck`。Runtime 会在每轮生成完整轨迹后连接 `/tmp/neck_model.sock`。

### Step 3 — Audio Computer：启动有限多轮测试

```bash
cd ~/projects/Project-Neck/modules/audio
python3 -m runtime.main duplex-test \
  --turns 2 --duration 5 --url ws://<algorithm-ip>:8765
```

若只测试一轮：

```bash
python3 -m runtime.main stream-test \
  --duration 5 --url ws://<algorithm-ip>:8765 --wait-for-robot
```

### 完整预期链路

```text
User voice
→ modules/audio microphone
→ WebSocket PCM
→ modules/algorithm ASR → Dialogue → TTS
→ robot PCM → WebSocket
→ modules/audio speaker
```

同时：

```text
User PCM / robot PCM + word timestamps
→ listener/speaker Motion Generation
→ composition / blend / silent return
→ final 30 fps RPY
→ /tmp/neck_model.sock
→ modules/motor parser / precheck / IK
→ Motors
```

注意：Algorithm 在发送 trajectory 后再等待到 speaking 状态边界发送 robot PCM；但 Motor Socket 当前没有应用层执行 ACK，因此音频同步不能证明 Motor 已接受或正确执行轨迹，必须同时观察 Motor 日志。

## 11. 全链路 Checklist

### 软件与语音

```text
[ ] Audio check-config 通过
[ ] 麦克风 capture-test 有有效 WAV
[ ] Algorithm checkpoint、词表、Whisper 模型存在
[ ] edge-tts 网络可用
[ ] Algorithm 正在 0.0.0.0:8765 监听
[ ] Audio 使用正确的 Algorithm IP/port
[ ] WebSocket 连接成功
[ ] user stream_start / 640-byte PCM / stream_end 正常
[ ] ASR 有结果，Runtime 进入 THINKING/SYNTHESIZING
[ ] TTS 有输出
[ ] robot PCM 返回 Audio
[ ] 扬声器正常播放
[ ] listener/speaker generation 和最终 Neck trajectory 已生成
[ ] trajectory 是 30 fps、radian、[roll,pitch,yaw]
```

### 真实硬件（仅实机测试）

```text
[ ] EtherCAT 网卡名称和接线确认
[ ] 从站数量正确
[ ] 电机 ID、passage、中心和机械限位确认
[ ] 急停可用并已复测
[ ] /tmp/neck_model.sock 存在
[ ] Motor 日志显示收到 valid trajectory
[ ] Motor precheck 通过，没有位置/速度拒绝
[ ] IK 输出正常
[ ] 已从低速、空载、小幅轨迹开始
[ ] 实机执行和停止正常
```

## 12. 实验记录

Algorithm Runtime 默认旁路保存到：

```text
experiments/v0_trajectory/sessions/
└── session_YYYYMMDD_NNN/
    ├── session_config.json
    ├── session_timeline.json
    └── turns/
        └── turn_NNN/
            ├── dialogue.json
            ├── timeline.json
            ├── neck_rpy.json
            ├── neck_rpy.csv
            └── model_generations/
                ├── generation_001_listener.json
                └── generation_002_speaker.json
```

```text
Session
└── Turn
    └── Motion Generation
```

- Session：一次 Runtime 启动后的完整测试。
- Turn：一轮用户到机器人交互。
- `model_generations/`：候选选择后、整轮 composition 前的独立原始模型轨迹。
- `neck_rpy.json/csv`：Algorithm 完成 composition、blend 和 silent return 后，实际准备发送给 Neck 的完整轨迹。
- 后续 Motor `TrajectoryPostprocessor` 可旁路增加 processed trajectory 和对比图，本阶段尚未实现。
- 原有 `experiments/v0_trajectory/runs/` 是旧的手动诊断数据，继续保留。

记录失败只打印 warning，不阻断主链路。可用 `--experiment-root` 改记录位置；不要让任何模块通过 `experiments/` 文件交换运行数据。

## 13. 轨迹可视化

从项目根目录按 Session 批量分析：

```bash
python3 tools/trajectory_visualizer.py session_<id>
```

也可以直接传入 Session 路径：

```bash
python3 tools/trajectory_visualizer.py \
  experiments/v0_trajectory/sessions/session_<id>
```

工具按 turn 编号处理所有可用 `neck_rpy.json` 和 `measured_rpy.json`，输出 roll/pitch/yaw position、velocity、acceleration 图和统计到对应 Session 的 `visualizations/turn_NNN/`。

## 14. 常见启动组合

### A. 只测试 Audio

```bash
cd ~/projects/Project-Neck/modules/audio
python3 -m runtime.main check-config
python3 -m runtime.main capture-test --duration 5 --output capture.wav
```

### B. 测试完整语音闭环，不动电机

```bash
# Algorithm Computer
cd ~/projects/Project-Neck/modules/algorithm
.venv/bin/python -m runtime --mock-neck

# Audio Computer
cd ~/projects/Project-Neck/modules/audio
python3 -m runtime.main duplex-test \
  --turns 2 --duration 5 --url ws://<algorithm-ip>:8765
```

### C. 只测试 Algorithm → Motor

```text
1. 完成硬件安全检查后启动真实 master_stack_test
2. 使用 model_socket_client 发送已人工审核的小幅轨迹，或启动不带 --mock-neck 的 Algorithm
3. 同时观察 Motor parser/precheck/IK 日志
```

当前没有安全的 Motor dry-run server，因此该组合属于实机测试。

### D. 完整实机系统

```text
1. modules/motor：sudo ./build/master_stack_test
2. modules/algorithm：.venv/bin/python -m runtime
3. modules/audio：duplex-test 或单轮 stream-test --wait-for-robot
```

## 15. 安全与常见定位

### 可能驱动真实机器头的操作

- `sudo ./build/master_stack_test`
- `tools/model_socket_client.py` 向真实 `/tmp/neck_model.sock` 发送合法 JSON
- 不带 `--mock-neck` 的 Algorithm Runtime 在真实 Motor Socket 存在时发送轨迹
- Motor CLI 的 `NeckPoseSet`、`NeckSequence` 等命令

执行前必须确认机械限位、电机中心、Motor ID/passage、EtherCAT 网卡、急停和人员安全区域。当前模型轨迹仍存在抖动/高动态问题；Motor 目前只会对超限位置/相邻帧速度进行拒绝，尚无正式 acceleration/jerk `TrajectoryPostprocessor`。在后处理完成并验证前，不建议直接执行高动态模型轨迹。

### 常见问题

| 现象 | 优先检查 |
|---|---|
| Audio 连接失败 | Audio `--url`、Algorithm IP、8765、防火墙、Runtime 是否完成模型加载并开始监听 |
| Runtime 启动失败 | checkpoint、checkpoint 词表、Whisper 路径、Python 依赖、edge-tts 网络 |
| 收到用户音频但不处理 | 当前 endpoint 是显式 `stream_end`，没有 VAD |
| robot audio 超时 | ASR/TTS 异常、edge-tts 网络、Runtime ERROR 日志、Audio `--robot-timeout` |
| `/tmp/neck_model.sock` 不存在 | Motor 未成功启动、EtherCAT 初始化/从站失败、Socket 权限 |
| client 正常退出但 Motor 不动 | Socket 无 ACK；检查 Motor parser、precheck、busy、IK 和速度超限日志 |
| Motor 找不到配置 | 从 `modules/motor` 根目录启动；当前只查找 `neck/neck_config.py` 或 `../neck/neck_config.py` |
| Motor 找不到网卡/从站 | `main.cpp` 当前硬编码 `enp4s0`；核对实际网卡、接线、权限并重新编译 |
| trajectory 被拒绝 | 检查 finite number、30 fps/radian/order、RPY/电机范围和相邻帧电机速度 |

修改代码前请先阅读根目录 [`AGENTS.md`](AGENTS.md)。
