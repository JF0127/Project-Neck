# Project-Neck — 项目说明与开发契约

> 本文件是仓库**唯一**需要维护的说明文档。领域级旧文档（`dataset/AGENTS.md`、`algorithm/AGENTS.md`、`runtime/AGENTS.md`、`runtime/motor/AGENTS.md`、各 `README.md`）已删除，历史内容可用 `git show <commit>:<path>` 查阅。今后任何改动只同步本文件。
>
> 保留的仅是不需要同步的参考材料：`runtime/motor/README.MD`（SOEM 主站）、`runtime/motor/使用说明/使用文档.md`（电机手册）、`thirdparty/` 与 `runtime/audio/board/xiaozhi-esp32/`（vendored 外部工程）。

---

## 1. 项目定位

仿生机器头：数据生产 → 模型训练 → 真机运行。三个独立领域：

```text
dataset/    Raw Data            → Canonical Dataset
algorithm/  Dataset             → Training → Model Package
runtime/    Model Package + Real Input → Robot Behavior
tools/      项目级分析/可视化工具
```

边界：三个领域不通过源码互相耦合。Algorithm 通过冻结 artifact 消费 Dataset；Runtime 只加载部署包（TorchScript），**不 import `algorithm` 训练源码**（`runtime/inference/baseline_v1.py` 即该契约的实现）。

## 2. 当前状态（以代码为准）

| 领域 | 状态 |
|---|---|
| Dataset V1 | **完成并冻结**：`clean_v1` → `mediapipe_v1` → `neck_pose_v1` / `speech_v1` → `fragment_v1_2_1` → `split_v1`。当前 51 unique source、57 clean clips、471 fragments、约 1.55 h；split 36/10/5 sources，无泄漏 |
| Algorithm | Baseline V1（audio+text → 30 fps `rpy_offset`）已实现；训练产物在 `algorithm/outputs/`（gitignored）。部署包 `runtime/models/baseline/{model.pt,vocab.json,config.yaml}`（TorchScript，gitignored） |
| Runtime | 语音链 + 动作链 + 实时姿态反馈已跑通真机（见 §3）。生成产物在 `runtime/generated/`（gitignored） |
| Motor | SOEM EtherCAT 主站；三电机；`model` / `measurement` / `feedback` 三个 UDS；速度后处理。反馈需先手动归零（§4.4） |
| 模型质量 | 当前 Baseline 输出幅度约 1–2°，效果一般，后续会替换；接口固定在 `runtime/inference/MotionBackend` |

仓库当前**没有自动化测试目录**（已按维护成本约定删除）；验证依靠运行命令、`ctest`（motor 保留原有 C++ 测试）和手动真机测试。

## 3. Runtime 链路

```text
生成 + 执行（每轮）:
  user PCM ─→ VAD ─→ ASR(zh, faster-whisper large-v3) ─→ DeepSeek ─→ Edge TTS
                                                                        │
                     ┌──────────────────────────────────────────────────┤
                     │  robot.wav / metadata.json                       │  PCM
                     ▼                                                  ▼
   TorchScript 模型 → MotionProcessor → trajectory.json → /tmp/neck_model.sock
                     │                                                  │
                     └─────────────── 同时触发 ──────────────────────────┘
                                        robot 音频 → WebSocket 客户端播放

实时姿态（旁路，只读）:
  EtherCAT 反馈缓存 → FeedbackServer(30 Hz) → /tmp/neck_feedback.sock
    → MotorFeedbackMonitor → RobotState(head_rpy rad, valid, timestamp, motion_executing)
```

- 产物目录（每轮覆盖）：`runtime/generated/{robot.wav, trajectory.json, metadata.json}`；`metadata.json` 最后写入并带 `complete: true`，作为“生成完成”的标记。
- 兜底：生成失败 → 固定文本“抱歉，我没听清楚，请再说一遍。”+ 默认摇头轨迹（±3° yaw，1 Hz）；`head_rpy_valid == false` → 只播音频、不发轨迹。
- 音频与轨迹通过 `motion.sync_offset_ms` 做微调（当前为 0，真机验证为同步）。
- 轨迹发送前会检查 `RobotState.motion_executing`，避免与正在执行的轨迹冲突。

## 4. 冻结接口

### 4.1 Audio ↔ Runtime

Runtime 是 WebSocket **server**，默认 `0.0.0.0:8765`；Audio 是 client。

- 16 000 Hz、mono、signed 16-bit little-endian、`pcm_s16le`、20 ms/frame、320 samples / 640 bytes。
- Binary frame 是裸 PCM；控制消息：

```json
{"type":"stream_start","stream_id":"user_<id>","source":"user","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
{"type":"stream_end","stream_id":"user_<id>"}
```

Robot 音频使用同一协议且 `source="robot"`。不得单端修改格式、时序、buffer 或错误语义。

### 4.2 Runtime → Motor

同机 Unix Domain Stream Socket `/tmp/neck_model.sock`：每次连接发送一个完整 UTF-8 JSON，client `shutdown(SHUT_WR)`，EOF 为消息边界。**没有应用层 ACK**，EOF 不代表执行成功。

```json
{"name":"turn_xxx","fps":30,"unit":"radian","order":["roll","pitch","yaw"],
 "trajectory":[[0.0,0.0,0.0]],"states":["silent"]}
```

固定 30 fps、`radian`、`[roll,pitch,yaw]`；每帧三个 finite number；`states` 若存在只能是 `speaking/listening/silent` 且与 trajectory 等长。Motor 侧会把速度超限段插帧后执行。

### 4.3 Motor → Runtime 反馈（只读）

`/tmp/neck_feedback.sock`，30 Hz NDJSON；valid 帧：

```json
{"type":"motor_state","version":1,"timestamp":1789044128.302934,"valid":true,"motion_executing":false,
 "unit":"degree","order":["pitch","roll","yaw"],
 "head_rpy":[-0.012,0.015,0.0001],"motor_angles":[10.0,26.0,145.0],"sequence":[28580,28569,28558]}
```

invalid 帧：`{"type":"motor_state","version":1,"timestamp":...,"valid":false,"motion_executing":...,"reason":"feedback_unavailable"}`。
Runtime 侧 0.2 s 无消息即判定 stale（`head_rpy_valid=false`）。

### 4.4 归零规程

电机是问答模式：**没有命令就没有反馈**。每次 motor 启动后必须手动执行一次：

```text
NeckPoseSet 0 0 0 0
```

位置帧常驻后反馈持续有效；冷启动姿态不通过查询轮询解决，属于操作规程。

### 4.5 Motor Socket 汇总

| Socket | 方向 | 用途 |
|---|---|---|
| `/tmp/neck_model.sock` | Runtime → Motor | 轨迹 JSON |
| `/tmp/neck_feedback.sock` | Motor → Runtime | 30 Hz 姿态 NDJSON |
| `/tmp/neck_measurement.sock` | Runtime → Motor | 配置测量输出（可选，记录执行期实测 RPY） |

## 5. Dataset V1 关键契约（摘要）

详细历史契约见 git 历史中的 `dataset/AGENTS.md`，以下为训练/消费必须遵守的部分。

- **ID 层级**：`source_video_id` → `<source_video_id>_<shot:04d>`（clip）→ `<clip_id>_f<idx:04d>`（fragment）。Split 必须按 `source_video_id` 分组。
- **时间/单位**：全部 seconds；RPY 为 radian。Clean 之后的时间轴都是 clip-local；fragment 同时保存 source timeline 与 local timeline。
- **Neck GT（`neck_pose_v1`）**：`R_rel(t) = R0.T @ R(t)`（R0 为该 clip 首个 valid MediaPipe 帧）；语义轴映射 `roll = yaw_z`、`pitch = roll_x`、`yaw = pitch_y`；内部短 gap 用 SLERP（≤0.20 s）；按 final-valid run 做 1.5 Hz 零相位 Butterworth；invalid 保持 NaN，不压缩时间轴。
- **Fragment（`fragment_v1_2_1`）**：variable-length spoken utterance；阈值 `min 2.0s / strong_pause 0.45s / soft_max 20s / absolute_max 30s`；audio 16 kHz mono s16；neck 只做切片，不重算归一化/滤波/插值。
- **Split（`split_v1`）**：seed 42、70/20/10、按 source 分组。**禁止 random fragment split**；不得重新分配现有 source；训练统计量只从 Train 估计。
- **训练 target**：`rpy_offset = neck_rpy - neck_rpy[0]`，invalid mask 与 padding mask 分开；不得把 NaN 当 0。
- **禁止隐式重算**：DataLoader / 训练代码不得触发 download、clean、mediapipe、ASR、neck pose、fragment、split。

## 6. 目录结构

```text
Project-Neck/
├── AGENTS.md                  # 唯一维护文档
├── dataset/                   # 数据生产（V1 冻结）
│   ├── configs/               # youtube.yaml, mediapipe.yaml
│   ├── models/                # mediapipe task、faster-whisper-large-v3（gitignored）
│   ├── datasets/zhubo_shuo_lianbo/   # 全部 artifact（gitignored）
│   ├── src/                   # crawler / cleaning / features / fragments / splits / cli.py
│   └── tests/                 # 已删除
├── algorithm/                 # 训练与部署导出（当前 Baseline V1）
│   ├── configs/baseline.yaml
│   ├── data/ features.py models/ losses.py metrics.py train.py
│   └── tests/                 # 已删除
├── runtime/                   # 真机在线运行
│   ├── __main__.py runtime.py contracts.py config.yaml
│   ├── audio_server.py asr.py vad.py dialogue.py tts.py
│   ├── feedback.py            # 实时姿态 monitor
│   ├── neck_client.py         # motor JSON 发送
│   ├── inference/             # 模型 I/O 与产物（原 motion_core）
│   │   ├── base.py processor.py baseline_v1.py
│   │   ├── motor_json.py default_motion.py generator.py artifacts.py
│   ├── generated/             # 每轮生成的音频+轨迹（gitignored）
│   ├── models/                # 部署资产：baseline/, whisper-large-v3-ct2, silero_vad（gitignored）
│   ├── audio/                 # macOS/Linux 音频客户端（采集/播放/WS）
│   └── motor/                 # SOEM EtherCAT、三电机、三个 socket
└── tools/trajectory_visualizer.py
```

## 7. 常用命令

### 7.1 Runtime 环境（Ubuntu，一次性配置）

```bash
cd /home/jhl/projects/Project-Neck
uv venv --python 3.10 runtime/.venv
source runtime/.venv/bin/activate
uv pip install --torch-backend=auto torch
uv pip install -r runtime/requirements.txt
uv pip install -r runtime/audio/requirements.txt
# CTranslate2 需要 CUDA 12 的 cuBLAS/cuDNN（与 torch 的 CUDA 13 并存）：
uv pip install "nvidia-cudnn-cu12==9.*" nvidia-cublas-cu12
```

`runtime/.venv/bin/activate` 末尾需追加（重建 venv 后要重新加，否则 large-v3 在 GPU 上会报 `libcublas.so.12` 找不到）：

```bash
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
```

### 7.2 Runtime / Motor / Audio

```bash
# Ubuntu：启动 runtime（需要 DEEPSEEK_API_KEY 环境变量）
source runtime/.venv/bin/activate
python -m runtime

# Motor：只编译与软件测试（不要启动 executable，除非明确要做真机测试）
cmake -S runtime/motor -B runtime/motor/build
cmake --build runtime/motor/build -j"$(nproc)"
ctest --test-dir runtime/motor/build --output-on-failure

# 真机（需要明确授权，会初始化 EtherCAT；先手动归零）
sudo ./runtime/motor/build/master_stack_test
# 控制台：NeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>（度）、NeckSequence <name>、NeckSequenceStop

# Audio 客户端（Mac；服务端需先运行）
cd runtime/audio
brew install portaudio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export no_proxy="10.255.0.35,127.0.0.1,localhost"; export NO_PROXY="$no_proxy"  # 有代理时必须
python3 -m runtime.main check-config
python3 -m runtime.main capture-test --duration 3 --output /tmp/capture.wav
python3 -m runtime.main stream-test --duration 5 --wait-for-robot
python3 -m runtime.main duplex-test --turns 2 --duration 5
```

默认连接 `ws://10.255.0.35:8765`（可用 `AUDIO_MODULE_WS_URL` 或 `--url` 覆盖）。

### 7.3 Dataset（V1 已完成，重跑需明确授权）

```bash
cd dataset
python -m src.cli --help
# discover / validate / download / clean
# extract-mediapipe / validate-neck-pose / compare-neck-smoothing / extract-neck-pose
# extract-speech / build-fragments / build-split
```

### 7.4 Algorithm 训练

```bash
source runtime/.venv/bin/activate
python -m algorithm.train --config algorithm/configs/baseline.yaml --epochs 50
```

- 数据入口：`dataset/datasets/zhubo_shuo_lianbo/splits/split_v1/{train,val,test}.jsonl`。
- 输出：`algorithm/outputs/<experiment>/run_<timestamp>_seed42/`（checkpoint、config、vocab、metrics）。
- 部署包生成：`runtime/models/baseline/{model.pt,vocab.json,config.yaml}` 由 Algorithm 导出（历史导出脚本未入当前仓库；Runtime 只消费该产物，接口见 `runtime/models/baseline/config.yaml` 的 `input_contract: baseline_tensor_v1`）。

## 8. 配置速查

`runtime/config.yaml`：

| 段 | 关键字段 |
|---|---|
| `vad` | `backend: silero`, `model_path`, `threshold`, `min_speech_ms`, `min_silence_ms` |
| `asr` | `backend: whisper`, `model_path`（large-v3 CT2）, `device: cuda/cpu`, `language: zh` |
| `dialogue` | `backend: deepseek`, `model`, `base_url`, `timeout_sec`, `temperature` |
| `tts` | `backend: edge`, `voice`（中文） |
| `motion` | `enabled`, `model_path`, `vocab_path`, `device`, `generated_dir`, `sync_offset_ms` |
| `motor` | `feedback_enabled`, `feedback_socket`, `feedback_stale_sec`, `send_enabled`, `socket_path`, `measurement_socket`, `mock` |
| `runtime` | `dialogue_fallback_text`, `cooldown_ms` |

`runtime/motor/neck/neck_config.py`（motor 唯一硬件配置）：`network_interface`、`slave_id`、三电机 `passage/id/min/max/center/max_velocity/speed_param/current_param`、RPY 范围、运动学 `c11/c12/c21/c22/k3`、`feedback.enabled/socket_path/rate_hz`。**不得为了让测试通过而修改标定/限位/电流/速度。**

## 9. 硬件与安全铁律

未经用户明确要求，不得：

- 运行 `sudo` Motor executable、`master_stack_test`、`NeckPoseSet`、`NeckSequence`；
- 向真实 `/tmp/neck_model.sock` 发送轨迹、连接真实 Motor socket 或 measurement socket；
- 执行 EtherCAT/CAN 命令，或连接真实麦克风/电机做软件验证；
- 修改 IK 系数、机械限位、中心、方向、电流、速度、加速度、jerk 或停止检查；
- 放宽任何 RPY / 电机位置 / 运动安全限制来让测试通过。

合法 Socket 数据会被 Motor 直接提交执行。所有软件测试必须使用纯软件 fixture/mock。

## 10. 修改规则

1. 修改前检查工作树，不得覆盖他人未提交修改。
2. 保持小步、局部、可测试；不做无关重构。
3. 不修改冻结协议（§4）、Motor 标定、Dataset V1 artifact、`split_v1` 映射。
4. Dataset 的 expensive 阶段（download/ASR/mediapipe/neck/fragment/split）未经授权不重跑，不使用 `--force`。
5. Runtime 主链路保持内存/WebSocket/Unix socket；`runtime/experiments/` 只做旁路记录。
6. 新 Algorithm 不复用 runtime 的 inference 代码作为训练基础；Runtime 不 import `algorithm`。
7. **文档只维护本文件**：任何行为、命令、配置变化同步到这里；不要再新增领域级 AGENTS/README。
8. 真机测试前确认：motor 已启动并手动归零、反馈 `pose valid`、急停可达；停止顺序为先 runtime 后 motor。
