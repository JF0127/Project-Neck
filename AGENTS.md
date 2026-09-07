# Project-Neck 统一开发规则

本文件适用于整个仓库。进入 `dataset/`、`algorithm/`、`runtime/` 或硬件子目录后，还必须
阅读该目录的 `AGENTS.md`；硬件安全和冻结协议始终以本文件为最低约束。

## 1. 三个独立领域

```text
dataset/   数据生产：Raw Data → Canonical Dataset
algorithm/ 深度学习研究：Dataset → Model Package
runtime/   真机运行：Model Package + Real Input → Robot Behavior
```

- `dataset/` 负责下载、清洗、ASR/word timestamps、视觉姿态、fragment 和 grouped split。
  它不训练模型，不参与在线 Runtime。
- `algorithm/` 是纯训练与研究项目，当前从零重建。它不得包含麦克风、TTS、Socket、IK、
  EtherCAT 或真机状态机。
- `runtime/` 负责 Audio、ASR、Dialogue、TTS、部署推理、轨迹发送、Motor、反馈和真机实验。
  它不得长期 import `algorithm/` 的训练源码。
- `tools/` 保存项目级工具。运行数据不是模块间通信接口。

正式交付边界是 artifact：Dataset 提供 manifest/样本；Algorithm 提供 checkpoint、配置、
词表与预处理 metadata；Runtime 独立加载部署包。

## 2. 当前状态

- Dataset V1 已完成并冻结；具体 contract 见 `dataset/AGENTS.md`。
- 新 Algorithm 尚未实现，禁止把旧 V3/CVAE/MultiCandidate 描述成新架构。
- Runtime 为保持现有真机能力，在 `runtime/motion_model/` 内保留了当前 V3 checkpoint 所需的
  最小 inference closure。这是部署实现，不是新 Algorithm 的训练源码。
- 当前 Runtime 没有 VAD/自动 endpoint；`stream_end` 是唯一 endpoint。Dialogue 仅 fixed/echo。
- Motor 只有现有 parser/IK/位置速度检查及 trajectory postprocessor；不得宣称完整硬件安全闭环。

## 3. 冻结接口：Audio ↔ Runtime

Runtime 是 WebSocket server，默认 `0.0.0.0:8765`；Audio 是 client。

固定 PCM：16,000 Hz、mono、signed 16-bit little-endian、`pcm_s16le`、20 ms/frame、
320 samples/640 bytes。Binary frame 是裸 PCM。

控制消息：

```json
{"type":"stream_start","stream_id":"user_<id>","source":"user","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
{"type":"stream_end","stream_id":"user_<id>"}
```

Robot audio 使用同一协议且 `source="robot"`。不得只修改单端格式、时序、buffer 或错误语义。
相关实现位于 `runtime/audio/runtime/`、`runtime/audio/tools/pcm_ws_server.py`、
`runtime/audio_server.py` 和 `runtime/tts.py`。

## 4. 冻结接口：Runtime → Motor

同机 Unix Domain Stream Socket：`/tmp/neck_model.sock`。每次连接发送一个完整 UTF-8 JSON，
client `shutdown(SHUT_WR)`，EOF 为消息边界。当前没有应用层 ACK；EOF 不证明执行成功。

```json
{
  "name":"model_output",
  "fps":30,
  "unit":"radian",
  "order":["roll","pitch","yaw"],
  "trajectory":[[0.0,0.0,0.0]],
  "states":["silent"]
}
```

正式合同固定 30 fps、radian、`[roll,pitch,yaw]`。每帧三个 finite number；states 若存在，
只能是 `speaking/listening/silent` 且等长。不得修改单位、轴序、符号、Socket 或 JSON schema。
两端位于 `runtime/{motion.py,neck_client.py}` 和 `runtime/motor/neck/`。

## 5. 真实入口

```bash
# 顶层 Runtime；先使用 mock Neck
python3 -m runtime --mock-neck

# Audio 客户端（从 Audio 项目目录）
cd runtime/audio
python3 -m runtime.main check-config
python3 -m runtime.main duplex-test --turns 2 --duration 5

# Motor 只编译和软件测试
cmake -S runtime/motor -B runtime/motor/build
cmake --build runtime/motor/build
ctest --test-dir runtime/motor/build --output-on-failure
```

不要把 `runtime/motor/build/master_stack_test` 当成普通测试运行。

## 6. 修改规则

1. 修改前检查工作树；不得覆盖他人未提交修改。
2. 以当前代码为准；文档冲突先记录，不为匹配旧文档而回退实现。
3. 保持小步、局部、可测试；不做无关重构。
4. 不修改冻结协议、Motor 标定、IK 系数、限位、电流或速度参数来让测试通过。
5. Runtime 主链路保持内存/WebSocket/Unix socket；`runtime/experiments/` 只能旁路记录。
6. Dataset artifact 不得隐式重算、移动、修复或改写；训练统计只能使用 Train。
7. 新 Algorithm 不复用 `runtime/motion_model/` 作为训练基础；从 Dataset V1 contract 重新设计。
8. Python 从 compile/import/unit test 开始；Motor 允许 configure/build/test，但禁止启动 executable。

## 7. 硬件安全铁律

未经用户明确要求：

- 不运行 `sudo` Motor executable、`master_stack_test`、`NeckPoseSet`、`NeckSequence`；
- 不向真实 `/tmp/neck_model.sock` 发送轨迹；
- 不执行 EtherCAT/CAN 命令，不连接真实麦克风或电机；
- 不放宽 RPY、电机位置、速度、加速度、jerk 或停止相关限制。

合法 Socket 数据会被 Motor 直接提交执行。算法或 parser 测试必须使用纯软件 fixture/mock。

## 8. 当前优先级

1. 保持新仓库边界、Runtime 现有链路和冻结协议稳定。
2. Algorithm 下一阶段先证明 Dataset V1 的真实 timestamp、invalid mask 和 target 语义正确，
   再实现 baseline；本次结构重构不包含新模型。
3. Runtime 的旧 V3 仅用于现有部署，未来通过独立 model package/adapter 替换。
