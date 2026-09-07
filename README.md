# Project-Neck

Project-Neck 是仿生机器头的数据、深度学习研究与真机运行工程。仓库按三个独立领域组织：

```text
dataset/   Raw Data → Cleaning/ASR/Pose → Fragments → Canonical Dataset
algorithm/ Canonical Dataset → Training/Evaluation → Model Package
runtime/   Model Package + Real Input → Audio/Motion/Motor → Robot
```

`tools/` 保存跨领域项目工具。三个领域不通过源码互相耦合：Dataset 通过冻结的数据
artifact 服务 Algorithm；Algorithm 最终通过 checkpoint、配置、词表和预处理 metadata
交付模型；Runtime 独立加载部署资产，不 import Algorithm 训练源码。

## 目录

| 路径 | 职责 |
|---|---|
| `dataset/` | 独立的 Dataset V1 数据生产项目；不训练模型、不参与在线运行 |
| `algorithm/` | 正在从零重建的纯深度学习研究项目；当前只有骨架 |
| `runtime/` | 真机系统集成：Audio、ASR、Dialogue、TTS、旧 V3 motion inference、Motor 和实验记录 |
| `tools/` | 项目级分析与可视化工具 |

更具体的开发规则见根 `AGENTS.md` 以及各领域自己的 `AGENTS.md`。

## Runtime（先使用纯软件模式）

Runtime 的统一入口位于仓库根目录：

```bash
python3 -m runtime --mock-neck
```

该命令仍会加载 ASR、TTS 和部署 motion model，但不会连接 Motor。默认部署资产：

```text
runtime/models/whisper-base-ct2/
runtime/models/neck_motion_v3/checkpoints/best.pt
```

Audio 客户端从其项目目录运行：

```bash
cd runtime/audio
python3 -m runtime.main check-config
python3 -m runtime.main stream-test --duration 5 --wait-for-robot
```

固定 Audio 协议为 16 kHz、mono、PCM s16le、20 ms/640 bytes per frame；Runtime 默认
监听 `0.0.0.0:8765`。

## Motor

只编译、不启动硬件：

```bash
cmake -S runtime/motor -B runtime/motor/build
cmake --build runtime/motor/build
ctest --test-dir runtime/motor/build --output-on-failure
```

> **硬件警告：**不要把 `runtime/motor/build/master_stack_test` 当作测试程序运行。
> 它会初始化真实 EtherCAT；向 `/tmp/neck_model.sock` 发送合法轨迹可能立即驱动机器头。

Algorithm → Motor 的现有部署协议保持不变：Unix socket `/tmp/neck_model.sock`，完整
UTF-8 JSON，30 fps、`radian`、`[roll,pitch,yaw]`。

## Dataset

Dataset V1 的生产、schema、时间轴和防泄漏约束见：

```text
dataset/AGENTS.md
dataset/README.md
```

未经明确授权，不运行下载、MediaPipe、ASR、neck pose、fragment 或 split 重建命令，
不修改 `dataset/datasets/` 中已有 artifact。

## Algorithm

`algorithm/` 当前刻意保持为空骨架，后续将从 Dataset V1 重新设计 DataLoader、时间对齐、
模型、loss、训练和评估。不要继续扩展 Runtime 内冻结的旧 V3 inference 实现，也不要让
Runtime 长期 import `algorithm/`。

## 真机实验与可视化

Runtime 旁路实验记录位于：

```text
runtime/experiments/v0_trajectory/
```

项目级可视化：

```bash
python3 tools/trajectory_visualizer.py session_<id>
```

实验文件不是 Runtime 模块间通信总线；在线主链路始终使用 WebSocket 和 Unix socket。
