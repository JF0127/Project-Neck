# Robot Runtime

`runtime/` 包含 Project-Neck 全部在线真机运行与系统集成代码：

```text
Audio → ASR → Dialogue → TTS
      → deployed motion inference → RPY → Motor → measured feedback
      → best-effort experiment logging
```

## 目录

- `audio/`：麦克风、扬声器、固定 PCM WebSocket client；`audio/board/` 包含 ESP32 固件。
- `motor/`：trajectory parser、postprocessor、IK、EtherCAT/CAN 与硬件执行。
- `motion_model/`：当前 V3 checkpoint 所需的冻结 inference closure，不是训练项目。
- `models/`：本地部署模型，默认不提交 Git。
- `experiments/`：真机 Session/Turn 旁路记录，默认不提交 Git。
- 根部 Python 文件：ASR、Dialogue、TTS、Audio server、motion 编排和 Neck socket client。

Runtime 不 import `algorithm/`。未来模型应以 checkpoint、config、vocab 和 preprocessing
metadata 组成独立部署包，再通过 Runtime adapter 接入。

## 安全入口

从仓库根目录执行：

```bash
python3 -m runtime --mock-neck
```

`--mock-neck` 不连接 `/tmp/neck_model.sock`，但仍会加载 ASR/TTS/motion deployment assets。
只检查 CLI 和 import 可使用：

```bash
python3 -m runtime --help
```

默认资产：

```text
runtime/models/whisper-base-ct2/
runtime/models/neck_motion_v3/checkpoints/best.pt
```

Audio 客户端命令从 `runtime/audio/` 执行；Motor 只允许 configure/build/CTest 软件验证。
未经明确授权，不启动 Motor executable、EtherCAT、麦克风或 Socket 实发。

固定接口、部署参数和安全规则见 `runtime/AGENTS.md` 与根 `AGENTS.md`。
