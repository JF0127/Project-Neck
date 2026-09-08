# Runtime 开发、架构与硬件安全规则

本文件适用于 `runtime/`。进入 `audio/`、`motor/` 或其他含 `AGENTS.md` 的子目录后，
还必须遵守对应子目录规则。冻结协议和硬件安全要求始终优先。

## 1. 项目边界与依赖

Project-Neck 只有三个核心领域：

```text
Dataset    = 数据生产
Algorithm  = 模型训练与导出
Runtime    = 真实机器人在线运行
```

依赖和 artifact 流向必须单向：

```text
Dataset → Algorithm → exported model → Runtime
```

Algorithm 负责训练并导出 PyTorch 部署模型。Runtime 只通过部署模型及其明确的输入/输出
Contract 与 Algorithm 协作。新的 Runtime 代码禁止直接：

```python
import algorithm
from algorithm ...
```

Algorithm 的训练 preprocessing 与 Runtime 的真实在线 preprocessing 是两套独立实现；
不得共享训练源码、Dataset loader、loss、训练入口或训练期 FeatureEncoder。两端共享的是模型
输入/输出 Contract，不是 Python 训练代码。

现有 `baseline_motion.py` 和顶层 `motion.py` 是尚未迁移的 legacy 参考实现，不得作为新 Runtime
架构的依赖基础。当前新 Motion 保持在 `motion_core/`；在 legacy `motion.py` 完整迁移并删除前，
不得将其改名为 `motion/`。

## 2. Runtime 职责

Runtime 负责：

- 真实在线数据 preprocessing；
- 加载 Algorithm 导出的部署模型；
- Runtime-specific model postprocessing；
- 公共 Motion processing；
- Audio、ASR、Dialogue、TTS、Motion、Motor 整体在线流程；
- Session、Turn、RobotState 生命周期；
- 运行记录。

运行记录只能是旁路 sink，不得作为 Audio、Motion 或 Motor 的在线数据总线。Runtime 主流程
未来统一由 `runtime.py` 管理，保持 one process = one robot = one Audio Client = one active
Session = at most one active Turn。不要无需求引入 EventBus、插件注册表或依赖注入框架。

当前 `__main__.py` 已接入新的语音 Runtime，只构造本地 VAD、ASR、Dialogue、TTS、Runtime
和 Audio WebSocket server。不得为了兼容旧入口把 `AlgorithmRuntime` 重新加入新架构；Motion、
Motor 和 Recorder 必须按后续阶段单独接入。

## 3. 冻结协议：Audio ↔ Runtime

Runtime 是 WebSocket server，默认 `0.0.0.0:8765`；Audio 是 client。

固定 PCM：16,000 Hz、mono、signed 16-bit little-endian、`pcm_s16le`、20 ms/frame、
320 samples/640 bytes。Binary frame 是裸 PCM。Text frame 为 `stream_start/stream_end` JSON：

```json
{"type":"stream_start","stream_id":"user_<id>","source":"user","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
{"type":"stream_end","stream_id":"user_<id>"}
```

Robot audio 使用同一协议且 `source="robot"`。不得单端修改格式、采样率、frame 大小、控制消息、
时序或错误语义。

## 4. 冻结协议：Runtime → Motor

同机 Unix Domain Stream Socket：`/tmp/neck_model.sock`。每次连接发送一个完整 UTF-8 JSON，
client `shutdown(SHUT_WR)`，EOF 为消息边界。当前没有应用层 ACK；EOF 不代表执行成功。

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

正式合同固定 30 fps、`radian`、`[roll,pitch,yaw]`。每帧必须是三个 finite number；states
若存在，只能是 `speaking/listening/silent` 且与 trajectory 等长。不得修改 schema、单位、
轴序、符号、Socket 或 EOF 边界。Runtime 内部 `FinalTrajectory` 不是 Motor JSON；序列化职责
属于后续 Neck/Motor adapter。

## 5. 硬件安全铁律

未经用户明确授权，不得：

- 启动 `motor/build/master_stack_test` 或任何 Motor/EtherCAT/CAN executable；
- 调用 `NeckPoseSet`、`NeckSequence` 或向真实 `/tmp/neck_model.sock` 发送数据；
- 连接 measurement socket 或执行 EtherCAT/CAN 命令；
- 连接或启动真实麦克风作为软件验证；
- 修改中心、方向、IK 系数、机械限位、电流、速度、加速度、jerk 或停止检查；
- 放宽任何 RPY、电机位置或运动安全限制来让测试通过。

Motor Socket 收到合法轨迹后可能立即驱动真实硬件。所有新 Runtime 骨架、Motion contract 和
processor 测试必须使用纯软件数据，不得探测真实 Socket 或设备。

## 6. 修改与验证规则

- 修改前检查工作树，不得覆盖他人未提交修改。
- 保持小步、局部、可测试；不迁移任务范围外的 legacy 业务。
- 不修改 `runtime/.venv/` 或任何嵌套 `.venv/` 中的第三方文件；若已被 Git 跟踪，只报告，
  不擅自删除。
- 源码编译应排除虚拟环境，例如：

```bash
python3 -m compileall -q -x '(^|/)\.venv/' runtime
```

- 新 Runtime 骨架测试使用：

```bash
python3 -m unittest discover -s runtime/tests -p 'test_runtime_core.py' -v
```

- 未经明确授权，不运行真实 Audio、Motor 或硬件集成验证。
