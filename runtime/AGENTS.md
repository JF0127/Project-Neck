# Runtime 开发与硬件安全规则

`runtime/` 负责所有真实机器人在线能力：Audio、ASR、Dialogue、TTS、motion inference、
trajectory、Motor、反馈和真机实验。它不负责训练，也不得 import `algorithm/` 训练源码。

## 边界

- `motion_model/` 是当前 V3 deployment checkpoint 的最小冻结推理实现；只为保持现有 Runtime。
- `models/` 保存本地部署资产；未来新 Algorithm 应交付独立 model package 后由 adapter 加载。
- `experiments/` 只能旁路记录，不得作为 Audio、motion 或 Motor 的运行时数据总线。
- `audio/` 和 `motor/` 保持自身现有结构与协议；目录移动不是修改业务行为的许可。

## 冻结协议

Audio ↔ Runtime：WebSocket server 默认 `0.0.0.0:8765`；16 kHz、mono、PCM s16le、
20 ms/640-byte Binary frame；Text frame 为 `stream_start/stream_end` JSON。

Runtime → Motor：Unix socket `/tmp/neck_model.sock`；完整 UTF-8 JSON，以 EOF 定界；固定
30 fps、`radian`、`[roll,pitch,yaw]`。不得修改 schema、单位、轴序、符号或 Socket。

## 安全

未经明确授权，不得：

- 启动 `motor/build/master_stack_test` 或任何 EtherCAT/CAN executable；
- 调用 `NeckPoseSet`、`NeckSequence` 或向真实 Motor Socket 发数据；
- 修改中心、方向、IK 系数、限位、电流、速度或安全检查；
- 连接真实麦克风作为软件验证的一部分。

Motor Socket 收到合法轨迹后可能立即执行。测试优先 mock、fixture、compile 和 CTest。

## 安全验证

```bash
python3 -m runtime --help
python3 -m compileall -q runtime
python3 -m unittest discover -s runtime/tests -v
cmake -S runtime/motor -B runtime/motor/build
cmake --build runtime/motor/build
ctest --test-dir runtime/motor/build --output-on-failure
```

最后三条 Motor 命令只构建/运行 CTest，不运行 `master_stack_test`。
