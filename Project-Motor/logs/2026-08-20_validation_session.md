# 实机验证会话记录（2026-08-20）

> 本文件是 Project-Motor（硬件执行层）与 Project-Neck（模型生成）联调会话的完整记录。
> 配套记录：`neck_calibration.md §6-8`（标定/验证数据）、`logs/audit_20260820/`（轨迹审计）。

## 0. 会话目标与结论

完成"颈部 RPY 轨迹执行层"（neck_control）的实机安全验证九级阶梯，并接入 Project-Neck 真实模型输出（V1 格式）。
**结论：全部通过。真实模型轨迹已在实机执行一次（25.84s 三态流程，重定时 8.80×）。**

## 1. 交付的工程成果（Project-Motor/Code）

### 1.1 执行层模块 `neck_control/`
- 输入类型/错误码、最小 JSON 解析器、**V1 模型输出映射**（`trajectory_io.cpp`：相对 rpy→绝对、
  整数 states→字符串，规则全部来自文件字段，不猜测）
- 坐标校准层（旋转矩阵、轴序/符号/固定旋转配置化）、IK 适配（复用已标定线性模型）
- S 曲线安全重定时器（jerk-limited，速度/加速度/jerk 三约束）、输入校验、状态机
- 执行器（后台控制线程 + 反馈/跟踪/超时监控）、mock/硬件适配器、遥测
- **进程级 watchdog**：控制循环心跳超时 → 直接制动 + FAULT（不依赖互斥锁）

### 1.2 CLI 命令（master_stack_test 内）
`NeckTrajDryRun / NeckTrajMock / NeckTrajRun / NeckTrajStop / NeckTrajEStop / NeckTrajAck /
NeckTrajStatus / NeckStaticCheck / NeckCalibMove / NeckCalibPose / NeckDisable`
- **全部实机运动命令为后台异步执行**：CLI 始终可响应 STOP/ESTOP（本轮最优先工程项，已完成）
- 自动反馈预热（重启后首次命令自动查询电机建立反馈）

### 1.3 工具
- `build/neck_meas2model <m1> <m2> <m3>`：实测电机角 → 硬件姿态 → 模型坐标（robot_actual_initial 用）
- `test/fixtures/mvp_stage1_hw_trajectory.json`：真实模型输出副本（V1）

### 1.4 回归测试
`cd build && ctest` → **238 项检查全部通过**（含 V1 映射、watchdog、异步可达性、真实轨迹 fixture）。

## 2. 实机验证九级阶梯（全部完成）

| 级 | 内容 | 结果/关键数据 |
| --- | --- | --- |
| 1-3 | 静态/单测、dry-run、mock | ✅ 238 项 |
| 4 | 硬件静止确认 + 急停实测 | ✅ 断动力、无位移、恢复无跳变；静止 0.001° 级、fault=0 |
| 5 | 单轴 ±2° 低速标定 | ✅ 三轴方向约定**全部验证**（含 roll 方向"假设"确认，无需改符号）；稳态跟踪 0.001° |
| 6 | 多轴低速空载 | ✅ 三轴联动无串扰，回程回差 ≤0.07° |
| 7 | 人工平滑轨迹 | ✅ 10.32s = 10.32s，峰值 ≤3.2°/s |
| 8 | 上游单段轨迹 | ✅ 20.82s，m2 18° 回中平稳，终点=中位 (10.0, 26.0, 177.0) |
| 9 | Speaking/Listening/Silent 全流程 | ✅ 28.85s，m2 达保守上限（23.9°/s、60°/s²）未超限 |
| +1 | **真实模型轨迹（38°/s 候选）** | ✅ mock 级：重定时 9.29×，峰值 ≤7.4°/s |
| +2 | **真实模型轨迹（104°/s 候选，实测起点）** | ✅ **实机执行**：重定时 8.80×，峰值 ≤19.8°/s，计划=实际 25.84s |

## 3. 实测观察记录（已写入 neck_calibration.md §6-8）

- 稳态跟踪误差 ~0.001°；S 曲线结束瞬间有 ~0.07° 收敛暂态
- yaw 轴保持下垂：首次收敛暂态 ~0.2°，随后有界（~0.01°/min 量级）
- 回差 ~0.1°（m3 往返净位移 +0.14°）
- 温度 27~29°C，保持电流 ≤0.19A，全程无故障码/无跟踪故障
- **断电重启后头部可能停在偏离上次位置数度处**（例：m1 从 10.0 → 6.71），属驱动器保持特性，静止且安全
- 保守限速（25/25/30 °/s、60/60/90 °/s²、300/300/450 °/s³）在 m2 上被逼近但从未突破

## 4. 当前状态（会话结束时）

- **硬件运行刚完成**（17:18:44 启动，17:19:09 完成：89 帧 25.84s，EXECUTING→READY）
- 预期终点：m1≈12.70、m2≈21.25、m3≈175.87（模型标定中位的硬件表示）
- **待补**：最终验证读数（NeckTrajStatus + MotorAngleGet ×3）与审计落盘
  （`/tmp/audit_mvp_hw`，由 NeckTrajStatus 触发写入）——下个会话第一件事
- 三把锁：**当前打开**（本轮实机运行需要）；收工流程：`NeckDisable` + 关锁 + 物理断电

## 5. 配置文件状态

| 项 | 值 |
| --- | --- |
| 限速（保守占位，待按阶梯放宽） | v 25/25/30 °/s；a 60/60/90 °/s²；j 300/300/450 °/s³ |
| 标定档限速 | v 10/10/15 °/s；a 30/30/45；j 150/150/225；单步 ≤5° |
| 跟踪误差/起点偏差上限 | 3° / 5° |
| 超时 | 指令/反馈 200ms、跟踪 300ms、watchdog 1000ms |
| 三把锁 | hardware_enabled / calib_confirmed / safety_confirmed（收工后应全关） |

## 6. 模型侧约定（Project-Neck 需持续遵守）

- 输出 V1 格式：`trajectory` 对象包装、`rpy` **相对 robot_actual_initial**（帧 0=[0,0,0]）、
  `states` 整数 + `states_legend`；绝对化规则 `R_abs = R(actual_initial) @ R(rpy)`
- **robot_actual_initial 必须用实测值**（`neck_meas2model` 换算），否则实机起点校验会拒绝
- Silent 终点必须精确落在 `robot_neutral_pose`（执行层强制校验）
- 候选随机性影响峰值：同一事件不同候选，峰值帧速 38°/s → 104°/s；
  **每次交付必须带 `meta.max_frame_rate_deg_per_s`**，执行层按实际值重定时
- 已补 `torch.manual_seed(seed)`——同 seed 严格可复现

## 7. 下一步（按优先级，来自操作员决策）

1. **补验证**：本次实机运行的最终读数/跟踪误差/审计（见 §4 待补项）
2. **30–60 分钟低强度连续运行**：观察温升、yaw 漂移、回差、通信稳定性
3. **限速阶梯**（每次 +30%，急停旁实测）——当前限速为保守占位，尚未在更高速度段验证
4. **A/B 视频测试**（无人近距离）→ 之后才考虑真人交互
5. 真人交互前的门槛：连续运行数据 + 限速阶梯 + 模型全链路稳定性

## 8. 关键路径速查

```
硬件项目:  /home/jhl/projects/Project-Motor/Code
模型项目:  /home/jhl/projects/Project-Neck
模型输出:  outputs/neck_motion/mvp_stage1_hw/trajectory.json（最新，104°/s 候选）
           outputs/neck_motion/mvp_stage1/trajectory.json（上一版，38°/s 候选）
模型输入:  outputs/neck_motion/mvp_stage1/events_current_pose.json（实测起点版）
执行层配置: neck_control/neck_trajectory_config.txt
标定记录:  neck_calibration.md
审计归档:  logs/audit_20260820/{smooth,segment,full}/
实机运行:  build/ 下 sudo ./master_stack_test → NeckTrajRun ../test/fixtures/mvp_stage1_hw_trajectory.json 0 <audit_dir>
```

## 9. 安全协议（每个实机会话必须重复）

1. 物理急停旁有人值守；2. 静止确认（MotorAngleGet + NeckStaticCheck）；3. 急停复测；
4. 三把锁开锁前完成离线预检；5. 运行中 CLI 可随时 STOP/ESTOP（后台模式）；6. 收工：NeckDisable + 关锁 + 物理断电。
