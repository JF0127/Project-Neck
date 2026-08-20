# Neck Trajectory Executor（颈部轨迹执行层）

独立、安全的“颈部轨迹执行层”，对接上游“颈部 RPY 轨迹生成项目”输出的 30 FPS RPY 轨迹。
模型输出只有经过本执行层后才能发送给硬件；禁止绕过安全层直接控制电机。

> 安全边界：**本仓库当前不自动执行任何实机运动。** 实机执行（`NeckTrajRun`）默认关闭，
> 需要 `mode.hardware_enabled` / `calib.confirmed` / `safety.confirmed` 三把锁全部显式打开，
> 且速度/加速度/jerk 参数经实机标定确认。未确认前系统明确拒绝实机执行。

## 1. 上游轨迹接口（输入）

上游输出为 30 FPS 连续轨迹 JSON（`test/upstream_sample.json` 为示例）：

```json
{
  "fps": 30,
  "coordinate_convention": {
    "unit": "radian",
    "order": ["roll", "pitch", "yaw"],
    "rotation": "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)"
  },
  "robot_actual_initial": [0.0, 0.0, 0.0],
  "robot_neutral_pose": [0.0, 0.0, 0.0],
  "trajectory": [[roll, pitch, yaw], ...],   // [N,3]，弧度
  "states": ["speaking", ...]                 // speaking/listening/silent，长度 = N
}
```

- `trajectory`：统一坐标系目标绝对 RPY（弧度，`[roll,pitch,yaw]`）；
- **模型输出 V1 格式**（`format_version=1.0`，`trajectory` 为对象包装、`rpy` 相对 `robot_actual_initial`、
  `states` 整数编码 + `states_legend`）：由 `loadNeckTrajectoryJson` 自动识别并显式映射
  （`R_abs = R(robot_actual_initial) @ R(rpy)`，与模型侧交付校验一致；映射规则见
  `trajectory_io.h`，全部来自文件自身字段，不猜测）。映射后仍走标准校验。
- `states`：行为状态（≠ 硬件安全状态），Silent 段终点必须等于 `robot_neutral_pose`；
- 可选防重放字段 `timestamp` / `command_id`（提供时强制单调递增，重复/乱序/过期一律拒绝）。

## 2. 坐标系与旋转约定

| 层 | 约定 |
| --- | --- |
| 模型（上游） | `R_model = Ry(yaw) @ Rx(pitch) @ Rz(roll)`，分量顺序 `[roll,pitch,yaw]`，弧度 |
| 硬件（本机） | `R_hw = Rz(yaw) @ Rx(pitch) @ Ry(roll)`，+X 右 / +Y 前 / +Z 上，度 |
| 映射 | `R_hw = R_calibration @ R_model`，再按 `hardware_axis_order` 命名提取欧拉角并施加 `axis_sign` 与 `rpy_offset_deg` |

大角度组合一律走旋转矩阵（`neck_control/coordinate_calibration.*`），**不做逐轴简单相加**。
约定本身已包含轴交换：模型 roll(绕Z) → 硬件 yaw(绕Z)，模型 yaw(绕Y) → 硬件 roll(绕Y)，
模型 pitch(绕X) → 硬件 pitch(绕X)。机构方向/安装差异通过 `axis_sign`、`rotation_matrix` 配置修正，
不允许在业务代码里散落正负号。

## 3. 硬件关节映射（复用已标定模型）

2 自由度并联倾摆 + 串联偏航（见 `neck_model.md`、`neck_calibration.md`）：

```text
电机1(通道1/ID1) + 电机2(通道2/ID2) --曲柄连杆并联--> pitch / roll（2x2 耦合矩阵 C）
电机3(通道3/ID3)                     --1:1 直驱--> yaw（k3=1）
```

逆解（`neck_kinematics.h`，参数在 `neck_config.txt`，已标定 2026-08-15）：
`m = m0 + C⁻¹·Δq`，`m3 = m30 + Δyaw/k3`。执行层只通过 `neckSolve` 访问，任何奇异/超限判定
`NECK_SINGULAR / NECK_POSE_OUT_OF_RANGE / NECK_MOTOR_OUT_OF_RANGE` 都会拒绝整段轨迹。

## 4. 管线

```text
上游 RPY JSON
  → 输入校验（N>0、[N,3]、有限值、fps∈[5,120]、约定匹配、states 长度、防重放、Silent 终点）
  → 坐标系转换（旋转矩阵，配置化轴序/符号/固定旋转/偏置）
  → RPY→关节 IK（neckSolve：姿态限位 + 电机限位）
  → 起点衔接检查（实际反馈起点偏差 ≤ max_start_pose_error_deg，否则禁止跳转）
  → 安全重定时（每段 jerk-limited S 曲线，速度/加速度/jerk 全部受限于配置）
  → 处理后重新验证（角度/速度/加速度/jerk 数值微分复检）
  → 时间驱动控制循环（单调时钟 + sleep_until，100 Hz，记录 jitter）
  → 硬件指令（位置伺服 + 每帧电机速度上限）
  → 反馈监控（反馈超时/跟踪误差/姿态超限/NaN/电机故障码/循环超时）
```

## 5. 安全参数（`neck_control/neck_trajectory_config.txt`）

| 参数 | 默认（保守占位，待标定） |
| --- | --- |
| `safety.max_velocity_deg_s` | `25, 25, 30`（电机输出轴，度/秒） |
| `safety.max_acceleration_deg_s2` | `60, 60, 90` |
| `safety.max_jerk_deg_s3` | `300, 300, 450` |
| `safety.rpy_limits_deg` | pitch `[-40,25]` roll `[-35,35]` yaw `[-117,58]` |
| `safety.joint_limits_deg` | m1 `[-85,64]` m2 `[-34,113]` m3 `[60,235]` |
| `max_start_pose_error_deg` / `max_tracking_error_deg` | `5` / `3` |
| `command_timeout_ms` / `feedback_timeout_ms` / `tracking_error_timeout_ms` | `200` / `200` / `300` |
| `return_to_neutral_on_timeout` | `false`（故障/超时默认不自动回中，避免二次危险） |
| `stop_on_timeout` | `true` |

**不要直接把上游观测到的 68°/s 当允许速度**；上述速度/加速度/jerk 均为保守占位，
实机前必须按 §9 标定确认并置 `safety.confirmed = true`。

超限处理：超角度 → 拒绝整段；超速度/加速度/jerk → **安全重定时**（S 曲线时间参数化），
不做逐帧截断，处理后重新验证。原始与处理后轨迹均可审计落盘（`saveAudit`）。

## 6. 状态机

```text
DISABLED → READY → EXECUTING → STOPPING → READY
                  ↓            ↓
            CALIBRATION   FAULT / ESTOP
```

- 未标定（`calib.confirmed=false`）不能进入 `EXECUTING`；
- `READY` 只需 `mode.hardware_enabled=true`（显式使能）——这是为**标定运动**留的门：标定是获得 `calib.confirmed` 的前提，不能在使能阶段就要求三把锁；
- `EXECUTING`（真实轨迹执行）必须三把锁全开（`hardware_enabled` + `calib_confirmed` + `safety_confirmed`）；
- `FAULT` / `ESTOP` 不能被普通轨迹命令（start/stop/pause）解除，只能 `NeckTrajAck` 显式确认；
- **进程级 watchdog**：独立线程监视控制循环心跳（`loop.watchdog_timeout_ms`，默认 1000ms）；运动中心跳超时 → 直接发布制动帧 + `FAULT(LOOP_OVERRUN)`，不依赖互斥锁（卡死也能制动）；
- 急停可从任何状态触发；`Speaking/Listening/Silent` 是上游行为状态，不等同于硬件安全状态；
- 所有迁移打日志（前缀 `[NeckState]`）。

## 7. 运行模式

| 模式 | 命令 | 行为 |
| --- | --- | --- |
| dry-run | `NeckTrajDryRun <json> [auditDir]` | 不连接硬件；完整执行解析/转换/IK/限幅/重定时/复验，输出处理后轨迹与审计 JSON |
| mock | `NeckTrajMock <json> [auditDir]` | MockAdapter 模拟执行器与反馈；验证控制循环、跟踪误差、超时、停止与急停 |
| 标定 | `NeckCalibMove <SlaveId> <Axis1..3> <DeltaDeg>` | 单轴低速小幅度运动：标定档限速（默认 10°/s）、单步幅度 ≤ `calib.max_step_deg`（5°）、关节限位检查、反馈/跟踪监控、急停随时可触发；全程在 `CALIBRATION` 状态（非 `EXECUTING`），完成后回 `READY`（后台执行） |
| 收工 | `NeckDisable [SlaveId]` | 停止轨迹 + 制动保持（软件级禁用）；离开前仍需物理断电/急停 |
| hardware | `NeckTrajRun <json> [SlaveId] [auditDir]` | 默认关闭；三把锁全开 + 标定完成后才允许，否则明确拒绝 |

```bash
mkdir build && cd build && cmake .. && make
ctest                          # 全部自动测试（dry-run + mock，无需硬件/root）
./master_stack_test            # 无硬件时电机命令不可用，但 dry-run/mock 可用
> NeckTrajDryRun ../test/upstream_sample.json /tmp/audit
> NeckTrajMock  ../test/upstream_sample.json /tmp/audit
> NeckTrajEStop / NeckTrajAck / NeckTrajStatus
```

## 8. 急停与故障恢复

- `NeckTrajEStop`：任何状态 → `ESTOP`；硬件侧持续发布全制动帧直到 `NeckTrajAck`；
- 故障（跟踪误差持续超限、实际姿态超限、反馈 NaN、电机故障码、循环超时）→ `FAULT`；
- 故障/急停恢复：`NeckTrajAck` 显式确认（实机还需适配器健康）→ `READY`；
- 超时（指令/反馈）默认 → 安全停止（保持位置自然减速）→ `READY`，**不自动回中**。

## 9. 标定流程（实机步骤，本仓库不自动执行）

1. **确认设备可安全停止**：物理急停旁有人值守；急停实测（按下断动力、恢复无跳变）；三轴静止、反馈新鲜、无故障码（`NeckStaticCheck`）；
2. 用 `NeckCalibMove` 逐轴做 ±2° 低速标定运动（标定档限速，全程 `CALIBRATION` 状态、急停保护），目视确认方向是否符合约定，必要时改 `calib.axis_sign`；
3. 用现有 CLI（`MotorPositionSet` / `MotorAngleGet`）+ 量角器按 `neck_model.md §8` 标定 `neck_config.txt` 的 C/k3/m0（已有 2026-08-15 记录，roll 方向仍为假设）；
4. 实测并填写 `safety.max_velocity/acceleration/jerk`（从保守值逐步放宽）；
5. 实测急停、通信断线、反馈丢失行为；
6. 置 `calib.confirmed = true`、`safety.confirmed = true`（`mode.hardware_enabled` 已在标定阶段打开）；
7. 按 §10 验证顺序逐级上电验证。

## 10. 验证顺序（未确认不跳级）

```text
静态检查与单元测试（ctest）
→ dry-run（NeckTrajDryRun）
→ mock（NeckTrajMock，含超时/急停/跟踪误差注入）
→ 硬件通信但电机不使能
→ 单轴 ±2° 低速标定
→ 多轴低速空载
→ 人工生成的平滑轨迹
→ 上游模型单段轨迹
→ Speaking/Listening/Silent 完整流程
```

## 11. 当前已验证范围与未验证风险

**已验证（本仓库自动测试 + dry-run/mock CLI + 2026-08-20 实机阶梯验证）：**
- **实机（九级阶梯 + 真实模型输出全部完成）**：2026-08-20 会话记录见 `logs/2026-08-20_validation_session.md`；真实模型轨迹（V1 格式，104°/s 峰值候选）已在实机完整执行 25.84s 三态流程（重定时 8.80×，峰值 ≤19.8°/s，计划=实际）。静止确认与急停实测（断动力、无跳变）、单轴 ±2° 低速标定（三轴方向约定全部验证，含此前为"假设"的 roll 方向）、多轴低速空载、人工平滑轨迹（10.32s）、上游单段轨迹（20.82s，回到中位）、Speaking/Listening/Silent 完整流程（28.85s）；全程无故障码、无跟踪故障，m2 速度/加速度在短语段达到保守上限（23.9°/s、60°/s²）未超限；温度 27~28°C；实机记录见 `neck_calibration.md §6-8`。
- 输入校验（NaN/Inf、形状、fps、约定、states、防重放、Silent 终点）；
- 坐标映射（轴交换、符号、固定旋转、旋转矩阵往返一致性）；
- S 曲线重定时（速度/加速度/jerk 约束成立，处理后复验通过）；
- 控制循环时序（jitter < 1ms 实测）、指令/反馈超时安全停止、跟踪误差 FAULT、
  急停任意状态触发且不可被普通命令解除、mock 全流程 Speaking→Listening→Silent、
  dry-run 不接触硬件。

**未验证（实机风险，需按 §10 逐级验证）：**
- 电机速度参数（spd 0~18000）与实际输出轴速度的换算关系（10:1 为文档值，待实机确认）；
- 电机内部加速度曲线与命令 S 曲线叠加后的真实运动学；
- 编码器反馈延迟/量化对跟踪误差判定的影响；
- 急停制动（`set_motor_cur_tor` ctrl_status=2）实机效果；
- roll 正方向假设（`neck_calibration.md §5`）；
- 线性 C 矩阵在大角度下的残差（~7°@极限）。

**实机驱动资格声明：** 在完成 §9 标定、§10 验证顺序且三把锁全部显式打开之前，
本系统不具备实机驱动资格；默认配置下 `NeckTrajRun` 必然被拒绝。

## 12. 文件结构

```text
neck_control/
  command_types.h        统一输入类型与错误码
  json_parser.*          最小 JSON 解析器
  trajectory_io.*        JSON ↔ 命令；审计 JSON 输出
  config.*               配置加载（key=value，与 neck_config.txt 风格一致）
  coordinate_calibration.* 坐标系/标定映射（旋转矩阵）
  inverse_kinematics.*   RPY→关节适配（复用 neck_kinematics.h）
  safety_limiter.*       限位检查工具
  trajectory_retimer.*   jerk-limited S 曲线重定时
  trajectory_validation.* 输入校验
  state_machine.*        硬件安全状态机
  telemetry.*            时序/故障遥测与日志
  adapter_interface.h    硬件适配器抽象
  mock_adapter.*         mock 执行器（含故障注入）
  hardware_adapter.*     EtherCAT 实机适配器（仅本文件依赖 SOEM）
  executor.*             执行层核心（管线 + 控制循环 + 监控）
  neck_trajectory_config.txt  安全/标定配置（默认三把锁关闭）
test/
  neck_control_tests.cpp 自动测试（ctest）
  upstream_sample.json   上游示例轨迹
```

构建：`neckcore`（纯 C++，测试/执行层共用）与 `motor`（+SOEM）分离；
`neck_control_tests` 不链接 SOEM，无需网卡/root。
