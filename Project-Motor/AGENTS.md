# Project Motor 新版颈部控制重构指导文档

> 目的：明确新版颈部控制系统的功能边界、保留资产、目录结构、配置方式和实现原则，作为后续重新编写上层代码时的唯一指导文档。
>
> 本次重构的核心不是继续修补旧 `neck_control/`，而是在保留现有项目核心功能和已验证物理参数的基础上，重新实现一个更简单、清晰、方便调试和方便模型集成的颈部上层控制系统。

---

# 1. 项目核心作用

本项目的核心作用是：

**接收上游模型或人工测试产生的颈部姿态轨迹，将其转换成三台颈部电机的目标位置，并通过现有 EtherCAT/CAN 电机接口发送到真实硬件。**

核心链路：

```text
上游模型 / 人工 Demo
        ↓
RPY Trajectory
        ↓
颈部坐标与运动学转换
        ↓
Motor1 / Motor2 / Motor3 目标角
        ↓
基础位置与速度检查
        ↓
现有电机发送接口
        ↓
EtherCAT / CAN
        ↓
三台颈部电机
```

新版必须保留两个主要上层入口：

1. 手动测试入口
2. 模型输入入口

两者最终必须共用同一套颈部运动执行逻辑。

---

# 2. 新版设计原则

新版重构遵循以下原则。

## 2.1 简单优先

不要重新建立旧版本中复杂的：

- 七状态状态机
- confirmed 三把锁
- FAULT / ACK 恢复机制
- 黏性 ESTOP 状态
- command_id 防重放
- timestamp 新鲜度限制
- pause / resume
- 自动超时回中
- 复杂 watchdog 状态体系
- 大量安全 Gate
- 复杂轨迹审计体系

除非后续真实需求证明必须存在，否则不要提前实现。

## 2.2 一个动作执行核心

人工 Demo 和模型输入只是两个不同的轨迹来源。

不要建立：

```text
Demo 控制系统
+
Model 控制系统
```

两套独立实现。

应该是：

```text
Demo JSON ─────┐
               │
               ↓
          Trajectory
               ↓
          Neck Motion
               ↓
             IK
               ↓
           Motors
               ↑
               │
Model Input ────┘
```

## 2.3 机械参数与动作数据分离

机器人自身的机械参数进入：

```text
neck/neck_config.txt
```

人工 Demo 动作进入：

```text
trajectories/*.json
```

不要把 Demo 轨迹硬编码到 C++。

不要把机械限位、中位、速度等参数写进 Demo JSON。

---

# 3. 新版建议目录

第一版只建立以下结构：

```text
project/
│
├── neck/
│   ├── neck_config.txt
│   │
│   ├── neck_config.h
│   ├── neck_config.cpp
│   │
│   ├── neck_kinematics.h
│   ├── neck_kinematics.cpp
│   │
│   ├── trajectory.h
│   ├── trajectory.cpp
│   │
│   ├── trajectory_io.h
│   ├── trajectory_io.cpp
│   │
│   ├── neck_motion.h
│   └── neck_motion.cpp
│
├── trajectories/
│   ├── standard_test.json
│   ├── calm_speaking.json
│   ├── active_speaking.json
│   ├── emphasis_speaking.json
│   └── ...
│
├── main.cpp
├── command.cpp
│
└── 现有厂家 EtherCAT / CAN / Motor 相关代码
```

暂时不要增加：

```text
state_machine/
safety/
executor/
watchdog/
calibration/
demo/
```

等以后真的产生需求再增加。

---

# 4. 各模块职责

## 4.1 neck_config

负责读取：

```text
neck/neck_config.txt
```

只负责：

- 硬件映射
- 电机机械参数
- 颈部姿态范围
- 运动学标定
- 人工可调运行参数
- 网络模型接口基础参数

不要负责：

- Demo 轨迹
- 状态机
- 行为逻辑
- speaking / listening 等动作策略

---

# 5. neck_config.txt

建议结构如下。

```ini
# ==============================
# Hardware
# ==============================

network_interface = enp4s0
slave_id = 0

ack_status = 2


# ==============================
# Motor 1
# ==============================

motor1.passage = 1
motor1.id = 1

motor1.min_position_deg = -85
motor1.max_position_deg = 64
motor1.center_position_deg = 10

motor1.max_velocity_deg_s = 25

motor1.speed_param = 50
motor1.current_param = 500


# ==============================
# Motor 2
# ==============================

motor2.passage = 2
motor2.id = 2

motor2.min_position_deg = -34
motor2.max_position_deg = 113
motor2.center_position_deg = 26

motor2.max_velocity_deg_s = 25

motor2.speed_param = 50
motor2.current_param = 500


# ==============================
# Motor 3
# ==============================

motor3.passage = 3
motor3.id = 3

motor3.min_position_deg = 60
motor3.max_position_deg = 235
motor3.center_position_deg = 177

motor3.max_velocity_deg_s = 30

motor3.speed_param = 50
motor3.current_param = 500


# ==============================
# Neck RPY Range
# ==============================

pitch_min_deg = -40
pitch_max_deg = 25

roll_min_deg = -35
roll_max_deg = 35

yaw_min_deg = -117
yaw_max_deg = 58


# ==============================
# Kinematics
# ==============================

c11 = 0.2170
c12 = -0.2413

c21 = 0.3218
c22 = 0.2963

k3 = 1.0

pitch_center_deg = 0
roll_center_deg = 0
yaw_center_deg = 0

det_eps = 0.000001


# ==============================
# Model Input
# ==============================

model.fps = 30

model.input_unit = radian

model.rpy_order = roll,pitch,yaw
```

---

# 6. 关于 startup position

新版不人为定义固定：

```text
startup_position
```

原因是：

电机上电时实际停在哪里，应通过真实反馈读取。

因此：

```text
启动
→ MotorAngleGet / feedback
→ 得到三个电机当前真实位置
```

真正需要固定配置的是：

```text
center_position
min_position
max_position
```

不要把：

- 上电位置
- 机械最小位置
- 人工中位

三者混在一起。

---

# 7. current 参数

厂家位置命令中的：

```text
current
```

是电流限制 / 电流阈值协议参数。

当前：

```text
current = 500
```

按照厂家已有代码注释，大约对应：

```text
50 A
```

但它不是“电机运行时始终输出 50A”。

它更接近位置控制模式允许的电流限制。

新版中不要硬编码：

```cpp
current = 500;
```

而是分别使用：

```text
motor1.current_param
motor2.current_param
motor3.current_param
```

当前暂时保持 500，后续根据真实电机参数人工调整。

---

# 8. 三电机运动学

当前人工定义中位：

```text
motor1 = 10°
motor2 = 26°
motor3 = 177°
```

对应：

```text
pitch = 0°
roll  = 0°
yaw   = 0°
```

定义：

```text
Δm1 = m1 - m10
Δm2 = m2 - m20
Δm3 = m3 - m30
```

正运动学：

```text
pitch = p0 + c11 * Δm1 + c12 * Δm2

roll  = r0 + c21 * Δm1 + c22 * Δm2

yaw   = y0 + k3 * Δm3
```

逆运动学：

```text
Δp = pitch - p0
Δr = roll  - r0
Δy = yaw   - y0

det = c11*c22 - c12*c21
```

```text
m1 = m10 + ( c22*Δp - c12*Δr ) / det

m2 = m20 + (-c21*Δp + c11*Δr ) / det

m3 = m30 + Δy / k3
```

必须保持当前：

```text
c11 =  0.2170
c12 = -0.2413
c21 =  0.3218
c22 =  0.2963
k3  =  1.0
```

本次重构不处理大角度约 7° 的线性模型残差。

这是后续独立的运动学标定问题。

---

# 9. 坐标和方向

硬件方向保持现有定义：

```text
pitch > 0 ：抬头
pitch < 0 ：低头

roll > 0 ：向右侧头
roll < 0 ：向左侧头

yaw > 0 ：向左转头
yaw < 0 ：向右转头
```

模型输入单位：

```text
radian
```

颈部内部运动学建议统一使用：

```text
degree
```

模型输入必须经过明确的单位转换。

组合旋转仍应使用旋转矩阵处理，不要直接把不同欧拉角约定逐轴相加。

---

# 10. 统一轨迹数据结构

新版内部只保留一种轨迹结构。

无论输入来自：

```text
Demo JSON
```

还是：

```text
网络模型
```

都统一转换成：

```text
Trajectory
```

建议基本结构：

```cpp
struct TrajectoryPoint {
    double pitch_deg;
    double roll_deg;
    double yaw_deg;
};

struct Trajectory {
    double fps;
    std::vector<TrajectoryPoint> points;
};
```

如果需要保留行为标签，可增加：

```cpp
enum class BehaviorState {
    Silent,
    Speaking,
    Listening
};
```

但这些标签：

```text
只作为 metadata
```

不直接参与电机运动控制。

---

# 11. 模型接口

新版只保留一种正式模型 JSON 格式。

建议正式统一成：

```json
{
  "fps": 30,
  "unit": "radian",
  "order": ["roll", "pitch", "yaw"],
  "trajectory": [
    [0.0, 0.0, 0.0],
    [0.01, -0.02, 0.03]
  ],
  "states": [
    "speaking",
    "speaking"
  ]
}
```

其中：

```text
fps
```

为轨迹采样率。

```text
trajectory[i]
```

为：

```text
[roll, pitch, yaw]
```

单位：

```text
radian
```

`states` 为可选字段，仅作为行为标签。

新版不再同时维护 Standard JSON 和 V1 JSON 两套正式协议。

旧格式如确实需要，可后续单独写转换工具。

---

# 12. Demo 轨迹

所有人工 Demo 放在：

```text
trajectories/
```

例如：

```text
trajectories/
├── standard_test.json
├── calm_speaking.json
├── active_speaking.json
└── emphasis_speaking.json
```

Demo JSON 与模型输入尽量使用同一种轨迹格式。

例如：

```json
{
  "name": "calm_speaking",
  "fps": 30,
  "unit": "degree",
  "order": ["pitch", "roll", "yaw"],
  "trajectory": [
    [0, 0, 0],
    [-1, -2, 7],
    [-5, -1, 8],
    [2, 0, 7],
    [0, 0, 0]
  ]
}
```

如果希望模型和 Demo 完全统一，也可以统一规定所有 JSON 都使用：

```text
radian
+
[roll,pitch,yaw]
```

由 `trajectory_io` 统一转换成内部 degree RPY。

这是更推荐的长期方案。

---

# 13. NeckSequence

旧命令：

```text
NeckSquence
```

新版不再兼容。

统一使用正确命名：

```text
NeckSequence
```

例如：

```text
NeckSequence standard_test
NeckSequence calm_speaking
NeckSequence active_speaking
NeckSequence emphasis_speaking
```

相比：

```text
NeckSequence 0
NeckSequence 1
```

更推荐直接使用文件名或动作名。

这样以后新增：

```text
thinking.json
listening.json
nod.json
look_left.json
```

无需修改 C++ 中的 Sequence ID 表。

例如：

```text
NeckSequence calm_speaking
```

实际加载：

```text
trajectories/calm_speaking.json
```

---

# 14. 手动 CLI

新版保留以下核心命令：

```text
MotorIdGet <SlaveId>

MotorIdSet <SlaveId> <MotorId> <NewMotorId>

MotorIdReset <SlaveId>

MotorAngleGet <SlaveId> <PassAge> <MotorId>

MotorZeroSet <SlaveId> <PassAge> <MotorId>

MotorStop <SlaveId> <PassAge> <MotorId>

MotorSpeedSet <SlaveId> <PassAge> <MotorId> <Speed> <Current> <AckStatus>

MotorPositionSet <SlaveId> <PassAge> <MotorId> <Position> <Speed> <Current> <AckStatus>

NeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>

NeckSequence <TrajectoryName>
```

其他上层 CLI 默认不保留。

---

# 15. NeckPoseSet

`NeckPoseSet` 是最基本的颈部功能测试接口。

流程：

```text
Pitch / Roll / Yaw
        ↓
基础合法性检查
        ↓
neckInverse()
        ↓
Motor1 / Motor2 / Motor3
        ↓
位置范围检查
        ↓
发送三个电机目标
```

输入：

```text
degree
```

它不需要复杂状态机。

---

# 16. NeckSequence

流程：

```text
Trajectory JSON
        ↓
trajectory_io
        ↓
Trajectory
        ↓
逐帧 RPY
        ↓
neckInverse()
        ↓
Motor targets
        ↓
基础速度 / 位置检查
        ↓
三电机一帧发布
```

Demo 和模型轨迹应该最终进入相同执行函数。

---

# 17. 电机发送

单电机测试继续使用现有厂家命令：

```text
set_motor_position()
set_motor_speed()
set_motor_cur_tor()
get_motor_parameter()
```

以及已有队列发送方式。

连续颈部轨迹优先使用：

```text
三个电机同时构造完整 EtherCAT_Msg
        ↓
NeckFramePublish()
```

而不是三个电机每一帧分别进入普通命令队列。

这样能够保证三个电机目标来自同一个控制时刻。

---

# 18. 必须保留的基础约束

新版只保留必要的基础约束。

## 必须保留

### 输入合法性

禁止：

```text
NaN
Inf
非法数组
非法单位
非法 MotorId / Passage
```

### RPY 范围

目标姿态不能超过：

```text
pitch_min/max
roll_min/max
yaw_min/max
```

### 电机位置范围

IK 输出必须满足：

```text
motor_min <= target <= motor_max
```

### 最大速度

连续轨迹必须保证真实目标变化不超过配置的：

```text
motor1.max_velocity_deg_s
motor2.max_velocity_deg_s
motor3.max_velocity_deg_s
```

同时厂家：

```text
speed_param
```

必须位于厂家协议允许范围。

注意：

```text
speed_param != degree/s
```

两者不能混用。

### 停止

必须保留：

```text
MotorStop
```

并且颈部连续轨迹需要有一个能够停止三个电机运动的入口。

---

# 19. 不迁移的旧机制

新版默认不迁移：

```text
state_machine
DISABLED
READY
EXECUTING
STOPPING
CALIBRATION
FAULT
ESTOP sticky state

hardware_enabled
calib.confirmed
safety.confirmed

NeckTrajAck
acknowledge()

pause()
resume()
cancel()

command_id anti-replay

timestamp freshness

silent 必须回中

timeout 自动回中

复杂 watchdog / heartbeat 状态体系

stop_on_timeout 多策略

复杂 S-curve / jerk safety framework

大量 telemetry / audit 状态
```

以后如果实际运行证明某项确实需要，再单独增加。

不要为了“可能以后用到”提前恢复旧架构。

---

# 20. 当前人工可调参数

新版的一个重要目标是：

**不重新编译代码，也能方便调机器人。**

因此以下内容全部配置化：

```text
network_interface
slave_id

三个 motor id
三个 passage

三个 motor min/max
三个 motor center

三个 max velocity

三个 speed_param
三个 current_param

ack_status

pitch/roll/yaw 范围

运动学 C 矩阵
k3

坐标转换参数
```

人工 Demo 动作则直接修改：

```text
trajectories/*.json
```

---

# 21. 新版最终核心架构

最终目标保持非常简单：

```text
                       ┌─────────────────┐
                       │   Model Input   │
                       └────────┬────────┘
                                │
                                │ JSON / RPY
                                ↓

Demo JSON ───────→   Trajectory

                                ↓

                         Neck Motion

                                ↓

                      Neck Kinematics

                                ↓

                     Motor1 / 2 / 3

                                ↓

                    Existing Motor API

                                ↓

                      EtherCAT / CAN
```

其中：

```text
neck_config.txt
```

为所有机械和运行参数提供唯一配置来源。

---

# 22. 本次重构不处理的问题

以下问题不属于本次代码重构：

## 线性运动学大角度残差

当前模型在大角度可能出现约：

```text
7°
```

残差。

本次保持当前运动学公式和标定参数。

以后如果需要，提高精度应单独进行：

```text
非线性标定
或
分段模型
或
查表 / 拟合
```

不要在本次代码重写过程中猜测修改。

---

# 23. 新版开发顺序

推荐严格按以下顺序实现。

## Phase 1

实现：

```text
neck_config
```

确保配置能够正确加载。

## Phase 2

实现：

```text
neck_kinematics
```

并使用现有实测数据验证 IK/FK。

## Phase 3

实现：

```text
NeckPoseSet
```

让单姿态控制先跑通。

## Phase 4

实现：

```text
Trajectory
trajectory_io
```

能够读取本地 JSON。

## Phase 5

实现：

```text
NeckSequence
```

能够执行：

```text
trajectories/*.json
```

## Phase 6

加入：

```text
最大速度检查
基础连续轨迹执行
三电机完整帧发布
```

## Phase 7

最后再接入：

```text
网络模型输入
```

模型输入只需要转换成相同的：

```text
Trajectory
```

后续执行完全共用。

---

# 24. 最终目标

新版代码应该做到：

```text
代码少
结构清楚
参数集中
轨迹外置
方便 Demo
方便调参
方便模型接入
不修改已验证的物理含义
不重新建立没有实际需求的复杂架构
```

判断一次重构是否成功的标准不是“功能数量更多”，而是：

> **一个新的 Codex / GPT 只需要阅读这份文档和少量核心代码，就能快速理解模型输入如何最终变成三个电机命令。**