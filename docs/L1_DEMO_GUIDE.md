# L1 一体化演示 · 使用指南

> **状态**:2026-08-21 实机验证通过(动作幅度 ~4°、语音同步、自动回中、3s→10s 空闲回中)。
> **配套**:`docs/INTERACTION_BLUEPRINT.md`(交互蓝图)、`docs/L1_STATUS.md`(L1 状态)、
> `Project-Net/HANDOFF.md`(模型端)、`Project-Motor/neck_control/README.md`(执行层)。

---

## 1. 这是什么

一条命令(两个终端)启动的 **L1 轮转式人机对话演示**:

```
你输入文字 → 脖子倾听动作 → 机器人回复(终端文字 + TTS 语音 + 说话动作)
→ 说完 ~0.3s 收尾 → 平滑回中 → 10s 无按键自动回中
```

动作来源:v3 模型生成 30fps 轨迹 → **关键点化**(平滑/时间锚定/幅度标准化)→
执行层逐段 S 曲线平滑执行。只保留"大致趋势"(波峰波谷),不逐帧跟踪。

---

## 2. 快速开始(双终端)

### 终端 1:Motor 服务(需要 root)

```bash
cd /home/jhl/projects/Project-Neck/Project-Motor
sudo ./build/master_stack_test --server
```

- 服务日志实时显示在此终端,便于排查;
- 服务挂掉/卡住:`Ctrl+C` 重启即可;
- 启动前若报 `bind: Address already in use`,先清理残留 socket:
  ```bash
  sudo pkill -f master_stack_test; sudo rm -f /tmp/neck_ctl.sock
  ```

### 终端 2:主控(不需要 root)

```bash
cd /home/jhl/projects/Project-Neck/Project-Net
python tools/live/l1_demo.py --no-server
```

启动后自动:
1. 连接服务 → **姿态检查**:电机角偏离中位自动回中(标定档 10°/s,单步 ≤4°);
2. 进入对话循环:直接打字(**回车提交**),10s 无按键自动回中;
3. `q` 回车或 `Ctrl+C` 收工(自动 `NeckDisable` 安全制动)。

### 无硬件开发验证(mock 模式)

```bash
# 终端 1: mock 服务(无需 root/网卡)
cd /home/jhl/projects/Project-Neck/Project-Motor/build
./master_stack_test --server --no-ethercat
# 终端 2: 主控加 --mock --no-play
cd /home/jhl/projects/Project-Neck/Project-Net
python tools/live/l1_demo.py --mock --no-play --skip-dryrun --no-server
```

---

## 3. 一次交互的完整流程

```
用户回车提交文字
  → 并行 TTS: 用户语音(女声, 不播放, 仅模型输入) + 机器人回复语音(男声, 预置表随机)
  → 倾听轨迹(模型 ~0.1s + 关键点化)下发 → 脖子开始倾听(输入后 ~3s 内)
  → 说话轨迹后台生成完毕, 倾听结束后下发
  → 主控轮询执行层段边界, 进入 speaking 段起点 → aplay 机器人语音
  → 终端打印回复文字 → 说话动作随语音展开
  → 说完 ~0.3s 收尾 → 回中段(位移小, 视觉近静止)
  → 10s 无按键 → 安全停止 + 电机角步进回中
```

语音播放时机由**执行层实际段边界**触发(不依赖名义时间),保证"语音起 ↔ 动作起 ≤0.2s"。

---

## 4. 关键文件

| 文件 | 职责 |
|---|---|
| `Project-Net/tools/live/l1_demo.py` | 一体化主控(服务连接/姿态回中/输入循环/双轨迹调度/语音触发/收工) |
| `Project-Net/tools/live/keyframe.py` | 关键点化:高斯平滑 → 时间锚定采样 → 幅度标准化 → S 曲线自检 → V1 导出 |
| `Project-Net/tools/live/tts_align.py` | edge-tts 合成(20s 超时)→ whisper 词级对齐(本地模型, CPU) |
| `Project-Net/models/neck_motion/mvp.py` | v3 模型生成(已重构为可 import 的 `generate()`, CLI 兼容) |
| `Project-Motor/main.cpp` | `--server` 服务模式(Unix socket 行协议, 多线程+互斥, 段边界上报) |
| `Project-Motor/neck_control/neck_trajectory_config.txt` | **执行层配置:限速/三把锁/fps 范围** |
| `Project-Motor/neck_config.txt` | 标定参数(含中位电机角 m10/m20/m30, 回中用) |

---

## 5. 可调参数(改完重启主控即可,服务不用动)

### 5.1 动作风格(`l1_demo.py` 顶部)

| 参数 | 当前值 | 说明 |
|---|---|---|
| `KP_INTERVAL` | 1.0s | 关键点网格间隔;越小动作越紧凑,收尾误差 ≤ 该值 |
| `KP_MAX_SCALE["speaking"]` | 4.0 | 说话段幅度放大上限(方向不变) |
| `KP_MARGIN` | 0.85 | S 曲线时间预算余量(安全) |
| `KP_MOTOR_GAIN` | 6.0 | 电机角/RPY 放大系数(执行层约束在电机角空间) |
| `ENERGY_SPEAKING` | 6.0 | 模型候选能量偏好(池顶, 幅度大) |
| `IDLE_TIMEOUT` | 10.0s | 无按键超时 → 自动回中 |
| `REPLIES` | 8 条英文 | 预置回复表(接 LLM 只需换 `random.choice` 一行) |

### 5.2 执行层限速(`neck_trajectory_config.txt`,改后**需重启服务**)

| 参数 | 当前值(B 档) | 说明 |
|---|---|---|
| `safety.max_jerk_deg_s3` | 1000,1000,1500 | A 档为 300;j 越大动作越快/幅度越大,需实机验证跟踪 |
| `safety.max_velocity_deg_s` | 25,25,30 | 幅度再大可放宽(25→35),先 dry-run/mock 再实机 |
| `input.fps_min` | 1 | 关键点轨迹 fps≈1,已放宽 |
| 三把锁 | true | 实机执行必需;收工后置 false |

---

## 6. 常见问题排查(踩坑记录)

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动报 `bind: Address already in use` | 残留 socket(属主 root) | `sudo pkill -f master_stack_test; sudo rm -f /tmp/neck_ctl.sock` |
| 主控提示"未检测到 Motor 服务" | 服务没起/已死 | 终端 1 重启服务 |
| 执行被拒绝:"实机模式未使能" | 服务 cwd 不在 Project-Motor,配置没读到(三把锁默认关) | 服务必须从 `Project-Motor` 目录启动(文档命令已保证) |
| 执行被拒绝:"起点偏差超限" | 脖子不在中位 | 主控启动/空闲会自动回中;也可手动摆回 |
| 卡在输入无反应 | 没按回车(逐字符模式下提示"10s 无按键") | 输入完整句子后**回车**提交 |
| TTS 报"合成超时/失败" | 网络无法访问微软 edge-tts | 检查网络;20s 后主控自动提示,可重试 |
| whisper 加载挂起 | 传了尺寸名(如 `base`)会从 HF 下载,网络不通即挂 | 必须传本地路径 `/home/jhl/projects/Project-Neck/model`(默认已配) |
| 动作与语音不同步(话说完还在动) | 重定时比例 >1.3(执行层按电机角 S 曲线拉长) | 看 dry-run 警告;调 `KP_INTERVAL`/`KP_MARGIN`/`KP_MOTOR_GAIN`,或放宽 jerk |
| 服务卡死(所有命令超时) | 服务内某命令无限等待(残留/异常状态) | 终端 1 Ctrl+C 重启服务;必要时 `pkill` + 清理 socket |

---

## 7. 安全边界(继承执行层,不放松)

- 三把锁(`hardware_enabled`/`calib_confirmed`/`safety_confirmed`)控制实机;
- 物理急停随时可触发;watchdog 心跳超时自动制动;
- 每次改动限速参数:先 `neck_traj_dryrun --mock` 离线验证 → 实机低速试一轮(急停旁有人);
- 收工流程:`NeckDisable`(主控自动)→ 服务 Ctrl+C → 关闭三把锁 → 物理断电;
- 每次实机后确认脖子回中位再离开(主控收工会回中)。

---

## 8. 已知限制与后续路线

| 项 | 状态 | 说明 |
|---|---|---|
| 动作幅度 | ~4°(B 档 j=1000) | 再大需放宽 vmax(25→35)并实机验证 |
| 语言 | 英文 | 中文 TTS 可行(模型无条件先验),未测 |
| 回复生成 | 预置表随机 | S3 换规则/LLM |
| 内容响应(听"你好"点头) | 无 | L4,需新数据新模型(研究已冻结) |
| 流式边听边动 | 无 | L3,远期 |
| 麦克风真对话 | 无 | L2,需 VAD/ASR |
| 服务端命令超时看门狗 | 未做 | 若服务频繁卡死再加固(方案 3) |

---

## 9. 命令速查

```bash
# 实机演示(双终端)
sudo ./build/master_stack_test --server        # 终端 1(Project-Motor)
python tools/live/l1_demo.py --no-server       # 终端 2(Project-Net)

# mock 开发(双终端)
./build/master_stack_test --server --no-ethercat
python tools/live/l1_demo.py --mock --no-play --skip-dryrun --no-server

# 轨迹离线预检(免 root)
./neck_traj_dryrun <trajectory.json> /tmp/audit

# 清理残留
sudo pkill -f master_stack_test; sudo rm -f /tmp/neck_ctl.sock
```
