# L1 实现状态与实机指南(2026-08-20 晚)

> 状态:软件链路已完成并通过离线验证;实机执行待明日进行。
> 配套:实现蓝图 `docs/INTERACTION_BLUEPRINT.md`;模型端文档 `Project-Net/HANDOFF.md`;
> 执行层文档 `Project-Motor/neck_control/README.md`。

---

## 0. 一句话状态

L1 轮转式对话的**全部软件链路已打通并通过离线验证**(文本 → 语音 → 轨迹 → 执行层
dry-run/mock 全通过);**发现并量化了一个核心问题:执行层重定时会把动作放慢 ~6 倍,
导致动作与语音脱同步**——明日实机前需决策(见 §2)。

---

## 1. 已完成内容(全部新增,现有代码零改动)

### 1.1 新文件

| 文件 | 职责 | 状态 |
|---|---|---|
| `Project-Net/tools/live/tts_align.py` | 文本 → edge-tts 合成 → whisper 词级对齐 → 16k wav + 词戳 json | ✅ 测试通过 |
| `Project-Net/tools/live/live_round.py` | L1 主控:键盘输入 → 回复表 → 双方语音 → events.json → 复用 mvp.py 生成轨迹 → 同步预检 → 播放指引 | ✅ 测试通过 |
| `Project-Motor/tools/neck_traj_dryrun.cpp` | 独立离线预检工具(链接 libneckcore,无需 root):V1 解析→校验→坐标变换→IK→重定时→复验;支持 `--vmax/--amax/--jmax` 实验与 `--mock` 完整控制循环模拟 | ✅ 编译+测试通过 |
| `docs/INTERACTION_BLUEPRINT.md` | 交互蓝图(待审查) | ✅ 已写 |

### 1.2 依赖(已装好)

- `edge-tts`、`faster-whisper`(venv 已装;venv 原先无 pip,已 ensurepip)
- whisper 模型:本地 `/home/jhl/projects/Project-Neck/model/`(faster-whisper base,完整)
- 注意:GPU 转写因缺 `libcublas.so.12` 自动降级 **CPU int8**(对齐 10 词仅 0.6s,足够用)

### 1.3 验证结果

**语音链路**(`tts_align.py`,en-US 双音色):
- 用户女声 `en-US-JennyNeural` / 机器人男声 `en-US-GuyNeural`
- TTS 合成 ~2s;whisper 词对齐 ~0.6s/句;词戳时间递增、与听感吻合

**轨迹链路**(`live_round.py` → `mvp.py`,复用现有 v3 checkpoint):
- 短句 "Hi!" / 中句 / 长句 均成功生成 V1 轨迹
- 同步预检自动降级生效:峰值帧速 16.9°/s 超标 → 自动以 0.5°/s 期望能量重试 → 13.3°/s OK
- 轨迹结构核验:rpy[0]=[0,0,0]、Silent 终点=neutral=[0,0,0]、states 三态齐全

**执行层验证**(`neck_traj_dryrun`,执行层真实代码,离线):
- 3 条 L1 轨迹 dry-run 全部通过(校验/坐标/IK/重定时/复验无拒绝)
- mock 完整控制循环(模拟电机):状态 READY、错误 OK、4699 帧 0 失败、抖动 0.2ms、
  跟踪误差 0.02°、watchdog 未触发 ✅

---

## 2. 核心发现:同步问题(必读)

### 现象

L1 轨迹(名义 7.7~10.6s)经执行层重定时后,计划时长变为 **44~72s**(比例 **5.7~6.7×**)。
语音只有 ~4s,动作被放慢后必然与语音脱同步。

### 原因(结构性,已读代码确认)

执行层重定时器(`trajectory_retimer.cpp`)对**每一帧间位移**都做 7 段式 rest-to-rest
S 曲线规划,段时长 = max(S曲线时长, 名义帧间隔 33ms)。在默认安全限速
(v=25°/s, a=60°/s², j=300°/s³)下,1° 位移需要 ~0.4s(名义仅 0.033s)→ 拉长 12 倍;
帧间典型 0.05~0.1° 位移也需要 0.16~0.2s → 拉长 5~6 倍。**这是"宁可慢、不可快"的安全
设计**,但与小幅度 30fps 自然动作轨迹根本性不兼容。

### 量化数据(同一条 L1 轨迹,只改限速)

| jerk 限速(°/s³) | 重定时比例 | 说明 |
|---|---|---|
| 300(默认) | 5.75× | 动作放慢近 6 倍 |
| 1000 | 3.94× | |
| 2000 | 3.19× | |
| 5000 | 2.44× | |
| 10000 | 2.00× | 已物理上极激进 |

**结论:放宽限速只能缓解(2× 封顶),无法根治**;根治需改重定时器(选项 C)。

### 决策选项(明日实机前选一个)

| 选项 | 做法 | 效果 | 风险 |
|---|---|---|---|
| **A(推荐先试)** | 默认限速直接实机,接受"动作缓慢" | 安全第一、零改动;动作幅度 ~1° 在 10s 内缓慢完成,语音 2.6s 播完后动作余韵 12s | 观感一般,但能验证全链路 |
| **B** | 阶梯放宽限速:j 300→1000→2000,每级实机验证跟踪/急停 | 比例 3.9→3.2,动作明显加快 | 需实机确认电机跟踪能力;j≥2000 未验证 |
| **C(根治,后续任务)** | 改重定时器为**段间速度连续**规划(非逐段 rest-to-rest) | 比例趋近 1,完美同步 | 改安全核心代码,需实现+完整回归,建议独立安排 |

> 提示:选项 B 修改 `Project-Motor/neck_control/neck_trajectory_config.txt` 的
> `safety.max_jerk_deg_s3` 等;选项 A/C 不需要。改限速后务必先 `neck_traj_dryrun --mock`
> 离线验证,再实机。

---

## 3. 明日实机步骤(推荐顺序)

### 3.1 生成一轮轨迹(可选,已有 neck_l1/trajectory_1.json)

```bash
cd /home/jhl/projects/Project-Neck/Project-Net
.venv/bin/python tools/live/live_round.py --no-play
# 输入任意英文句子, 脚本自动: 合成双方语音 → 生成轨迹 → 打印 Motor 命令
```

### 3.2 离线预检(免 root)

```bash
cd /home/jhl/projects/Project-Neck/Project-Motor/build
./neck_traj_dryrun ../neck_l1/trajectory_1.json /tmp/audit_l1    # 全管线
./neck_traj_dryrun ../neck_l1/trajectory_1.json /tmp/audit_mock --mock   # 控制循环
```

### 3.3 实机(需要 root + 硬件)

```bash
cd /home/jhl/projects/Project-Neck/Project-Motor && sudo ./build/master_stack_test
# 在交互 CLI 中:
> NeckStaticCheck 5          # 静止确认(可选)
> NeckTrajDryRun ../neck_l1/trajectory_1.json
> NeckTrajRun    ../neck_l1/trajectory_1.json 0 /tmp/audit_l1_hw
```

**同步播放语音**(另一终端):
```bash
# 等 NeckTrajRun 确认动作启动后(说话段前有 ~2.5s+倾听段长缓冲):
aplay ../neck_l1/audio/bot.wav
```
> 先做一次**纯动作**(不播语音)验证一轮,再播语音,便于区分问题。

### 3.4 收工

```bash
# Motor CLI: > NeckDisable
# 关闭 neck_trajectory_config.txt 三把锁 + 物理断电
```

---

## 4. 已知问题与备忘

1. **同步问题**(§2)是唯一阻塞项;其余全部通过。
2. 每次生成时 `trajectory_<n>.json` 为最新;旧轮次保留在 `neck_l1/`。
3. `mvp.py` 每轮子进程启动 ~1.6s(含模型加载),L1 可接受;S2 再考虑常驻。
4. 语音/动作同步的缓冲:说话段前 silent 1.5s + 倾听段 + silent 1.0s,手动操作容差大;
   但受 §2 放慢影响,缓冲实际也放大了,同步以 §2 决策为准。
5. 三把锁当前为打开状态(上次实机会话遗留),收工需关闭。
6. 上次实机运行审计 `/tmp/audit_mvp_hw` 缺失,如需补记可复跑 `NeckTrajRun` 完成落盘。

---

## 5. 命令速查

```bash
# L1 一轮(交互)
cd /home/jhl/projects/Project-Neck/Project-Net
.venv/bin/python tools/live/live_round.py

# 仅生成(离线,不播放)
.venv/bin/python tools/live/live_round.py --no-play

# 预检(免 root)
cd /home/jhl/projects/Project-Neck/Project-Motor/build
./neck_traj_dryrun <trajectory.json> [audit_dir] [--jmax 1000,1000,1500 ...] [--mock]
```
