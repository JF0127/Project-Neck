# Project-Neck 交接文档（HANDOFF）

> 用途：项目状态冻结交接。新会话/新模型接手时先读本文件，再按需读
> `models/neck_motion/README.md`（完整文档入口）与各脚本头注释。
> 最后更新：MVP 阶段 1 离线接入（实机联调进行中）。

---

## 0. 一句话状态

第一版"话语级机器人颈部运动生成"已完成 **研究冻结 + 软件推理/调度链路闭环**：
v3 多候选模型作为**角色动作先验**（speaker/listener），配三态 MVP 调度
（Speaking/Listening/Silent），已产出供实机校验层测试的轨迹 JSON。
**尚未获得实机直接驱动资格**（执行层限速/标定/急停未完成）。

---

## 1. 数据（只读，严禁修改/重建管线）

- 根目录：`data/source/response-net/processed_dataset/`
- 规范：`FIELDS.md`（**必读**：字段、relative RPY 语义、旋转约定
  `R = Ry(yaw)@Rx(pitch)@Rz(roll)`、fragment 独立性）
- 文件：train/val/test.jsonl（每条 fragment 展开 speaker+listener 两条样本）、
  aligned_utterances.jsonl（fragment 级，含词时间戳与 overlap 统计）、
  audio/*.wav（原始采样率，44.1kHz 为主）、poses/*_{speaker,listener}_rpy.npy
  （**绝对** RPY，审计用）、*_relative_rpy.npy（相对，训练监督）、*_delta_rpy.npy
- 切分：按 dialogue_id（80/10/10，seed 42），**严禁重新划分**
- 过滤（第一版修正）：按 `true_overlap`（实际词区间交集，不含窗口 padding，
  `overlap_filter.py`）> 0.1s 剔除，按 fragment 过滤。结果：
  窗口 overlap 过滤 694 个（14.8%）→ true_overlap 过滤 **275 个（5.9%）**，
  419 个原误删恢复、0 新增。过滤后样本：**train 7154 / val 754 / test 906**；
  映射文件 `outputs/neck_motion/overlap_analysis/true_overlap.json`（缺失时
  train.py 自动生成）
- 词表：只从 train 构建，`outputs/neck_motion/vocab.json`（5798 词，含
  `<PAD><UNK><SILENCE><START_OF_DIALOGUE>`）

---

## 2. 模型研究历程（全部结论已冻结）

| 版本 | 方案 | 5-epoch 验证结果 | 结论 |
|---|---|---|---|
| v1 | 确定性回归（共享编码器+双头，直接预测 relative RPY） | val MAE 2.82°，**不如全零基线 2.783°**（corr≈0.04、幅度 0.36 vs 2.78） | 回归把多峰动作平均为零（均值坍缩）；损失尺度不是主因（分项 β/L1 后无变化） |
| v2 | 片段级条件 CVAE（z_dim=16，后验+先验，KL warm-up+free-bits） | 先验采样多样性 0.02°、幅度 0.17°、best-of-10=2.789° | **posterior collapse 与 KL 参数无关**（β_kl=0 对照曲线逐位相同）；重构目标驱动解码器忽略 z |
| **v3** | **条件多候选轨迹模型（K=8，训练=部署先验采样）**：`min_k L_recon + λ_div·有界多样性 + λ_smooth·acc hinge + λ_amp·幅度 hinge` | **best-of-8 2.50°、多样性 1.92°、幅度 1.40°（目标 2.78）、acc 52°/s²**；best-of-K 单调 3.26/2.69/2.50/2.36(K=16) | 摆脱零轨迹坍缩；但经错配/节奏/探针诊断证明为**无条件角色动作先验** |
| v3.5 | 条件建模诊断（role-only 对比、音频/文本错配、节奏对齐） | role-only 2.462° ≈ 完整 2.472°；错配无差异；目标-音频节奏相关仅 +0.023 | 候选不依赖输入条件；数据中节奏信号极弱 |
| v4 | 条件预测信号探针（低维标签：nod/shake/方向/能量） | speaker 同步任务：全部条件增量 ≤0；listener 后反应 nod +0.04 | 任务错位假说（Listener 标签语义错误） |
| v4.1 | Listener 后反应标签重建（固定 0.3s 窗口） | nod +0.04（5 seeds） | 首个一致正增量，但未做抽样不确定性检验 |
| v4.2 | 原始时间轴审计 + 可变长反应窗口 + **dialogue block bootstrap** | 空隙中位数 0.15s；可用样本 train 468/val 66；**full−role_only 差值 95% CI [−0.072, +0.089] 含 0** | **数据不足，+0.04 无法确认**；不上高维特征/扩散 |

**研究冻结结论（证据链完整）**：
- 音频/文本/对方姿态在当前样本定义下无稳定条件增量；
- Listener 后反应窗口在原数据中结构性不足（管线只后垫 0.5s，78% 样本 0.5s 内有人接话）；
- 未来若重启内容条件模型：需专门采集"话语后反应"数据（更长姿态覆盖），
  并先过 `condition_probe.py` 的 bootstrap 检验。

---

## 3. 交付物

### 3.1 模型（冻结）
- **v3 角色动作先验**：`outputs/neck_motion_v3/checkpoints/best.pt`（多候选 K=8，
  z_dim=16，model.type=candidates，5.25M 参数）+ 词表 `outputs/neck_motion/vocab.json`
- 能力边界：按 speaker/listener 生成**平滑、多样**的颈部动作；**不具备可靠的内容/
  语义/节奏响应能力**；`best-of-K` 是 oracle 指标（需真实标签），线上效果取决于
  候选策略与实机 A/B，不可直接当作线上精度
- 训练产物归档：`outputs/neck_motion_v2*`（CVAE 失败记录）、`outputs/neck_motion/`
  （v1 各轮、探针、审计产物）

### 3.2 代码（`models/neck_motion/`）
| 文件 | 职责 |
|---|---|
| `dataset.py` | 数据加载（true_overlap 过滤、16k mono 音频、collate）+ `build_inference_batch`（推理/MVP 共用） |
| `audio_features.py` / `text_features.py` | log-Mel（纯 torch）/ 词表与词→30Hz 帧对齐 |
| `model.py` | ConditionEncoder + SequenceDecoder + 回归版（v1） |
| `cvae.py` | CVAE（v2，保留结论）+ `build_model`（按 model.type 分发） |
| `candidates.py` | **多候选模型（v3，当前线上模型）**，支持 `fixed_z` |
| `losses.py` | 分项损失（pose SmoothL1 β=0.1 / vel·acc L1）+ MultiCandidateLoss |
| `metrics.py` | 指标（MAE/RMSE/动力学/候选采样指标/诊断） |
| `train.py` | 训练/验证/dry-run/`--model-type {regression,cvae,candidates}`/`--ablate`/`--role-only` |
| `eval.py` | checkpoint 评估（回归/采样模型，best-of-K 等） |
| `diagnose.py` | 错配/节奏诊断 |
| `condition_probe.py` | 条件信号探针（`--task speaker|listener`、`--seeds`、dialogue bootstrap） |
| `reaction_audit.py` | 反应窗口时间轴审计 |
| `overlap_filter.py` / `rotations.py` | true_overlap 计算 / RPY↔矩阵（约定 R=Ry@Rx@Rz） |
| **`mvp.py`** | **MVP 三态调度（当前实机联调入口）** |
| `infer.py` | 单段推理（`--num-candidates` 导出候选池） |
| `baselines.py` / `analyze_overlap.py` | 基线 / 重叠分析 |
| `config.yaml` | 默认配置（model.type、candidates 超参、损失分项） |

### 3.3 MVP 集成（`mvp.py`）
- 三态：Speaking → v3 speaker 先验；Listening → v3 listener 先验；
  Silent → 统一坐标线性回中到 **`robot_neutral_pose`**（标定中位）
- **姿态语义（关键）**：`robot_actual_initial` = 每段开始实际姿态，仅作坐标系基准
  （输出统一相对轨迹相对它定义，首帧 [0,0,0]）；Silent 回中目标是 neutral，
  两者不同时绝对姿态精确回到 neutral（已验证偏差 <1e-5）
- 候选策略：`first` / `random`（--seed 可复现）/ `style_fixed`（**持久化固定
  latent 模板** `style_z.npy`，`--style-z-path` 加载，跨运行逐帧一致）/
  `energy_match`（候选能量 °/s 最接近期望；缺省启发式 speaking=池中位数、
  listening=P25；`--expected-energy` 可覆盖）
- 段间：按段起点姿态旋转合成到统一坐标，`--blend-sec`（默认 0.3s）边界线性平滑
- 输出：`unified_relative_rpy.npy`、`states.npy`（0/1/2）、`summary.json`、
  `candidates_<i>.npy`（A/B 用）、`--export-json`（自描述轨迹 JSON，实机校验用）

---

## 4. 实机联调现状（阶段 1 离线接入，进行中）

### 4.1 轨迹 JSON 接口（自描述，校验层零猜测）
`mvp.py --export-json` 输出，字段：`fps=30`、`units=radian`、
`rpy_order=roll,pitch,yaw`、`rotation_convention="R = Ry(yaw) @ Rx(pitch) @ Rz(roll)"`、
`reference`（统一相对 actual_initial）、`n_frames`、`robot_actual_initial`、
`robot_neutral_pose`、`states`（与 rpy 等长，legend 0/1/2）、`rpy`；
meta 含 strategy/seed/`max_frame_rate_deg_per_s`（实测峰值，供安全重定时）。

### 4.2 已交付文件（隔壁校验层测试用）
| 文件 | 内容 |
|---|---|
| `outputs/neck_motion/mvp_stage1/trajectory.json` | 89 帧三态，全零姿态版（对照组） |
| `outputs/neck_motion/mvp_stage1_hw/trajectory.json` | **当前主交付**：89 帧，actual_initial=实测 `[5.1493e-05, -0.011578916, -0.019520317]`，neutral=`[-0.02, 0.03, -0.01]`；峰值帧速 104.1°/s |
| `outputs/neck_motion/mvp_stage1/events.json` / `events_current_pose.json` | 输入事件（可复现） |

交付前校验全部 PASS：起点 rpy[0]=[0,0,0]（=actual_initial，实机起点偏差应≈0）、
Silent 终点绝对姿态=neutral（非平凡非零检验）、states 长度=89、三态齐全。

### 4.3 实机推进路线图（按顺序，尚未执行）
1. 完成机器人坐标系和轴方向标定（对齐 R=Ry@Rx@Rz 与 relative_rpy 语义）；
2. 实现角度、速度、加速度与 jerk 约束（速度参考：speaking 段最大 68°/s
   （长片段）/38°/s（stage1 全零版）/104°/s（stage1_hw 版）；执行层做安全重定时，
   重定时比例=计划时长/名义时长）；
3. 先低速、空载测试紧急停止与回中（**急停复测：按下→动力断、无位移；释放→
   重新使能、无跳变——需实机侧执行，软件侧无法代办**）；
4. style_fixed / random / energy_match 主观 A/B 测试（候选池导出已就绪）；
5. 确定默认角色策略后进入 MVP 演示。

### 4.4 实机侧待办（软件侧配合点）
- 校验层 `loadNeckTrajectoryJson` 对接（用户侧）；若接口字段出入，按代码/配置
  确认后加显式映射，**不猜测**
- 急停复测结果补报
- 坐标标定（m1/m2/m3 与 RPY 的换算在执行层）

---

## 5. 命令速查

```bash
# MVP 生成轨迹 JSON（实机联调主入口）
python models/neck_motion/mvp.py --checkpoint outputs/neck_motion_v3/checkpoints/best.pt \
    --events <events.json> --data-root data/source/response-net/processed_dataset \
    --strategy energy_match --export-json <out>/trajectory.json

# 训练 / 验证（研究冻结，一般不再用）
python models/neck_motion/train.py --dry-run
python models/neck_motion/train.py --model-type candidates --epochs 30

# 评估 / 诊断 / 探针（复现研究结论用）
python models/neck_motion/eval.py --checkpoint .../best.pt --samples 10
python models/neck_motion/diagnose.py --checkpoint .../best.pt
python models/neck_motion/condition_probe.py --task listener --seeds 5
```

---

## 6. 安全边界（重要）

- 当前仅"**软件推理与调度链路完成**"；**尚未获得实机直接驱动资格**；
- 实机执行层完成角度/速度/加速度/jerk 限制与坐标标定之前，不得直接驱动机器人；
- MVP 层不负责关节限位、限速、坐标标定、硬件下发；
- 模型不处理 Silent 判定（由上层状态机）、不负责内容/语义响应。

---

## 7. 环境

- `.venv`（Python 3.10；torch 2.13+cu130、torchaudio（仅 functional.resample）、
  soundfile、numpy、pyyaml；无 torchcodec，读取音频用 soundfile）
- 硬件：RTX 3090（CUDA 可用）；训练 1 epoch ≈ 90s（candidates K=8）
- 音频加载：统一 mono + 16kHz（内存中，不改写 WAV）
