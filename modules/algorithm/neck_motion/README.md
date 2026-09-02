# 话语级机器人颈部运动生成（第一版基线）

基于 ResponseNet 已处理数据集的第一版"话语级颈部运动生成"模型：读取一个完整
fragment（当前话语音频 + 文本 + 词级时间戳 + 上一片段文本 + role），输出与目标等长的
相对颈部 RPY 轨迹 `pred_relative_rpy: [N, 3]`（弧度，30 fps）。

**第一版范围**：只生成相对轨迹。不负责机器人坐标系转换、关节限位、速度/加速度/jerk
限制、Silent 状态判定与硬件控制。

---

## 目录结构

```
neck_motion/
  __init__.py       包说明
  config.yaml       默认配置（数据/音频/文本/模型/损失/训练）
  dataset.py        JSONL + RPY + 音频加载、overlap 过滤、collate
  audio_features.py 音频读取（mono 16kHz）、80 维 log-Mel、插值对齐
  text_features.py  词表（仅 train 构建）、tokenization、词→30Hz 帧对齐
  model.py          共享编码器 + SpeakerHead/ListenerHead 双头模型
  losses.py         掩码 Huber 损失（pose / velocity / acceleration）
  metrics.py        验证指标（MAE/RMSE/速度/加速度/分角色/首帧误差）
  train.py          训练 / 验证 / dry-run / checkpoint / 断点续训
  eval.py           用 checkpoint 在 val/test 评估（回归指标 / CVAE·多候选采样指标）
  infer.py          推理：相对 RPY + 可选机器人旋转矩阵 + 候选池导出
  mvp.py            MVP 集成：Speaking/Listening/Silent 三态调度 + 候选选择策略
  cvae.py           片段级条件 CVAE（v2，验证未通过，保留实现与结论）
  candidates.py     条件多候选轨迹模型（v3，min_k + 有界多样性 + 平滑/幅度约束）
  baselines.py      全零 / 仅 role 基线（train 统计 -> val 评估）
  condition_probe.py 条件预测信号探针（speaker/listener 分开，dialogue bootstrap）
  reaction_audit.py 反应窗口原始时间轴审计（空隙分布 / 可变长窗口可用性）
  analyze_overlap.py  overlap 分布分析 + true_overlap 生成（过滤核对）
  overlap_filter.py  true_overlap 计算（实际词区间交集，不含窗口 padding）
  rotations.py      RPY <-> 旋转矩阵（数据约定 R=Ry@Rx@Rz，轨迹段相对化）
  README.md
```

所有输出写入 `outputs/neck_motion/`，**不修改、不污染** `data/source/response-net/`。

---

## 数据接口

样本（train/val/test.jsonl 中一条，speaker 与 listener 各一条，共享同一窗口与音频）：

```json
{
  "sample_id": "<uid>_speaker",
  "role": "speaker",
  "current_utterance": {
    "text": "完整话语",
    "audio_path": "audio/<uid>.wav",
    "word_timestamps": [{"text": "词", "start_time": 0.0, "end_time": 0.2}]
  },
  "previous_listener_context": {"text": "上一片段文本", "audio_path": "..."},  // 可为 null
  "initial_pose": [roll, pitch, yaw],   // 仅元数据，不进入网络
  "target_rpy_path": "poses/<uid>_speaker_relative_rpy.npy",   // 监督（相对，弧度）
  "target_delta_rpy_path": "poses/<uid>_speaker_delta_rpy.npy",
  "fps": 30, "num_frames": 289, "duration_sec": 9.62
}
```

数据约定（与 `data/source/response-net/processed_dataset/FIELDS.md` 一致）：

- **relative RPY**：`relative_rpy[t] = R(initial_pose)^T @ R_abs[t]` 分解出的
  `[roll, pitch, yaw]`（弧度，`R = Ry(yaw)@Rx(pitch)@Rz(roll)`），首帧恒 ≈ 0。
- 每个 fragment 独立训练/生成/播放；不假设相邻 fragment 时间连续；
  `previous_context` 只作对话语义上下文，不参与姿态参考、标签计算或轨迹拼接。
- 所有划分已按 `dialogue_id` 完成，代码**不重新划分、不混合 split**。

加载器行为：

- 音频统一 downmix 为 mono、重采样到 16 kHz（内存中完成，不改写 WAV）。
- 默认按 `aligned_utterances.jsonl` 的 `overlap_other_speaker_sec > 0.1` 过滤
  （按 fragment 过滤，speaker/listener 两条一起剔除），日志输出过滤数量；
  `--no-filter-overlap` 或 `filter_overlap_sec: null` 可关闭。
- 词表只从 **train split** 构建并保存到 `outputs/neck_motion/vocab.json`；
  val/test 复用同一词表，绝不从 val/test 构建。

---

## 模型结构

```
当前音频 → log-Mel(80) @16kHz → 两层 Conv1d+GELU 音频编码器 ────────────┐
当前文本+词时间戳 → 词向量 → 30Hz 帧对齐（平均池化 / <SILENCE> 填充） ├─ 逐帧 concat
上一片段文本 → Embedding+双向GRU → context 向量（广播到每帧）          │
role embedding（广播） + 词覆盖二值特征 + 归一化时间位置特征           │
                                                                      ├─> Linear → hidden_dim
                                                                      ├─> 位置编码（正弦/可学习）
                                                                      ├─> Transformer Encoder ×4
                                                                      ├─> SpeakerHead / ListenerHead
                                                                      └─> pred_relative_rpy [B, N, 3]
```

默认配置：`hidden_dim=256`、音频编码器两层 Conv1d(5)+GELU、Transformer 4 层 /
8 头 / FFN 1024 / dropout 0.1、`SpeakerHead: 256→128→3`、`ListenerHead` 同构独立参数。

要点：

- 音频 log-Mel（约 100 帧/秒）按样本**线性插值**对齐到目标的 N 个 30 Hz RPY 帧。
- 词向量按 `start_time/end_time` 对齐到覆盖的帧；同帧多词平均池化；无词帧用
  可学习 `<SILENCE>` 向量；另加每帧覆盖率二值特征与归一化时间特征。
- 首片段上一片段文本为 null，使用 `<START_OF_DIALOGUE>` token。
- Speaker 与 Listener **共用全部输入编码器与 Transformer**，仅输出头独立，按样本
  role 路由。
- 模型直接预测完整 relative RPY（非逐帧累加 delta，避免漂移），**首帧强制为
  `[0,0,0]`** 以匹配数据定义。
- `initial_pose` 保留在元数据与推理接口中，但**不进入神经网络**（避免引入人物
  坐姿与相机偏置）。

---

## 损失与指标

```
L_total = 1.0 × L_pose + 0.5 × L_velocity + 0.1 × L_acceleration
```

- 均为弧度单位的 Smooth L1（Huber），只对有效帧计算，padding 不参与；
  长度不足的样本安全跳过速度/加速度项（无 NaN）；每项单独记录。
- `L_pose`：预测相对 RPY vs target relative RPY。
- `L_velocity`：相邻帧差分（预测差分 vs 数据提供的 delta RPY 文件）。
- `L_acceleration`：二阶差分。

验证指标（按样本聚合，速度/加速度单位为 °/s 与 °/s²）：

```
relative RPY MAE（rad 与 degree）、RMSE（rad 与 degree）、velocity MAE（°/s）、
acceleration MAE（°/s²）、speaker / listener 各自 MAE、首帧误差、
输出/目标幅度（|pred| / |t|）与逐样本相关 corr（轨迹诊断）
```

---

## 使用

### 1. dry-run（推荐先跑，验证全链路，约 1 分钟）

```bash
python neck_motion/train.py --dry-run
```

完成：数据加载 → 特征提取 → forward → loss → backward → optimizer step，
并对 train/val 各一个 batch 做自检（16kHz mono、文本对齐长度 = N、输出 shape、
padding 不计入 loss、Speaker/Listener 路由、首帧为零、无 NaN/Inf），全部 PASS 后退出，
不保存 checkpoint。

### 2. 训练

```bash
# 默认配置（config.yaml：30 epochs，batch 16，lr 1e-4，num_workers 4）
python neck_motion/train.py

# 自定义
python neck_motion/train.py \
    --config neck_motion/config.yaml \
    --epochs 30 --batch-size 16 --lr 1e-4 --num-workers 4 --seed 42 \
    --output-dir outputs/neck_motion

python neck_motion/train.py \
    --config neck_motion/config.yaml \
    --epochs 10 --batch-size 16 --lr 1e-4 --num-workers 4 --seed 42 \
    --output-dir outputs/neck_motion

# 断点续训
python neck_motion/train.py --resume outputs/neck_motion/checkpoints/last.pt
```

- 设备自动选择 CUDA → MPS → CPU；`--device cpu/cuda/mps` 可强制。
- 每 epoch 保存 `checkpoints/last.pt`，val loss 改善时保存 `checkpoints/best.pt`；
  训练/验证日志写入 `outputs/neck_motion/train.log`，每 epoch 指标写入
  `val_metrics.json`。

### 2b. 重叠过滤核对（可选）

```bash
python neck_motion/analyze_overlap.py
```

导出 overlap 分布、被过滤 fragment 明细（CSV）与 true_overlap 映射到
`outputs/neck_motion/overlap_analysis/`。第一版结论：窗口 overlap（含 padding）
过滤 694 个 fragment（14.8%），其中约 93% 是轮次边界处窗口 padding 捕获的
正常相邻发言；改用**实际词区间交集**（true_overlap，不含 0.3s/0.5s padding，
由 overlap_filter.py 计算）后只过滤 275 个（5.9%），419 个原误删样本被恢复，
0 个新增误删。新过滤集 true_overlap 中位数 0.24s、最大 1.04s，是真实的
词级交叉。

### 3. 推理

```bash
python neck_motion/infer.py \
    --checkpoint outputs/neck_motion/checkpoints/best.pt \
    --input sample.json \
    --data-root data/source/response-net/processed_dataset
```
输入 JSON 与训练样本结构兼容（见 `infer.py` 头注释）。输出：

- `predicted_relative_rpy.npy`：`[N, 3]` 弧度（30 fps，首帧为 0）。
- 若提供 `robot_actual_initial: [roll, pitch, yaw]`，额外输出
  `predicted_robot_rotation_matrices.npy`：`[N, 3, 3]`，其中
  `R_robot[t] = R(robot_actual_initial) @ R(pred_relative_rpy[t])`
  （`R = Ry(yaw)@Rx(pitch)@Rz(roll)`，与数据集约定一致）。

推理脚本不负责将其下发到机器人，不实现任何硬件控制。

### 4. 基线 / 消融 / 评估

```bash
# 全零、仅 role 均值基线（train 统计 -> val 评估，与模型同过滤条件）
python neck_motion/baselines.py

# 消融训练：去音频 / 去文本（输出到 outputs/neck_motion/ablate_<name>/）
python neck_motion/train.py --epochs 5 --ablate audio
python neck_motion/train.py --epochs 5 --ablate text

# 用 checkpoint 在 val/test 上重新评估（含轨迹诊断）
python neck_motion/eval.py --checkpoint outputs/neck_motion/checkpoints/best.pt --split val
```

### 第一版基线结论（修正过滤后，5 epoch 试验）

**第 1 轮（Huber β=1 rad 全局）**

| 模型/基线 | MAE° | RMSE° | vel°/s | acc°/s² | \|pred\|° | corr |
|---|---|---|---|---|---|---|
| 全零基线 | **2.783** | **3.716** | **5.529** | **86.8** | 0 | — |
| 仅 role 均值 | 2.783 | 3.716 | 5.540 | 87.0 | ≈0 | — |
| 完整模型 | 2.819 | 3.743 | 6.025 | 105.7 | 0.36 | 0.040 |
| 去音频 | 2.825 | 3.747 | 5.946 | 103.4 | 0.36 | 0.027 |
| 去文本 | 2.811 | 3.737 | 5.888 | 100.1 | 0.29 | 0.035 |

**第 2 轮（分项损失：pose SmoothL1 β=0.1；velocity/acc L1）**

分项 β 依据 train target 实际分位数（弧度）：|rpy| P50=0.029 / P75=0.069 /
P90=0.131；|Δ|/帧 P50=0.0015 / P75=0.0038；|Δ²|/帧² P50=0.0009 / P75=0.0020。
速度/加速度量级太小，SmoothL1 二次区会把梯度再压弱，故直接用 L1。

| 模型 | MAE° | RMSE° | vel°/s | acc°/s² | \|pred\|° | corr |
|---|---|---|---|---|---|---|
| 完整模型 | 2.795 | 3.724 | 5.641 | 90.8 | 0.22 | 0.027 |
| Speaker | 3.524 | 4.582 | 7.474 | 116.2 | 0.14 | 0.009 |
| Listener | 2.067 | 2.866 | 3.809 | 65.3 | 0.30 | 0.046 |

**判定（预先固定）：MAE < 2.783°？✗（2.795）；corr 明显 > 0.04？✗（0.027）；
幅度明显 > 0.36°？✗（0.22）** —— 三项全部不达标，且与第 1 轮（β=1）几乎无
差异（0.22 vs 0.36 幅度、0.027 vs 0.040 相关）。**结论：问题不是损失尺度，而是
“一对一确定性回归”会把多种合理颈部动作平均为零**（relative RPY 逐帧条件均值
≈ 0）。按预定标准停止 30 epoch 训练，转向概率/多样性生成目标。

### v2：片段级条件 CVAE（5 epoch 验证）

实现（`cvae.py`）：共享 ConditionEncoder → h_cond；后验 q(z|h,target)（GRU 编码
target）+ 先验 p(z|h)；z 对整段共享（z_dim=16）广播注入解码；重构（分项损失）+
KL（含 warm-up 与 free-bits）。

先验采样 K=10 的验证指标（val，754 条）：

| 指标 | 结果 | 说明 |
|---|---|---|
| 采样幅度 | 0.17°（目标 2.78°） | 退化 |
| 跨采样 σ | 0.01° | 同一输入 10 次采样几乎相同 |
| 多样性（两两距离） | 0.02° | 采样退化为确定性输出 |
| best-of-10 MAE | 2.789° | ≈ 全零基线（2.783°） |
| vel / acc | 0.44°/s / 10°/s² | 平滑但无内容 |
| max 幅度 P95 | 0.5°（目标 9.2°） | 远低于目标运动范围 |

**关键诊断：posterior collapse 与 KL 参数无关。** 全程 β_kl=0（无任何 KL 压力）
的对照实验与 β=0.001+free-bits+warm-up 的训练曲线逐位相同；训练后 KL≈0.01
nats，KL 项对总损失贡献 <0.1%。机制：relative RPY 条件均值≈0 时，解码器
“忽略 z 直接输出零”的重构损失已接近最优，q(z|h,target) 失去梯度并塌缩；
KL warm-up 与 free-bits 无法挽救此类重构驱动的 collapse。

**v2 判定：未通过**。CVAE 的重构目标与“多种合理颈部动作”不相容。下一步候选：
(a) DLow 式 best-of-K 多样性重构损失（CVAE 内最后手段）；(b) 条件扩散
（去噪目标无均值坍缩，与多峰条件分布匹配，成本高一个量级）。

### v3：条件多候选轨迹模型（5 epoch 验证，通过）

实现（`candidates.py`）：训练与部署完全一致的先验采样（无后验，杜绝作弊路径）。
固定标准正态先验 z（z_dim=16，整段共享）→ K=8 候选一次解码（batch 拼接）：

```
L = min_k L_recon(ŷ_k, y) + λ_div·max(0, div_target − pairwise_dist)   # 有上界多样性
    + λ_smooth·hinge(|Δ²| − acc_ceiling)                               # 每条候选
    + λ_amp·hinge(|rpy| − amp_ceiling)                                 # 每条候选
```

5 epoch 验证（val 754 条，先验采样）：

| 指标 | 结果 | 判定（门槛） |
|---|---|---|
| best-of-8 MAE | **2.502°**（K=16 → 2.361°；K 单调：3.26/2.69/2.50/2.36） | ✓ < 2.783° |
| 候选两两距离 | **1.92°** | ✓ >> 0.02° |
| 幅度 | **1.40°**（目标 2.78°，跨候选 σ 0.62°） | ✓ >> 0.17° |
| 平滑度 | vel 2.46°/s、acc 52°/s²（目标 P50 44/P75 101 °/s²） | ✓ 无失控 jerk |
| 先验采样 | 训练/部署同一路径，无后验 | ✓ |

消融（条件生成分布对模态的依赖）：

| 模型 | best-of-8 | 多样性 | 幅度 | vel°/s | acc°/s² |
|---|---|---|---|---|---|
| 完整 | 2.502° | 1.92° | 1.40° | 2.46 | 52 |
| 去音频 | 2.482° | 2.05° | 1.54° | **0.76** | **17** |
| 去文本 | 2.503° | 1.97° | 1.45° | 1.93 | 41 |

音频移除后生成轨迹动力学显著变弱（vel 2.46→0.76°/s，acc 52→17°/s²）——
**音频携带节奏/运动信息，条件分布确实依赖输入模态**；文本影响较小。
best-of-K 作为 oracle 覆盖度指标对“静态候选池”分辨力有限，动力学维度
是消融的主要可观测窗口。

### v3.5：条件建模诊断（role-only 对比 + 错配测试 + 节奏对齐，未通过）

新增 `diagnose.py`：同一 checkpoint 在 val 上跑三种条件模式（匹配 / batch 内
音频错配 / 文本错配），并新增 role-only 多候选基线（仅保留 role+时长+时间位置，
同 K/损失/约束）。

| 模型/条件 | best-of-8 | 多样性 | 幅度 | vel°/s | 节奏相关 |
|---|---|---|---|---|---|
| 完整-匹配 | 2.472° | 1.95° | 1.41° | 2.48 | **+0.121** |
| 完整-音频错配 | 2.475° | 1.96° | 1.43° | 2.52 | +0.106 |
| 完整-文本错配 | 2.476° | 1.94° | 1.41° | 2.47 | +0.130 |
| role-only-匹配 | **2.462°** | 2.07° | 1.52° | 0.70 | N/A |
| role-only-音频错配 | 2.475° | 2.05° | 1.51° | 0.70 | N/A |

判读：
- **完整模型不优于 role-only**（best-of-8 2.472 vs 2.462，role-only 甚至略好）；
- **错配对 best-of-K/多样性/幅度/vel/acc 完全无影响**，节奏相关仅降 0.015
  （文本错配反而 +0.130）——候选生成不依赖输入条件；
- **数据固有节奏信号极弱**：目标轨迹速度包络与音频能量包络的相关仅 +0.023
  （median 0.026）；预测的 +0.121 是平滑轨迹与平滑能量包络的伪相关，错配不敏感。
- 唯一模态敏感信号是**运动能量**（vel 2.48 vs 0.70°/s）：音频/文本调制了
  “动多少”，但没有调制“何时动、动成什么样”。

**诊断结论：v3 学到的是宽覆盖的无条件动作先验 + 条件调制的运动能量，
不是“依据对话内容的颈部运动”**。按预定流程不投入长训练，转向条件扩散
（推荐在压缩后的轨迹 latent 上做扩散）。

### v4：条件预测信号探针（`condition_probe.py`，可证伪实验，结论：无信号）

在进入扩散前做的低成本数据层面检验：用窗口条件预测本人运动的 4 个低维标签
（nod / shake / yaw 方向 / log 运动能量），单任务小 MLP，5 个条件组
（role_only / audio / text / other=对方头部 RPY / full），两种窗口模式
（whole=整窗信号上限；causal=前 60% 条件 -> 后 40% 标签）。

| 模式 | 条件组 | nod AUROC | shake AUROC | dir bal-acc | energy R² |
|---|---|---|---|---|---|
| whole | role_only | **0.840** | **0.823** | 0.413 | **0.232** |
| whole | +audio | 0.824 | 0.825 | 0.364 | 0.151 |
| whole | +text | 0.809 | 0.805 | 0.379 | 0.034 |
| whole | +other | 0.791 | 0.820 | **0.421** | 0.244 |
| whole | full | 0.782 | 0.807 | 0.397 | 0.067 |
| causal | role_only | **0.805** | **0.842** | **0.418** | **0.147** |
| causal | +audio | 0.806 | 0.844 | 0.389 | 0.134 |
| causal | +text | 0.791 | 0.817 | 0.392 | −0.189 |
| causal | +other | 0.801 | 0.833 | 0.398 | 0.117 |
| causal | full | 0.801 | 0.814 | 0.363 | −0.136 |

（AUROC 0.5 / bal-acc 0.333 / R² 0 为无信号基线；“+other”即“对方头部 RPY 统计”，
部署时相机可观测。）

结论：
- **任何条件（音频 / 文本 / 对方头部 RPY）相对 role_only 均无显著正向增量**——
  whole 与 causal 两种设置一致；
- 本人运动的可预测信号几乎全部来自 role + 时长（nod AUROC 0.84、energy R² 0.23），
  即“角色行为差异”而非“对话内容”；
- 对方 energy 与本人 energy 负相关（−0.40），主要由 role 传导（speaker 动得多 /
  listener 动得少）——两人头部运动是“角色分工”而非“互相响应”；
- 与 v3.5 错配/节奏诊断完全一致：**当前数据定义下条件信息不足，扩散模型
  无法凭空创造“话语内容决定头部轨迹”的信息**。

建议：目标 A（自然但不必精确响应话语）→ 保留 v3 作为按 role 采样的动作先验；
目标 B（根据对方内容反应）→ 需补充更强的视觉条件数据（对方人脸/视线/表情/
手势，部署时相机可得）与机器人自身运动历史/交互状态；新数据到达后，先用
`condition_probe.py` 验证条件确实携带预测信号，再做带视觉条件的 latent 扩散。

### v4.1：任务错位修正——Listener 后反应标签重建（`condition_probe.py --task listener`）

用户指出 Listener 任务语义错位：产品语义是“对方说完后机器人再反应”，但原标签
是**对方说话期间**的同步 Listener RPY。已重建：

- 标签 = 对方最后一个词结束后 [end+0.2, min(end+2.0, 窗尾)] 的 Listener 运动
  （实际可用约 0.3s——数据管线只保留末词后 0.5s 轨迹）；
- 反应段用绝对轨迹重新相对化（`rotations.py`：R(段起点)^T @ R_abs[t]，首帧≈0）；
- 只保留反应窗口内无他人发言干扰的样本（同 dialogue 其他 fragment 词区间重叠
  >0.05s 即剔除）；
- Speaker / Listener 分开评估，5 个随机种子报告 mean±std。

**样本量**：listener 后反应可用样本大幅减少（train 3577→561，val 377→77，
84% 被剔除——绝大多数因反应窗口内下一人已接话；数据管线后垫 0.5s 是瓶颈）。

| 任务 | 条件组 nod AUROC 增量（相对 role_only，mean±std, 5 seeds） | 判定 |
|---|---|---|
| speaker（同步） | audio −0.011±0.004 / text −0.013±0.002 / other −0.021±0.003 / full −0.019±0.002 | 无增量 |
| listener（后反应） | audio **+0.035±0.009** / text **+0.039±0.012** / other **+0.041±0.021** / full **+0.045±0.017** | **一致正增量（2–3σ）** |

（listener 基线：role_only nod AUROC 0.489——0.3s 反应窗口的 nod 与时长无关；
shake/direction/energy 无稳定增量；direction 88% 为静止类，不可分。）

**结论（修正）**：
1. 任务错位假说成立——重建 Listener 后反应标签后，**首次检测到一致的条件增量**
   （nod +0.04 AUROC，音频/文本/对方运动均贡献，跨 seed 稳定）；
2. 正式结论应表述为“**现有低维统计表征 + 同步标签**未检测到增量信号”，
   而非信息论意义上的条件不存在；
3. 数据管线限制：0.5s 后垫使反应窗口仅 0.3s 且 84% 样本因他人接话被剔除；
   若将窗口后垫延长至 2.5s 重新处理，listener 后反应任务的样本量与信号
   预计都会显著改善；
4. 在此之前：v3 按角色动作先验收尾；listener 后反应列为下一版数据需求。

### v4.2：原始时间轴审计 + 可变长反应窗口 + bootstrap 置信区间（`reaction_audit.py`）

**空隙分布**（4682 fragment，话语末词结束 -> 同 dialogue 下一个词开始）：
<0.1s 20.4% / 0.1–0.2s 28.3% / 0.2–0.3s 15.7% / 0.3–0.5s 12.8% /
0.5–1.0s 8.0% / 1.0–2.0s 2.0% / >2.0s 或无下一词 13.0%。**空隙中位数约 0.15s**
——自然对话轮次衔接极快，与产品判断一致（机器人若准备马上回答，颈部动作应
由 Speaker 模型覆盖；独立 Listener 反应只需 0.3–0.8s 的短点头/保持）。

**可变长反应窗口**（start=末词+0.2s，end=min(start+2.0s, 下一人开口前)，
最低 0.3s，姿态由同 dialogue 各 fragment 窗口的 listener 绝对轨迹拼接插值，
不受 +0.5s padding 限制）：train 852 / 3808（22.4%）、val 113 / 397（28.5%）
话语可用——**“延长后垫会增加样本量”的推断不成立**：78% 的样本在前 0.5s 内
已有人接话，反应窗口本身为空或 <0.3s，与后垫长度无关。

**重建后探针（train 468 / val 66 条，5 seeds + dialogue block bootstrap）**：

| 条件组 nod AUROC（vs role_only=0.632） | 增量 |
|---|---|
| audio | −0.025±0.011 |
| text | −0.007±0.016 |
| other（对方 RPY） | −0.074±0.023 |
| full | +0.008±0.015 |

**bootstrap：full−role_only 差值 mean=+0.0097，95% CI [−0.072, +0.089]，
P(Δ>0)=0.587**——置信区间含 0，v4.1 的 +0.04 增量在抽样不确定性下无法确认
（v4.1 只有 5 个训练种子，未覆盖 77 条验证样本的抽样波动）。

**结论（按预定义流程）**：有效反应样本不足（val 66），不上高维特征；
v3 作为“角色动作先验”正式收尾。Listener 后反应任务需要新的数据采集/管线
（为反应窗口保留更长的姿态覆盖），才有条件做进一步验证。

---

## MVP 集成与冻结声明（第一版交付）

### 能力边界（准确描述）

v3（`outputs/neck_motion_v3/checkpoints/best.pt`，多候选 K=8）是第一版
**角色动作先验**：能根据 speaker / listener 生成平滑、多样的颈部动作，
**不具备可靠的内容、语义或节奏响应能力**（研究结论：当前数据定义下
音频/文本/对方姿态无稳定条件增量；Listener 后反应窗口在原数据中不足；
+0.04 AUROC 经 dialogue bootstrap 后无法确认）。

**best-of-K 是 oracle 指标**（用真实标签选候选，线上没有标签），
不能当作线上效果；线上效果取决于候选选择策略与实机 A/B 主观测试。

### 候选选择策略（`mvp.py --strategy`）

| 策略 | 行为 |
|---|---|
| `first` | 固定取第 0 个候选（可复现，等效固定 z 模板） |
| `random` | 随机采样一个候选（`--seed` 可复现） |
| `style_fixed` | 固定风格：使用**固定 latent 模板**（`--style-z-path` 指定 [K, z_dim] npy，或自动生成并保存 `style_z.npy`，seed=0），取第 `--style-idx` 个模板——同一风格跨段/跨调用逐帧可复现（不是每次重新采样后取同索引） |
| `energy_match` | 按期望动作强度选：候选运动能量（°/s）最接近期望者；`--expected-energy` 或按 role 启发式（speaking=池中位数，listening=P25） |

策略自然度需通过视频/实机主观 A/B 测试判断；`--num-candidates` 可导出每段
候选池（`candidates_<i>.npy`）供测试。

### 三态状态机

```text
Speaking  → v3 speaker 动作先验（多候选 + 选择策略）
Listening → v3 listener 动作先验（同上）
Silent    → 规则化保持/回中（统一坐标线性回中到 robot_neutral_pose）
```

**姿态语义（重要）**：`robot_actual_initial` 是每段开始时的实际姿态，
仅作为坐标系基准（统一相对轨迹相对它定义）；`robot_neutral_pose` 是机器人
标定中位（默认 `[0,0,0]`），**Silent 回中目标是 neutral 而不是 actual_initial**——
若两者不同，Silent 段结束时绝对姿态精确回到 neutral（已验证，偏差 <1e-5）。

```bash
python neck_motion/mvp.py --checkpoint outputs/neck_motion_v3/checkpoints/best.pt \
    --events events.json --data-root data/source/response-net/processed_dataset \
    --strategy energy_match
```

段间语义：各段相对轨迹按“段起点姿态”旋转合成到统一坐标系（相对
`robot_actual_initial`），段边界 `--blend-sec` 内线性平滑，输出轨迹首帧
`[0,0,0]`、末段 Silent 回中。输出 `unified_relative_rpy.npy`、`states.npy`
（0=silent/1=speaking/2=listening）、`summary.json`。

MVP 层不负责：关节限位、速度/加速度/jerk 限制、坐标标定与硬件下发
（执行层职责；演示中 speaking 段最大帧间速度约 68°/s，供限速参考）。
**安全边界**：当前仅“软件推理链路就绪”；在实机执行层完成角度、速度、
加速度、jerk 限制与坐标标定之前，不能直接驱动机器人。

### 研究冻结声明

当前数据与模型研究冻结：音频/文本/对方姿态在当前样本定义下无稳定条件增量，
Listener 后反应需要专门采集“话语后反应”数据（更长姿态覆盖）才能重新开启
内容条件模型；在此之前不再投入高维特征或扩散模型。

### 交付状态定义（冻结）

**软件推理与调度链路完成**：v3 角色动作先验 + 候选选择策略（first / random /
style_fixed 固定 latent 模板 / energy_match）+ 三态调度（Speaking / Listening /
Silent 回中到 robot_neutral_pose）+ 统一坐标轨迹输出。
**尚未获得实机直接驱动资格**：实机执行层（限速、坐标标定、安全停止）未实现，
在此之前不得直接驱动机器人。

### 实机推进路线图（按顺序）

1. 完成机器人坐标系和轴方向标定（对齐 R = Ry@Rx@Rz 与 `relative_rpy` 语义）；
2. 实现角度、速度、加速度与 jerk 约束（参考：演示中 speaking 段最大帧间
   速度约 68°/s，P95 17.6°/s；listening 最大 20.7°/s）；
3. 先低速、空载测试紧急停止与回中（Silent 回中目标 = robot_neutral_pose）；
4. 对 style_fixed / random / energy_match 做主观 A/B 测试
   （`--num-candidates` 候选池导出已就绪）；
5. 确定默认角色策略后进入 MVP 演示。

### 推理 / 评估入口

```bash
python neck_motion/infer.py --checkpoint .../best.pt --input sample.json  # 单段
python neck_motion/eval.py --checkpoint .../best.pt --samples 10          # 采样指标
python neck_motion/diagnose.py --checkpoint .../best.pt                   # 错配诊断
python neck_motion/condition_probe.py --task speaker|listener --seeds 5   # 条件信号探针
```

---

## 依赖

- Python 3.10+；`torch`、`torchaudio`（仅用 `functional.resample`，纯 torch）、
  `soundfile`（读 WAV）、`numpy`、`pyyaml`。
- 无大型预训练权重下载、无外部在线服务。
- 若 `soundfile` / `torchaudio` 缺失，脚本会给出明确安装提示，不会悄悄退化。

## 已知限制（第一版明确不处理）

1. **Silent 状态**：模型总是输出轨迹，不判定"该不该动"；静默/无反应策略由上层状态机负责。
2. **坐标标定**：不学习/不校正人物绝对坐姿与相机偏置；部署时由执行层提供
   `robot_actual_initial` 并做坐标系换算。
3. **硬件安全约束**：关节限位、速度/加速度/jerk 限制、奇异姿态等均不在本模型内。
4. **Listener 语义**：第一版 Listener 是"整段人类话语结束后生成反馈"，**不是流式反应**；
   模型一次性读取完整 fragment。
5. **上下文音频**：第一版只使用上一片段 `text`，不使用上一片段 audio。
6. **两人同时说话**：默认按 `overlap_other_speaker_sec > 0.1` 过滤此类 fragment。
