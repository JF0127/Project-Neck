# Project-Neck — 项目说明与开发契约

> 本文件是仓库**唯一**需要维护的说明文档。领域级旧文档（`dataset/AGENTS.md`、`algorithm/AGENTS.md`、`runtime/AGENTS.md`、`runtime/motor/AGENTS.md`、各 `README.md`）已删除，历史内容可用 `git show <commit>:<path>` 查阅。今后任何改动只同步本文件。
>
> 保留的仅是不需要同步的参考材料：`runtime/motor/README.MD`（SOEM 主站）、`runtime/motor/使用说明/使用文档.md`（电机手册）、`thirdparty/` 与 `runtime/audio/board/xiaozhi-esp32/`（vendored 外部工程）。

---

## 1. 项目定位

仿生机器头：数据生产 → 模型训练 → 真机运行。三个独立领域：

```text
dataset/    Raw Data            → Canonical Dataset
algorithm/  Dataset             → Training → Model Package
runtime/    Model Package + Real Input → Robot Behavior
tools/      项目级分析/可视化工具
```

边界：三个领域不通过源码互相耦合。Algorithm 通过 artifact 消费 Dataset；Runtime 只加载部署包（TorchScript），**不 import `algorithm` 训练源码**（`runtime/inference/baseline_v1.py` 即该契约的实现）。

## 2. 当前状态（以代码为准）

| 领域 | 状态 |
|---|---|
| Dataset | Clean V2 首帧以最大人物锁定主主持人，后续用人脸 identity 逐帧跟踪；忽略画中画/附加窗口内的其他人物和脸。主主持人首次出现完整脸或脖子不可用时，保留此前有效前缀，删除失败帧及之后全部内容；仅第 0 帧失败等无有效前缀情况才淘汰。输出由模型自动分男女并按主播人脸 embedding 聚类 |
| Algorithm | Baseline V1（audio+text → 30 fps `rpy_offset`）已实现；训练产物在 `algorithm/outputs/`（gitignored）。部署包 `runtime/models/baseline/{model.pt,vocab.json,config.yaml}`（TorchScript，gitignored） |
| Runtime | 当前 Ubuntu 主链为 Silero VAD + Qwen3-ASR Streaming + DeepSeek 对话 + Doubao TTS；动作使用 DeepSeek MotionPlan V2。默认 `motion.send_to_motor=false`，只生成 relative trajectory artifact，不启动反馈或连接 Motor；显式开启后才经 MotionProcessor 发送 Motor（见 §3） |
| Motor | SOEM EtherCAT 主站；三电机；`model` / `measurement` / `feedback` 三个 UDS；速度后处理。反馈需先手动归零（§4.4） |
| Motion | 默认 `DeepSeekMotionBackend` 消费 robot text、TTS duration、PCM phrase alignment 与可解释 prosody，输出允许空 segments 的高层 `MotionPlan`；Validator 后将同轮 `ProsodyAnalysis`（失败时为 None）传给 Continuous Motion Generator V4：独立三轴慢速 Postural Flow + 高能量 phrase 的局部 Prosodic Accent + `nod/turn/tilt/shake` 语义 modulation，合成为 30 fps relative RPY。Legacy sparse parser/compiler、`BaselineV1Backend` 与部署包均保留。完整 absolute trajectory 在 start continuity 与 neutral return 后统一经过未改默认行为的固定长度 `TrajectoryOptimizer` |

仓库当前**没有自动化测试目录**（已按维护成本约定删除）；验证依靠运行命令、`ctest`（motor 保留原有 C++ 测试）和手动真机测试。

## 3. Runtime 链路

```text
当前 Qwen Streaming 生成 + 执行（每轮）:
  user PCM ─→ Silero VAD ─→ Qwen3-ASR Streaming ─→ DeepSeek ─→ Doubao TTS V3
                                                                      │
                     ┌────────────────────────────────────────────────┤
                     │  robot.wav / metadata.json                     │  PCM
                     ▼                                                ▼
  PCM phrase alignment → Prosody → DeepSeek MotionPlan V2 → Validator
                     → Continuous Motion Generator V4 → raw_relative_trajectory.json
                     │                                                │
                     ├─ send_to_motor=false: 仅保存 relative artifact ─┤
                     │                                                │
                     └─ send_to_motor=true: MotionProcessor → TrajectoryOptimizer
                                              → trajectory.json → /tmp/neck_model.sock
                                                                      ▼
                                      robot 音频 → WebSocket 客户端播放

`python -m runtime` 是唯一 production entry，直接构建本地 Qwen3-ASR vLLM streaming 主链；旧 Whisper/faster-whisper Runtime、wrapper、配置和模型已删除，不提供兼容 fallback。默认 `send_to_motor=false` 时不创建 `NeckClient`、不启动 `MotorFeedbackMonitor`，DeepSeek Motion 仍正常调用且不依赖 measured RPY。

实时姿态（旁路，只读）:
  EtherCAT 反馈缓存 → FeedbackServer(30 Hz) → /tmp/neck_feedback.sock
    → MotorFeedbackMonitor → RobotState(head_rpy rad, valid, timestamp, motion_executing)
```

- 产物目录（每轮覆盖）：`runtime/generated/{robot.wav,prosody.json,deepseek_motion_plan_raw.json,motion_plan.json,raw_relative_trajectory.json,metadata.json}`；开启 Motor 发送时另写 `final_trajectory.json` 与 Motor 文档 `trajectory.json`。relative trajectory 固定 30 fps、radian、`[roll,pitch,yaw]`，不经过 measured RPY 或 MotionProcessor；final artifact 是经过 MotionProcessor/Optimizer 的 absolute RPY。
- Speech Alignment 在完整 TTS PCM 生成后执行：中文文本保守切为最多 4 个 phrase，优先使用 10 ms PCM RMS 检测到的真实停顿边界，其次使用局部低能量 valley；证据不足时显式标记 `proportional_fallback`。Prosody V1 从 PCM/alignment 提取 segment duration、前后 pause、RMS mean/peak、句内 relative energy、energy peak time、字符语速和离散等级；当前无可靠 F0 依赖，artifact 明确记录该特征未提取。alignment/prosody 不可用时回退原 `text + duration` payload，不伪造音频特征。
- MotionPlan V2 顶层固定为 `mode/duration_sec/segments`，segment 固定为 `start_sec/end_sec/action/primary_axis/amplitude_deg/reason`；空 segments 表示没有 semantic gesture，但 V4 仍生成慢速 postural flow；有高能量 prosody 时可叠加非语义的 pitch accent。临时 action vocabulary 为 `nod→pitch`、`turn→yaw`、`tilt→roll`、`shake→yaw`。Validator 集中检查 schema、时长、动作—轴兼容、±5° 幅度及 overlap；任一非法 segment 令整份 plan 失败，不再局部接受。
- Continuous Motion Generator V4 三层独立生成（`generate_layers()` 可在软件实验导出完整 radian RPY 层）：Postural Flow 用每轴不同的不均匀间隔、长 pattern、minimum-jerk MOVE + soft HOLD，roll/pitch/yaw 目标幅度分别为 0.9/1.4/1.9°，间隔约 2.4～3.2/2.2～3.0/1.8～2.7 s；majority slow/fast 的语速只将间隔乘 1.06/0.94，不用随机或正弦。Prosodic Accent 仅对 high energy 或 relative_energy>1.15 的 phrase，在 energy_peak_time_sec 附近生成 0.3～0.75° pitch 短时 minimum-jerk excursion，宽度 0.35～0.65 s（受 segment duration 限制）；无可靠 prosody 时该层全零，暂不按 pause 调节 postural hold。Semantic Gesture 仍使用 nod/turn/tilt 单轴 smooth excursion 与 shake 的 `0→+A→-0.8A→0`，归零回到当时 carrier 而非 global zero。第一帧严格 zero，末帧不强制归零；合成逐轴 ±5°，越界依序缩小同向 postural、prosodic、最后 semantic（数值末级 clamp）。旧 `motion_compiler.py` sparse action parser/compiler 保留供 legacy 回归和 V1 audit，不在默认 production path。
- `TrajectoryOptimizer` 在完整 absolute RPY 上执行对称 `[1,4,6,4,1]/16` smoothing、position-domain 速度投影和迭代加速度投影；固定 30 fps、帧数、首末姿态和 states。默认参数仍为 1 pass、60 deg/s、500 deg/s²、最多 64 轮投影，不做 jerk limit 或语义优化。
- DeepSeek Motion API、JSON、MotionPlan validation/generation、Motor 或反馈失败均只跳过本轮动作，语音正常播放，不使用默认摇头；`head_rpy_valid == false` 同样只播音频。Baseline A/B 路径仍保留原 fallback。
- Qwen 主链等待 Audio Client 在实际启动本地播放时回传 `robot_playback_started`，随后发送轨迹；`motion.sync_offset_ms` 仅作为该事件之后的微调（当前为 0）。
- 轨迹发送前会检查 `RobotState.motion_executing`，避免与正在执行的轨迹冲突。
- `tools/motion_diversity_audit.py` 保留原 V1 审计与产物；`tools/motion_diversity_audit_v2.py` 对同一固定 16 句各生成一次真实 Doubao TTS，并复用 PCM/alignment/prosody 调用 V2 Planner 各 3 次，结果在 `runtime/experiments/motion_diversity_audit_v2/`。当前 V2 audit 实测 48 次请求中 46 次完整成功，G2 的 run 1/3 因 action overlap 被 Validator 拒绝并保留 raw/error；完整比较见该目录 `report.md`。`tools/continuous_motion_v4.py` 纯软件合成 prosody/plan，导出 V4 三层及 raw/optimized、V3 同 plan 对比的 metrics/7 张曲线至 `runtime/experiments/continuous_motion_v4/`（gitignored；无需网络/硬件）。

## 4. 冻结接口

### 4.1 Audio ↔ Runtime

Runtime 是 WebSocket **server**，默认 `0.0.0.0:8765`；Audio 是 client。

- 16 000 Hz、mono、signed 16-bit little-endian、`pcm_s16le`、20 ms/frame、320 samples / 640 bytes。
- Binary frame 是裸 PCM；控制消息：

```json
{"type":"stream_start","stream_id":"user_<id>","source":"user","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
{"type":"stream_end","stream_id":"user_<id>"}
{"type":"robot_playback_started","stream_id":"robot_<id>"}
```

Robot 音频使用同一协议且 `source="robot"`。Client 收完 Robot stream 后，在本地 `paplay`/PortAudio 真正启动时回传对应 stream ID 的 `robot_playback_started`；Qwen Runtime 收到后才发送 Motor 轨迹。不得单端修改格式、时序、buffer 或错误语义。

### 4.2 Runtime → Motor

同机 Unix Domain Stream Socket `/tmp/neck_model.sock`：每次连接发送一个完整 UTF-8 JSON，client `shutdown(SHUT_WR)`，EOF 为消息边界。**没有应用层 ACK**，EOF 不代表执行成功。

```json
{"name":"turn_xxx","fps":30,"unit":"radian","order":["roll","pitch","yaw"],
 "trajectory":[[0.0,0.0,0.0]],"states":["silent"]}
```

固定 30 fps、`radian`、`[roll,pitch,yaw]`；每帧三个 finite number；`states` 若存在只能是 `speaking/listening/silent` 且与 trajectory 等长。Motor 侧会把速度超限段插帧后执行。

### 4.3 Motor → Runtime 反馈（只读）

`/tmp/neck_feedback.sock`，30 Hz NDJSON；valid 帧：

```json
{"type":"motor_state","version":1,"timestamp":1789044128.302934,"valid":true,"motion_executing":false,
 "unit":"degree","order":["pitch","roll","yaw"],
 "head_rpy":[-0.012,0.015,0.0001],"motor_angles":[10.0,26.0,145.0],"sequence":[28580,28569,28558]}
```

invalid 帧：`{"type":"motor_state","version":1,"timestamp":...,"valid":false,"motion_executing":...,"reason":"feedback_unavailable"}`。
Runtime 侧 0.2 s 无消息即判定 stale（`head_rpy_valid=false`）。

### 4.4 归零规程

电机是问答模式：**没有命令就没有反馈**。每次 motor 启动后必须手动执行一次：

```text
NeckPoseSet 0 0 0 0
```

单电机角度查询命令为 `MotorAngleGet <SlaveId> <PassAge> <MotorId>`，复用电机参数查询协议并输出实际角度。

位置帧常驻后反馈持续有效；冷启动姿态不通过查询轮询解决，属于操作规程。

### 4.5 Motor Socket 汇总

| Socket | 方向 | 用途 |
|---|---|---|
| `/tmp/neck_model.sock` | Runtime → Motor | 轨迹 JSON |
| `/tmp/neck_feedback.sock` | Motor → Runtime | 30 Hz 姿态 NDJSON |
| `/tmp/neck_measurement.sock` | Runtime → Motor | 配置测量输出（可选，记录执行期实测 RPY） |

## 5. Dataset 接口

### 5.1 Clean V2 有效前缀清洗与主播归类

`src/processing/clean_videos.py` 的输入是原始视频目录，输出是新的完整目录；不改写输入。首个解码帧以检测框面积最大的 person 锁定主主持人，并要求其面积达到画面阈值；person/face 关联允许 YOLO Pose 的人物框从脖子开始而省略头部（横向扩展 10%、向上扩展人物框高度的 35%）。后续帧只接受与累计主主持人 ArcFace embedding 达到 tracking cosine threshold 的 person/face 配对；同一张主脸关联到多个 pose box 时，优先选择双肩与脖子证据最强的框。画中画、新闻素材窗口和其他附加窗口里的非主主持人人物/人脸全部忽略，不因全画面人数或脸数大于 1 而停止。

逐一解码并检查**每一帧**；主主持人首次无法继续匹配、主主持人脸触边而不完整、主主持人双肩关键点不足，或主主持人脸—双肩几何不能支持脖子可见时，保留 `[0, failure_frame)`，删除失败帧及之后全部内容，即使主持人后续恢复也不再保留。截断前缀用 ffmpeg 精确到失败帧时间重新编码为 MP4；从未失败的完整视频原样复制。只有第 0 帧失败、空视频、损坏视频等没有可用前缀时才写入 rejected。YOLO11 Pose 提供 person 与双肩，InsightFace `buffalo_l` 检测脸、跟踪主主持人 identity、预测男女并生成聚类 embedding。这里“脖子可见”是脸框与双肩关键点的模型几何判定，不是像素级脖子分割。

合格视频按主主持人的模型预测结果分为 `male/female`，再在各性别内按视频平均归一化 embedding 和 cosine threshold 贪心聚类为匿名 `anchor_XXXX`。输出结构：

```text
<output>/
├── male/anchor_0001/<source_id>.<ext>
├── female/anchor_0001/<source_id>.<ext>
├── accepted.jsonl
├── truncated.jsonl            # 被截断但保留有效前缀的视频及停止原因
├── truncated_details/<source_id>.json
├── rejected.jsonl             # 无有效前缀视频的原因与证据
├── rejected_details/<source_id>.json
├── diagnostic_previews/       # 首个失败帧；标出 person/face 框、帧号和时间
└── manifest.json              # complete=true 才表示发布完成
```

输出以 staging directory 构建后原子发布；已有输出必须显式传 `--force` 才会整体替换。每条截断或淘汰记录会立即写入 staging 下对应 JSONL 和独立 details JSON，因此中断后仍可检查已处理视频；诊断必须给出原因码、中文原因、首个失败帧、视频秒数、保留帧数/时长、画面尺寸、检测到的 person/face 数量及框、相关置信度/几何值与对应阈值。模型阈值集中在 `dataset/configs/cleaning.yaml`。Pose 权重首次缺失时下载至 `dataset/models/`；InsightFace 首次初始化时下载 `buffalo_l` 到同一模型根目录。Dataset 依赖 `onnxruntime-gpu`；CUDA 模式要求 Torch CUDA 与 ONNX Runtime CUDA provider 同时可用，GPU 包也保留 CPU provider 供显式回退。清洗结果是模型判定，失败帧预览用于审查误检。

### 5.2 人工标注真源

最终人工句子标注只消费人工审核完成的标注包，不允许未经审核的模型文本、语音边界或句子边界直接成为训练真源。允许使用本地 ASR 生成 `review.status=pending` 的单视频粗标注 JSON 供人工校正；粗标注不是人工标注包，后续 fragment、RPY 和训练处理不得在审核完成前消费它。`dataset/src/crawler/` 仅负责原始视频采集，不参与标注。

每个标注包结构固定为：

```text
<package>/
├── metadata.jsonl             # 必需，权威机器接口
├── segments/                  # 必需，人工边界切出的 WAV
│   └── <segment_id>.wav
├── metadata.csv               # 可选，仅供人工查看
└── README.txt                 # 可选，不作为机器接口
```

`metadata.jsonl` 每行一个 UTF-8 JSON object，字段必须且只能按以下语义使用：

```json
{"id":"Test_0000_001","source":"Test_0000.wav","file":"segments/Test_0000_001.wav",
 "start_sec":0.0,"end_sec":2.31,"duration_sec":2.31,"text":"主播说联播，今天我来说。"}
```

| 字段 | 契约 |
|---|---|
| `id` | 包内唯一、非空的片段 ID；派生产物沿用该 ID |
| `source` | 无目录分量的源 WAV 文件名；同一 JSONL 内必须一致 |
| `file` | 包内相对 WAV 路径，不允许绝对路径或 `..` |
| `start_sec/end_sec` | 源音频时间轴上的人工边界，秒；递增、不得重叠，允许人工明确保留间隙 |
| `duration_sec` | `end_sec - start_sec`，误差不超过 1 ms |
| `text` | 人工确认的非空原文；标点属于标注内容 |

片段音频固定为 16,000 Hz、mono、signed 16-bit PCM WAV；文件实际时长与 `duration_sec` 误差不超过 1.5 sample。行顺序就是源时间顺序。`metadata.csv` 和 `README.txt` 不得反向覆盖 JSONL。

### 5.3 派生 RPY 接口

`extract_neck_rpy.py` 只按人工边界切分视觉特征，不改变标注。调用方必须保证传入视频与 `source` WAV 共用同一个从 0 开始的时间轴。输出 `<rpy_root>/<id>/{rpy,roll,pitch,yaw}.npz` 和根目录 `manifest.json`：

- `rpy.npz`：`video_timestamps: float64[N]`、`local_timestamps: float64[N]`、`valid: bool[N]`、`values: float32[N,3]`、`order=[roll,pitch,yaw]`、`unit=radian`。
- 单轴 NPZ：相同时间戳和 valid，`values: float32[N]`。
- 不压缩帧时间轴；检测失败保留该帧并以 `valid=false`、`values=NaN` 表示。
- 当前姿态参考为源视频第 0 帧：`R_rel(t) = R_frame0.T @ R(t)`。这是视觉派生特征，不是人工文本标注。

`extract_fragment_rpy.py` 消费人工审核后的同目录 `<stem>.corrected_fragments.json`，要求 `timeline=original_audio_video`、`time_unit=second`，且 `review.status` 为 `human_corrected` 或 `human_segmented`。它按 `fragments[{id,start_time,end_time,duration,text}]` 从同名 MP4 逐帧提取，在相邻 `<stem>.neck_rpy/` 下按 fragment ID 写入相同 `{rpy,roll,pitch,yaw}.npz` 接口及根 `manifest.json`；参考姿态同样固定为视频第 0 帧，检测失败帧保留为 invalid/NaN。默认使用 MediaPipe GPU delegate（Linux EGL/OpenGL ES，在当前 NVIDIA GPU 上运行，并非 Torch CUDA backend），可显式传 `--device cpu` 回退。输出通过 staging directory 原子发布，已有输出默认拒绝覆盖，仅显式 `--force` 时替换。该 corrected-fragments JSON 是当前人工审核边界接口，但在片段 WAV 和 canonical `metadata.jsonl` 建成前仍不等同于最终训练标注包。

`visualize_slices.py` 联合检查人工音频、文本和 RPY；`visualize_roll_atoms.py` 只是候选动作诊断，不写回人工标注，也不是训练标签接口。

### 5.4 单句人工动作真机测试 fixture

`processing_test/Test_0000_motion_test/generate.py` 为 `Test_0000_001` 手工定义三轴 HOLD/MOVE；`Test_0000_002/generate.py` 生成 002；`generate_003_005.py` 生成 003～005。每条输出各自目录下的 `{trajectory.json,trajectory.png}`，固定 30 fps、radian、`[roll,pitch,yaw]`、`speaking`，MOVE 使用 smoothstep 连接，带符号 MOVE 角度按相对上一姿态的变化量累计，定义本身不来自 RPY 自动检测。001/002 分别为 69/112 帧；003/004/005 分别为 322/201/95 帧，终态依次为 `[0,-1,0]°`、`[0,0,0]°`、`[5,2,-2]°`。`combine_001_005.py` 不增加过渡或停顿，按顺序原样拼接为 799 帧（26.633 s）的 `Test_0000_001_005`。Motor 手工测试副本为 `runtime/motor/trajectories/Test001.json`～`Test005.json` 和 `Test001_005.json`，JSON 内部 name 与文件名一致。`plot_raw_references.py` 为 `Test_0000_001`～`005` 各生成一张共享时间轴的人工文本、原始 WAV 波形和原始逐帧 RPY 图到 `raw_references/`；除显示用 rad→degree 换算外，不做平滑、插值、滤波或动作提取。`trajectory.json` 是可被 Motor 接受的绝对零中心 RPY 文档，但生成/验证步骤不得连接 socket；真机发送仍须遵守 §9 和用户明确授权。

### 5.5 单视频目录与音频提取

`prepare_video_audio.py` 将指定目录下每个顶层 MP4 移入同名子目录，并从完整音轨提取同名 WAV；输出音频固定为 16,000 Hz、mono、signed 16-bit PCM。音频先在 staging directory 中生成并验证，成功后才移动原 MP4 和原子发布该视频目录；已有同名目录时拒绝覆盖。当前以 `clean_v2/male/kanghui` 为测试组。

### 5.6 本地 ASR 粗标注

`rough_transcribe.py` 使用 `dataset/models/asr/{paraformer-zh,fsmn-vad,ct-punc}` 对指定目录内同名 MP4/WAV 对进行离线中文识别，每个视频原子写入一个同名 `rough_transcript_v1` JSON。JSON 使用完整音视频的秒时间轴，包含带标点的 `segments[{id,start_sec,end_sec,text}]`，并固定标记 `review.status=pending`；已有 JSON 默认拒绝覆盖，只有明确传入 `--force` 才能替换。该输出仅是人工审核初稿。

`qwen_transcribe.py` 递归处理指定目录内的 WAV 或 MP4，使用本地 `models/qwen` Qwen3-ASR-1.7B 和 `models/qwen3-forced-aligner`，固定 `language=Chinese` 并启用 forced alignment。WAV 仍严格要求 16 kHz mono PCM s16；MP4 通过 ffmpeg pipe 只读解码为相同格式并以 `(float32 waveform,16000)` 直接送入模型，不生成中间 WAV、不修改原视频。每个输入在原目录旁原子写入 `<stem>.qwen_asr_v1.json`，`audio` 记录真实输入文件名，保留完整 `text` 与字级 `timestamps[{text,start_time,end_time}]`，并固定标记 `review.status=pending`；同目录同 stem 的 WAV/MP4 会因输出冲突而拒绝，已有 JSON 默认拒绝覆盖，只有显式 `--force` 才替换。该结果只供人工组装和审核，不是人工标注真源。

### 5.7 主播人脸选择诊断

`speaker_identity_debug.py` 使用固定 seed 从原始视频目录随机选择 10 个视频，只读取每个视频 `0.0/0.5/1.0/1.5/2.0s` 五帧，并使用本地 InsightFace `buffalo_l` 的检测、识别和性别模型验证主人脸自动选择。每帧优先在脸中心位于画面上方 60% 的候选中选择 bbox 面积最大者；没有上部候选时退化为全画面最大脸。独立输出目录包含每视频一张五帧 contact sheet、逐帧 `results.jsonl` 和带 `complete=true` 的 `manifest.json`；不保存 embedding、不聚类、不生成 speaker ID，也不修改、移动或切分输入视频。输出以 staging directory 原子发布，已有输出默认拒绝覆盖，仅 `--force` 时替换。

### 5.8 主播视频 embedding 与相似度诊断

`speaker_similarity.py` 对原始视频目录内全部顶层视频复用相同的五帧采样与主人脸选择规则，汇总每个视频成功帧的 InsightFace normed embedding：先求算术均值，再做一次 L2 normalize，得到唯一 `video_embedding`；五帧全部失败的视频显式标记为 unresolved。脚本只计算有效视频间的 embedding 点积（cosine similarity），输出按文件名排序的 `video_embeddings.npz`、100×100 `similarity_matrix.npy`（unresolved 行列为 NaN）、每视频前五近邻 `top5_neighbors.jsonl` 和 `manifest.json`，并报告非对角唯一视频对及逐视频 Top-1 的 min/mean/median/max。gender 仅作为 metadata，不参与计算；该诊断不使用阈值、不聚类、不生成 speaker ID，也不修改原始视频。输出同样原子发布且默认拒绝覆盖。

### 5.9 最终主播身份映射

原 `speaker_similarity_v1` 产物中的 99 个有效视频以 cosine similarity `>= 0.50` 建立无向边，并以 connected components 生成全局唯一 `speaker_0001`～`speaker_0009`；编号按 similarity matrix 的文件名顺序首次遇到 component 的顺序确定。`datasets/zhubo_shuo_lianbo/speaker_mapping.json` 是按视频文件名排序的 JSON array，每项固定为 `{video,speaker_id,gender}`；gender 只来自 embedding 诊断 metadata，不参与建图。低质量视频 `RPQbD38py38.mp4`、`wM8M9Uz3zpw.mp4` 及其映射记录已删除，当前映射 97 条，九个 speaker 的记录数依次为 `19/15/14/6/12/3/7/17/4`；其中历史映射项 `4JZOAXfXtdc.mp4` 的原片此前已删除，因此 `videos/` 当前实际有 96 个 MP4。原 `speaker_similarity_v1` 仍保留生成映射时使用的 100×100 审计矩阵，不随原片删除而改写。

### 5.10 音频损坏诊断

`audio_corruption_debug.py` 在正式切分前只读处理原视频：通过 ffmpeg pipe 将音轨解码为 16 kHz mono PCM，不生成中间 WAV、不修改原片；按视频 stem 匹配指定目录下已有的 `*.qwen_asr_v1.json`，并使用 Dataset 自有 `models/silero_vad/silero_vad.jit`。诊断以 1 秒窗口计算 RMS/dBFS、Silero 512-sample 帧的 speech ratio 及 Qwen timestamp 覆盖；窗口同时满足 `RMS >= -30 dBFS`、`speech ratio <= 0.10`、无 timestamp 重叠时记为异常。连续异常达到 10 秒，或异常总时长占完整音轨至少 15%，状态为 `suspected_corrupt_audio`。输出 `audio_corruption_debug/{results.jsonl,manifest.json}`，只提供候选和证据，不自动删除或切分视频；输出原子发布且默认拒绝覆盖。全部匹配 transcript 必须事先存在，否则整批拒绝启动，避免静默漏检。

### 5.11 Fragment 自动切分诊断

`fragment_split_debug.py` 只处理一个 MP4 及其同 stem `qwen_asr_v1` JSON，不裁剪媒体。脚本将 Qwen 完整文本中的标点重新附着到字级 timestamp，语义完整优先：理想时长 2～8 秒，8～10 秒正常接受，为等待句号/问号/叹号等完整句边界可延长至 12 秒；只有超过 12 秒仍无完整句边界时，才按 `0.8/0.5/0.3s` 停顿和逗号、顿号、分号等弱标点选择内部自然边界，最后在 12 秒内强制切分。不会仅因接近 8～10 秒就在明显未结束的弱标点处切断；以“因为/但是/如果/所以”等连接结构结尾的候选也不作为自然边界。不足 1.5 秒的结果按合并后是否超过 12 秒及接近理想时长的代价优先并入前后片段。输出 `fragments.json`，边界仍为 `review.status=pending` 的诊断建议，不是人工标注真源，任何 RPY/训练步骤不得直接消费。当前已对 `videos/` 中全部 96 个现存 MP4 生成 `fragment_split_debug/<video_stem>/fragments.json`：共 1511 个 fragment，其中 2 个超过 12 秒、0 个短于 1.5 秒；统一文本审查筛出的 90 个可疑项中，63 个高置信度错字/错词已仅在 fragment 文本及其审查/质量镜像中保守修正，边界、ID、原始 Qwen ASR、媒体和 RPY 均未改动，另 27 个保留在 `fragment_quality_debug/remaining_review.jsonl` 等待听审。这些仍只是冻结 V1 规则的诊断产物，未裁剪媒体，也未成为人工标注真源。

### 5.12 Fragment 质量诊断

`fragment_quality_debug.py` 只读扫描 `fragment_split_debug/*/fragments.json`，不改写边界、不删除数据、不裁剪媒体。逐 fragment 检查空文本、至多 5 个 lexical character 的极短文本、连续重复、控制字符/常见乱码、以逗号/顿号/分号/冒号或未完成连接词结尾，以及文字—时长失配（至少 15 字且超过 7 字/s；至多 12 字且时长至少 6s）；视频 fragment 数少于 5 或多于 30 时把视频级 flag 附到该视频各记录，仅表示 `suspected` 而非确认错误。输出 `fragment_quality_debug/{results.jsonl,manifest.json}`，采用 staging directory 原子发布且默认拒绝覆盖；人工文本审查另将不能保守自动修正的条目写入同目录 `remaining_review.jsonl`。当前 96 个视频、1511 个 fragment 的结果为 1303 `ok`、208 `suspected`；reason 次数见 manifest。该诊断仍不是人工审核，不得作为删除、RPY 或训练的自动依据。

### 5.13 Fragment 终端人工听审

`src/tools/review_fragments.py` 只消费 `fragment_quality_debug/remaining_review.jsonl`。每条按既有时间边界用 ffmpeg 从原 MP4 只读解码临时 16 kHz mono WAV，优先以 `paplay`（再依次 `ffplay`、`aplay`）播放，支持保留、编辑、重播、跳过和退出。确认或编辑成功后立即原子更新进度；编辑仅同步对应 `fragment_split_debug/<stem>/fragments.json` 的 `text`、`fragment_review.txt`、质量 `results.jsonl/manifest.json`，不修改原始 Qwen JSON、媒体、ID、边界、RPY 或轨迹。每次操作前校验 queue 文本与当前 fragment 唯一匹配，多个镜像使用带回滚的批量替换；启动与退出时复核 96 个视频、1511 个 fragment 的 ID/时间签名以及媒体 stat 和 Qwen SHA-256。运行期使用 `/tmp/project-neck-fragment-review.lock` 防止并发审查。

Algorithm 暂无可用训练入口，不得从训练代码隐式触发任何 Dataset 处理。

## 6. 目录结构

```text
Project-Neck/
├── AGENTS.md                  # 唯一维护文档
├── dataset/                   # 原始采集 + 人工标注消费
│   ├── configs/               # youtube.yaml（采集）、cleaning.yaml（Clean V2）
│   ├── datasets/zhubo_shuo_lianbo/  # videos/、人工标注包、派生产物（gitignored）
│   ├── models/                # mediapipe、FunASR、Qwen3-ASR、forced aligner、Silero VAD 本地模型
│   └── src/
│       ├── crawler/           # 原始视频采集
│       └── processing/        # Clean V2、人工包校验、RPY 派生、诊断可视化
├── algorithm/                 # 训练与部署导出（当前 Baseline V1）
│   ├── configs/baseline.yaml
│   ├── data/ features.py models/ losses.py metrics.py train.py
│   └── tests/                 # 已删除
├── runtime/                   # 真机在线运行；唯一入口 python -m runtime
│   ├── __main__.py qwen_streaming_runtime.py contracts.py config.yaml
│   ├── audio_server.py vad.py dialogue.py datetime_tool.py web_search.py tts.py doubao_tts.py logging_utils.py
│   ├── feedback.py            # 实时姿态 monitor
│   ├── neck_client.py         # motor JSON 发送
│   ├── inference/             # 模型 I/O 与产物（原 motion_core）
│   │   ├── base.py processor.py baseline_v1.py deepseek_motion.py motion_compiler.py
│   │   ├── motion_plan.py motion_plan_validator.py prosody.py speech_alignment.py
│   │   ├── trajectory_generator.py trajectory_optimizer.py
│   │   ├── motor_json.py default_motion.py generator.py artifacts.py
│   ├── generated/             # 每轮生成的音频+轨迹（gitignored）
│   ├── models/                # 部署资产：baseline/、silero_vad（Qwen 模型在 dataset/models/qwen）
│   ├── audio/                 # macOS/Linux 音频客户端（采集/播放/WS）
│   └── motor/                 # SOEM EtherCAT、三电机、三个 socket
└── tools/                     # trajectory_visualizer.py、V1/V2 motion diversity audit
```

## 7. 常用命令

### 7.1 Runtime 环境（Ubuntu，一次性配置）

```bash
cd /home/jhl/projects/Project-Neck
uv venv --python 3.10 runtime/.venv
source runtime/.venv/bin/activate
uv pip install --torch-backend=auto torch
uv pip install -r runtime/requirements.txt
```

### 7.2 Runtime / Motor / Audio

```bash
# Ubuntu production entry（唯一正式入口；需要 DeepSeek 与 Doubao key）
source runtime/.venv/bin/activate
export DEEPSEEK_API_KEY="<your-key>"
export TAVILY_API_KEY="<your-key>"  # web_search 后端
export VOLCENGINE_TTS_API_KEY="<your-key>"
export VOLCENGINE_TTS_SPEAKER="zh_female_vv_uranus_bigtts"  # 可选
python -m runtime

# MotionPlan V2 数据结构、prosody、validator、Continuous Generator V4、pipeline、legacy compiler 和 Optimizer 纯软件回归
python -m unittest runtime.qwen_streaming_runtime_test runtime.inference.motion_plan_test runtime.inference.prosody_test runtime.inference.motion_plan_validator_test runtime.inference.trajectory_generator_test runtime.inference.deepseek_motion_test runtime.inference.motion_compiler_test runtime.inference.speech_alignment_test runtime.inference.trajectory_optimizer_test -v

# V4 三层 + V3 对比的纯软件可视化（使用已安装 matplotlib 的 Dataset 环境，不连接硬件）
dataset/.venv/bin/python tools/continuous_motion_v4.py

# V1 结果保留；V2 audit：真实 16 次 TTS + 48 次 DeepSeek Motion API，不连接硬件
python tools/motion_diversity_audit_v2.py --phase collect --reset
python tools/motion_diversity_audit_v2.py --phase analyze

# Motor：只编译与软件测试（不要启动 executable，除非明确要做真机测试）
cmake -S runtime/motor -B runtime/motor/build
cmake --build runtime/motor/build -j"$(nproc)"
ctest --test-dir runtime/motor/build --output-on-failure

# 真机（需要明确授权，会初始化 EtherCAT；先手动归零）
sudo ./runtime/motor/build/master_stack_test
# 控制台：NeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>（度）、NeckSequence <name>、NeckSequenceStop

# Audio 目录内的独立 Qwen3-ASR 常驻模块（暂不接 VAD/WebSocket；单进程内所有 WAV 复用一次模型加载）
cd runtime/audio
source .venv/bin/activate
python -m runtime.qwen_asr /path/to/audio.wav
# 固定 WAV 的 vLLM streaming 诊断：转为 16 kHz mono float32，warmup 后按 100 ms 实时时序模拟输入
python -m runtime.qwen_asr_streaming

# runtime/audio/tools/qwen_streaming_server.py 仅为兼容 debug helper，委托给同一 production entry，
# 不再作为启动正式系统的命令。每轮产物写入 runtime/generated/；默认不连接 Motor。
# DeepSeek event 诊断应回到仓库根目录执行：python -m runtime --debug-deepseek-events

# Audio 客户端（Ubuntu 本机 PulseAudio/PipeWire；服务端需先运行）
# 依赖 pactl/parec/paplay（Ubuntu 包通常为 pulseaudio-utils）
# 固定 source/port: alsa_input.pci-0000_00_1f.3.analog-stereo / analog-input-rear-mic
# 固定 sink/port:   alsa_output.pci-0000_00_1f.3.analog-stereo / analog-output-lineout
cd runtime/audio
source .venv/bin/activate
python -m runtime.main conversation --url ws://127.0.0.1:8765

# Audio 客户端（Mac；服务端需先运行）
cd runtime/audio
brew install portaudio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export no_proxy="10.255.0.35,127.0.0.1,localhost"; export NO_PROXY="$no_proxy"  # 有代理时必须
python3 -m runtime.main check-config
python3 -m runtime.main capture-test --duration 3 --output /tmp/capture.wav
python3 -m runtime.main asr-stream       # 单连接持续多轮：采集 → 接收完整 robot stream → 播放 → 恢复采集
python3 -m runtime.main stream-test --duration 5 --wait-for-robot
python3 -m runtime.main duplex-test --turns 2 --duration 5
# 自然多轮半双工对话（单连接持续运行，服务端 VAD 分轮，Ctrl+C 停止）
python3 -m runtime.main conversation
```

默认连接 `ws://10.255.0.35:8765`（可用 `AUDIO_MODULE_WS_URL` 或 `--url` 覆盖）。Linux Audio client 默认使用 PulseAudio/PipeWire 的 `parec`/`paplay`，启动采集/播放前用 `pactl` 明确选择 Rear Microphone 与 Rear Line Out；仍固定输出 16 kHz、mono、s16le、20 ms/640 bytes，不直接打开 ALSA `hw:*`。macOS 继续使用 sounddevice/PortAudio。`conversation` 与当前 `asr-stream` 都在一个 WebSocket 上持续多轮半双工运行，不逐轮重连：每轮使用新的 stop event 和 `MicrophoneCapture`，收到机器人 `stream_start` 后停止本轮采集；Server 在完整 TTS PCM 就绪后无实时 pacing 地快速发送完整 Robot stream，Client 收完后启动本地播放并立即回传 `robot_playback_started`，Server 据此异步提交轨迹；播放完成后 Client 进入下一轮并重新发送 user `stream_start`。`NeckClient.send()` 只等待 Motor 解析、预检查并提交后台轨迹线程后关闭 socket，不等待整条轨迹执行完成，且不再阻塞 Qwen output turn。日志包含 `[CLIENT]/[SERVER]/[MOTION]` monotonic 时间点；跨机器 monotonic 时钟不可直接比较。

### 7.3 Dataset

```bash
cd dataset
python -m src.cli --help                 # discover / validate / download / clean

# 对每一帧做严格筛选；首次失败前的有效前缀保留，不下载新原片
python -m src.cli clean \
  --input datasets/zhubo_shuo_lianbo/videos \
  --output datasets/zhubo_shuo_lianbo/clean_v2
# 重建现有输出需显式添加 --force；快速软件检查可用 --limit N

# 固定 seed 随机抽取 10 个原始视频，检查前 2 秒五帧的主人脸选择；不聚类、不改原片
python -m src.processing.speaker_identity_debug

# 全部原始视频的五帧 video embedding、两两 cosine similarity 和 Top-5；不聚类
python -m src.processing.speaker_similarity

# 每个 MP4 建立同名目录，并提取 16 kHz mono PCM s16 WAV
python -m src.processing.prepare_video_audio \
  --input datasets/zhubo_shuo_lianbo/clean_v2/male/kanghui

# 本地 FunASR 粗标注；每个视频写入同名 pending-review JSON
python -m src.processing.rough_transcribe \
  --input datasets/zhubo_shuo_lianbo/clean_v2/male/kanghui/Test

# 本地 Qwen3-ASR + forced aligner；WAV/MP4 均递归写入相邻的 *.qwen_asr_v1.json
# MP4 音轨只读 pipe 解码，不生成中间 WAV
python -m src.processing.qwen_transcribe \
  --input datasets/zhubo_shuo_lianbo/videos

# 只读解码原视频音轨，联合已有 Qwen timestamps、RMS 与 Silero VAD 诊断损坏候选
python -m src.processing.audio_corruption_debug \
  --input datasets/zhubo_shuo_lianbo/videos \
  --transcripts /path/to/qwen_transcript_root

# 单视频 Qwen 文本/字时间戳的 fragment 边界诊断；不裁剪媒体
python -m src.processing.fragment_split_debug \
  --video datasets/zhubo_shuo_lianbo/videos/07OiKCM0h54.mp4 \
  --transcript datasets/zhubo_shuo_lianbo/videos/07OiKCM0h54.qwen_asr_v1.json \
  --output datasets/zhubo_shuo_lianbo/fragment_split_debug/07OiKCM0h54/fragments.json

# 全量只读检查 fragment 文本、语义结尾、文字/时长及视频 fragment 数
python -m src.processing.fragment_quality_debug

# 逐条播放 remaining_review.jsonl；Enter 保留，e 编辑，r 重播，s 跳过，q 保存退出
python -m src.tools.review_fragments

ANN=datasets/zhubo_shuo_lianbo/processing_test/Test_0000_manual_v1/metadata.jsonl
VIDEO=datasets/zhubo_shuo_lianbo/videos/Test.mp4
RPY=datasets/zhubo_shuo_lianbo/processing_test/Test_0000_manual_rpy

# 必须先验证人工包；失败时不得继续派生
python -m src.processing.manual_annotations --annotations "$ANN"
python -m src.processing.extract_neck_rpy --video "$VIDEO" --annotations "$ANN" --output "$RPY"

# 按人工审核后的 corrected fragments 批量提取每句 RPY
python -m src.processing.extract_fragment_rpy \
  --input datasets/zhubo_shuo_lianbo/clean_v2/male/kanghui

python -m src.processing.visualize_slices --annotations "$ANN" --rpy-root "$RPY" \
  --output datasets/zhubo_shuo_lianbo/processing_test/Test_0000_manual_visualization
python -m src.processing.visualize_roll_atoms --annotations "$ANN" --rpy-root "$RPY" \
  --output datasets/zhubo_shuo_lianbo/processing_test/Test_0000_manual_roll_visualization

# 生成 001～005 的无处理人工设计参考图
python datasets/zhubo_shuo_lianbo/processing_test/Test_0000_motion_test/plot_raw_references.py
# 生成 001～005 的人工动作 JSON/PNG，不发送到 Motor
python datasets/zhubo_shuo_lianbo/processing_test/Test_0000_motion_test/generate.py
python datasets/zhubo_shuo_lianbo/processing_test/Test_0000_motion_test/Test_0000_002/generate.py
python datasets/zhubo_shuo_lianbo/processing_test/Test_0000_motion_test/generate_003_005.py
python datasets/zhubo_shuo_lianbo/processing_test/Test_0000_motion_test/combine_001_005.py
```

以上处理接口适用于同格式的任意人工标注包，不再硬编码 `Test.mp4`。所有派生步骤采用 staging directory 后原子替换输出，不覆盖人工标注包。

### 7.4 Algorithm 训练

```bash
source runtime/.venv/bin/activate
python -m algorithm.train --config algorithm/configs/baseline.yaml --epochs 50
```

- 原 Dataset 训练入口已删除；新处理 artifact 完成前，该训练命令没有可用的数据入口。
- 输出：`algorithm/outputs/<experiment>/run_<timestamp>_seed42/`（checkpoint、config、vocab、metrics）。
- 部署包生成：`runtime/models/baseline/{model.pt,vocab.json,config.yaml}` 由 Algorithm 导出（历史导出脚本未入当前仓库；Runtime 只消费该产物，接口见 `runtime/models/baseline/config.yaml` 的 `input_contract: baseline_tensor_v1`）。

## 8. 配置速查

`dataset/configs/cleaning.yaml`：模型路径、YOLO Pose 推理尺寸（默认 960）、person/shoulder/face/neck 几何阈值、主主持人首帧最小面积、逐帧 identity tracking cosine 阈值、跨视频 identity 聚类阈值和支持的视频扩展名。默认严格检查每个解码帧，该行为不可通过采样参数放宽；非主主持人的附加窗口内容不参与验收、性别判断或聚类。

`runtime/config.yaml`：

| 段 | 关键字段 |
|---|---|
| `vad` | `backend: silero`, `model_path`, `threshold`, `min_speech_ms`, `min_silence_ms` |
| `asr` | `backend: qwen3_streaming`, 本地 `model_path`, `language: Chinese`, `chunk_size_sec`, `unfixed_chunk_num/unfixed_token_num`, vLLM 的 `gpu_memory_utilization/max_inference_batch_size/max_new_tokens` |
| `dialogue` | `backend: deepseek`, `model`（当前 `deepseek-flash` / DeepSeek-V4.1-Flash，Responses API streaming，`max_output_tokens=4096`、`reasoning.effort=none`；纯当前日期时间强制使用 `zoneinfo.ZoneInfo("Asia/Shanghai")` 的本地 `get_current_datetime`（UTC+08:00，禁止搜索或由模型自行推算/转换），天气/新闻等实时互联网信息使用本地 `web_search`（含“今天/目前/当前/最近”时自动加入上海绝对日期，每轮最多 3 次），普通静态知识自动跳过工具；默认使用适合语音播放的自然口语，回复尽量控制在 30 个汉字以内（含标点），仅在无法完整回答时才允许略微超过）, `base_url`, `timeout_sec`, `temperature` |
| `tts` | production 固定 `backend: doubao`（V3 HTTP Chunked、`seed-tts-2.0`、16 kHz mono PCM；speaker 来自 `VOLCENGINE_TTS_SPEAKER`，默认 `zh_female_vv_uranus_bigtts`） |
| `motion` | `enabled`, `backend: deepseek/baseline`（默认 deepseek），`send_to_motor`（默认 false；false 时只生成 relative artifact且不启动反馈/Socket），DeepSeek 的 `deepseek_model/deepseek_timeout_sec/deepseek_temperature/deepseek_max_output_tokens`，Baseline 的 `model_path/vocab_path/device`，以及 `generated_dir/sync_offset_ms` |
| `motor` | `feedback_enabled`, `feedback_socket`, `feedback_stale_sec`, `send_enabled`, `socket_path`, `measurement_socket`, `mock` |

`runtime/motor/neck/neck_config.py`（motor 唯一硬件配置）：`network_interface`、`slave_id`、三电机 `passage/id/min/max/center/max_velocity/speed_param/current_param`、RPY 范围、运动学 `c11/c12/c21/c22/k3`、`feedback.enabled/socket_path/rate_hz`。当前电机绝对角度 `[min,center,max]` 分别为 M1 `[-84,7,41]°`、M2 `[107,145,234]°`、M3 `[57,147,238]°`；Pitch 为 `[-45,40]°`，Roll 为 `[-40,40]°`，Yaw 保持 `[-117,58]°`。**不得为了让测试通过而修改标定/限位/电流/速度。**

## 9. 硬件与安全铁律

未经用户明确要求，不得：

- 运行 `sudo` Motor executable、`master_stack_test`、`NeckPoseSet`、`NeckSequence`；
- 向真实 `/tmp/neck_model.sock` 发送轨迹、连接真实 Motor socket 或 measurement socket；
- 执行 EtherCAT/CAN 命令，或连接真实麦克风/电机做软件验证；
- 修改 IK 系数、机械限位、中心、方向、电流、速度、加速度、jerk 或停止检查；
- 放宽任何 RPY / 电机位置 / 运动安全限制来让测试通过。

合法 Socket 数据会被 Motor 直接提交执行。所有软件测试必须使用纯软件 fixture/mock。

## 10. 修改规则

1. 修改前检查工作树，不得覆盖他人未提交修改。
2. 保持小步、局部、可测试；不做无关重构。
3. 不修改冻结协议（§4）和 Motor 标定；Dataset 新处理契约需在重新设计后明确记录。
4. Dataset 爬取代码当前保持不变；未经授权不重新下载原始视频。
5. Runtime 主链路保持内存/WebSocket/Unix socket；`runtime/experiments/` 只做旁路记录。
6. 新 Algorithm 不复用 runtime 的 inference 代码作为训练基础；Runtime 不 import `algorithm`。
7. **文档只维护本文件**：任何行为、命令、配置变化同步到这里；不要再新增领域级 AGENTS/README。
8. 真机测试前确认：motor 已启动并手动归零、反馈 `pose valid`、急停可达；停止顺序为先 runtime 后 motor。
