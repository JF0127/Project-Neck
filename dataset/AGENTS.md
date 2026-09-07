# AGENTS.md — dataset 数据与工程契约

> 本文件面向 Codex、Coding Agent、Research Agent 和后续自动化任务。它不是普通用户教程，而是当前仓库的权威数据导航与安全边界。开始任何任务前先阅读本文件；若文档、代码和实际 artifact 出现冲突，应停止并核对，不能静默猜测或重算数据。

`dataset/` 是 Project-Neck 的独立离线数据集生产系统，不属于机器人实时 Runtime。它负责视频发现与下载、原始数据管理、数据清洗、fragment 构建、ASR/audio/MediaPipe/neck-pose 特征提取、split、数据质量分析，并向顶层 `algorithm/` 发布训练/验证/测试数据。机器人在线系统统一位于顶层 `runtime/`。

## 1. 项目定位

`dataset` 是“机器头 / Robot Head”研究项目的数据构建模块。当前核心研究问题是：

```text
speech-conditioned neck motion generation

输入：audio，以及可选的 text / word timing
输出：与讲话同步的人类 neck/head RPY motion
```

本仓库当前提供 `Speech → Neck Motion` 监督学习所需的 paired Ground Truth。面部表情生成不是本模块的核心任务；MediaPipe blendshape/landmark 虽被保留，但不代表 facial expression model 已实现。

当前 neck Ground Truth **不是严格的 neck joint motion capture**。它来自：

```text
MediaPipe head pose
→ clip-relative head orientation
→ human-semantic RPY
```

新闻主持人的 torso 和机位通常较稳定，因此当前把 relative head orientation 当作 neck motion baseline proxy。它仍可能包含 torso/camera 影响，不能描述成真实颈关节角。

## 2. 数据集名称、来源与选择理由

- 数据集根目录：`datasets/zhubo_shuo_lianbo/`
- 主要来源：YouTube 上与“主播说联播”有关的公开视频。
- 当前配置允许频道：`CCTV中国中央电视台`、`CCTV春晚`；见 `configs/youtube.yaml`。
- 下载媒体保存在 `datasets/zhubo_shuo_lianbo/videos/`。

选择新闻主持人视频是因为通常具有：单人主持、稳定镜头、相对稳定 torso、可见 face/neck/shoulders、连续 speech。相较自然多人对话，这些条件更适合稳定提取 speech-to-neck-motion baseline 的视觉 Ground Truth。该选择降低了提取噪声，但没有消除 head/torso ambiguity。

## 3. 权威 Pipeline 与模块边界

```text
YouTube
  ↓
Discovery / Validation / Download       src/crawler/
  ↓
Clean V1                               src/cleaning/clean_v1.py
  ↓
MediaPipe V1                           src/features/mediapipe.py
  ↓
Neck Pose V1                           src/features/neck_pose.py
  ↓
Speech V1                              src/features/audio.py + asr.py
  ↓
Fragment V1.2.1                        src/fragments/fragment_v1.py
  ↓
Split V1                               src/splits/split_v1.py
  ↓
Future Dataset / DataLoader / Model    尚未实现或冻结
```

边界原则：

- `analysis/neck_pose_v0` 和 `analysis/neck_smoothing_v0` 是诊断/选型结果，不是正式训练标签。
- 正式 neck 标签只能来自 `features/neck_pose_v1/`。
- Fragment 层只能分组和切片现有 Speech V1 / Neck Pose V1，不得重新做 ASR、pose、滤波或 neutral normalization。
- Split 层只生成 manifest 引用，不复制或改写真实样本文件。
- Future Dataset/Model 层直接消费 Split V1；不得重新发现样本或隐式运行上游阶段。

## 4. ID 与层级关系

```text
source_video_id                    YouTube source video ID
    ↓  one-to-many
clean clip(s): <source_video_id>_<shot_index:04d>
    ↓  one-to-many
fragment(s): <clip_id>_f<fragment_index:04d>
```

例如：

```text
source_video_id = xOAq83DzDZg
clip_id          = xOAq83DzDZg_0005
fragment_id      = xOAq83DzDZg_0005_f0000
```

一个 source video 可以产生多个 clean clips；一个 clean clip 可以产生多个 fragments。因此 **57 clean clips 不等于 57 source videos**。当前 57 clips 来自 51 个 unique `source_video_id`。Split 必须在最上层 `source_video_id` 上进行。

ID 是稳定 contract：不要用 UUID 替换 fragment ID，也不要从 `fragment_id` 猜测 source split。Fragment 构建优先使用 Clean V1 `metadata.jsonl` 中的明确映射；后缀解析只是代码中的显式兼容 fallback。

## 5. 时间坐标与单位总约定

所有下游时间值单位均为 **seconds**。RPY 单位为 **radian**。

| 层 | 时间原点 | 说明 |
|---|---|---|
| 原始 source video | source-video `t=0` | Clean manifest 的 `start_time/end_time` 属于此坐标 |
| Clean / MediaPipe / Neck / Speech clip | clean clip `t=0` | downstream `timestamps`, segment/word `start/end` 均为 clip-local |
| Fragment source timeline | 所属 clean clip `t=0` | fragment 的 `source_start/source_end` 不是原始 YouTube 绝对时间 |
| Fragment local timeline | fragment audio `t=0` | word 使用 `local_start/local_end`；neck timestamp 由 clip timestamp 减 fragment `source_start` |

重要约束：

- 不要丢失 fragment 的 source timeline。
- Audio 是 16 kHz sample timeline；neck 通常按原 clip FPS 采样。二者不是同长度数组，必须按 timestamp 对齐。
- Fragment 的首个 `neck_timestamps.npy` 值可能略大于 0，因为它来自原视频帧网格；不要假定必定精确为 0。
- Failed visual frames仍占据原 frame index/timestamp；禁止通过删帧压缩时间轴。

## 6. Discovery / Validation / Download

### 6.1 Discovery

入口：`src/crawler/discover.py`；默认 query 和 title keyword 均为“主播说联播”。

输出：`metadata/discovery/candidates.jsonl`。每行字段：

```text
id, title, url, channel, channel_id, duration,
view_count, upload_date, search_query
```

按 video ID 去重；缺 ID 或标题不含 keyword 的结果不进入 candidates。

### 6.2 Validation

入口：`src/crawler/validation.py`。通过 yt-dlp 完整 metadata 验证标题、allowed channel、非直播、正 duration、公开可访问性和可下载视频 format。

输出：

- `metadata/validation/validated.jsonl`
- `metadata/validation/rejected.jsonl`
- `metadata/validation/validation_preview.txt`

validated 记录主要字段：

```text
id, title, url, channel, channel_id, upload_date, duration,
description, view_count, availability, live_status, validated
```

### 6.3 Download

入口：`src/crawler/downloader.py`。yt-dlp 选择不高于 1080p 的 best video/audio 并合并/remux 为 MP4，不在此阶段做训练预处理。

输出：

- `videos/<video_id>.mp4`
- `metadata/videos.jsonl`：`id/title/url/channel/upload_date/duration/local_path/downloaded`
- `metadata/download_failed.jsonl`
- `download_archive.txt`

这些 crawler contracts 未经明确要求不要更改，也不要自动重新访问网络。

## 7. Clean V1

### 7.1 目标和算法

输入：`videos/*.mp4`。输出训练候选 presenter shots：

```text
datasets/zhubo_shuo_lianbo/clean_v1/
├── clips/<clip_id>.mp4
├── debug/<clip_id>.jpg
├── metadata.jsonl
└── rejected.jsonl
```

当前冻结行为：

- PySceneDetect `ContentDetector(threshold=30.0)` 做 scene detection。
- shot 少于 `2.0 s` 拒绝。
- 以 `3 FPS` 抽样做 QA。
- YOLO pose 检查单人；YuNet 检查单脸。
- `single_person_ratio >= 0.90`。
- `single_face_ratio >= 0.90`。
- `shoulder_visible_ratio >= 0.80`。
- median `face_size_ratio >= 0.15`。
- 每个 shot 计算一次固定 4:5 crop；不是逐帧移动 crop。
- 保留原视频 FPS，不做人为降帧。
- FFmpeg `libx264`, preset `medium`, CRF `18` 输出 H.264 clean clip；存在音轨时直接 copy。

拒绝原因包括：`too_short`, `no_person`, `multiple_people`, `no_face`, `multiple_faces`, `face_too_small`, `shoulders_not_visible`, `unstable_detection`, `crop_invalid`, `export_failed`。

### 7.2 Clean manifest schema

`clean_v1/metadata.jsonl` 每行是一个 accepted clip：

```text
clip_id, source_video_id, source_path,
shot_index, start_time, end_time, duration,
single_person_ratio, single_face_ratio,
shoulder_visible_ratio, face_size_ratio, sampled_frames,
crop: {x, y, w, h},
output_path
```

`start_time/end_time` 是原始 source-video timeline。Clean clip 本身重新以 `t=0` 开始。Clean V1 当前 manifest 没有 `feature_version` 字段；不要虚构该字段或把缺少它视为数据损坏。

### 7.3 当前规模

- 100 downloaded MP4。
- 57 accepted clean clips。
- 94 rejected shot records。
- accepted clips 对应 51 个 unique source videos。

## 8. MediaPipe V1

### 8.1 输入输出

输入：

```text
clean_v1/clips/<clip_id>.mp4
```

输出：

```text
datasets/zhubo_shuo_lianbo/features/mediapipe_v1/
├── clips/<clip_id>/
│   ├── timestamps.npy
│   ├── valid.npy
│   ├── landmarks.npy
│   ├── blendshapes.npy
│   ├── transforms.npy
│   └── metadata.json
├── metadata.jsonl
└── failed.jsonl
```

使用 MediaPipe Tasks Face Landmarker，`num_faces=1`，开启 blendshape 和 facial transformation matrix 输出。model 位于 `models/mediapipe/face_landmarker.task`。

### 8.2 Array contract

| 文件 | shape | dtype | 含义 |
|---|---:|---|---|
| `timestamps.npy` | `[T]` | `float64` | seconds，clip-local，严格递增 |
| `valid.npy` | `[T]` | `bool` | 该帧是否有完整且一致的单脸结果 |
| `landmarks.npy` | `[T,478,3]` | `float32` | 原始 MediaPipe normalized x/y + relative z |
| `blendshapes.npy` | `[T,52]` | `float32` | 按 metadata 中 `blendshape_names` 排序 |
| `transforms.npy` | `[T,4,4]` | `float32` | MediaPipe API 原生 facial transform matrix |

**不要**为了适配论文中的 `3×4` 或 12D 描述修改 `transforms.npy`。当前真实 contract 是完整 `4×4`。

Failed frame contract：保留 timestamp 和 frame index，`valid=False`，三个 feature arrays 对应位置为 `NaN`。禁止静默删除 failed frame、前向填充或压缩 timeline。

`metadata.json` 记录 `clip_id/source_video_id/source_path/clip_path/fps/frame_count/duration/valid counts/landmark_count/blendshape_names/transform_shape/model/feature_version/status`。当前 `feature_version = mediapipe_v1`。

## 9. Neck Pose V1 — 正式 Ground Truth

路径：

```text
datasets/zhubo_shuo_lianbo/features/neck_pose_v1/
├── clips/<clip_id>/
│   ├── timestamps.npy
│   ├── valid.npy
│   ├── relative_rotation.npy
│   ├── relative_rpy_raw.npy
│   ├── neck_rpy.npy
│   └── metadata.json
├── metadata.jsonl
└── failed.jsonl
```

### 9.1 精确计算流程

实际代码顺序如下：

```text
MediaPipe transform M(t)
  ↓
A(t) = M(t)[:3, :3]
  ↓
SVD nearest-SO(3) projection → R(t)
  ↓
对内部、两端 valid、长度 <= 0.20 s 的 rotation gap 做 quaternion SLERP
  ↓
R0 = 第一个 source-valid MediaPipe frame 的 R
  ↓
R_rel(t) = R0.T @ R(t)
  ↓
ZYX Euler candidate [roll_x, pitch_y, yaw_z]
  ↓
human semantic axis mapping [roll, pitch, yaw]
  ↓
每个 final-valid contiguous run 内做 1.5 Hz Butterworth zero-phase filtering
  ↓
neck_rpy.npy
```

`R0` 是当前 **Clean clip** 的第一个 valid MediaPipe frame；若 frame 0 invalid，则使用 first valid frame。Relative rotation 必须通过矩阵乘法 `R0.T @ R(t)` 获得，**禁止 Euler angle subtraction**。

### 9.2 冻结的 semantic RPY mapping

Candidate convention：

```text
R = Rz(yaw_z) Ry(pitch_y) Rx(roll_x)
candidate = [roll_x, pitch_y, yaw_z]
```

Human semantic mapping：

```text
human_roll  = candidate_yaw_z
human_pitch = candidate_roll_x
human_yaw   = candidate_pitch_y

neck_rpy[:, 0] = roll
neck_rpy[:, 1] = pitch
neck_rpy[:, 2] = yaw
```

- 单位：`radian`
- axis order：`[roll, pitch, yaw]`
- 当前 mapping 没有 sign flip。

不要自行改轴、重排或 sign flip。机器人电机 convention 属于后续 robot mapping，不属于 Dataset GT 定义。

### 9.3 Invalid gap 与 smoothing contract

短 gap 必须同时满足：

- internal gap，不在序列两端；
- 两端都有 valid rotation；
- duration `<= 0.20 s`。

满足时使用 rotation-space shortest-path quaternion SLERP。长 gap 或边缘 gap 保持 `NaN`、`valid=False`，禁止填充。

Smoothing 设置冻结为：

```text
Butterworth low-pass
order = 4
cutoff = 1.5 Hz
zero-phase = true
implementation = scipy.signal.sosfiltfilt
```

滤波按每个 final-valid contiguous run 独立执行；短到不足 `sosfiltfilt` padding 的 run 保持未滤波并在 metadata 计数。1.5 Hz 来自 1.0/1.5/2.0 Hz 诊断比较，是当前平滑与动作细节的折中。未来机器人执行若需要更平滑，应在 robot/model postprocess 实现，不能回改训练 GT。

### 9.4 Array contract

| 文件 | shape | dtype | 含义 |
|---|---:|---|---|
| `timestamps.npy` | `[T]` | `float64` | 原 MediaPipe clip-local timestamps |
| `valid.npy` | `[T]` | `bool` | SLERP 后 final-valid mask |
| `relative_rotation.npy` | `[T,3,3]` | `float32` | `R0.T @ R(t)`；invalid 为 NaN |
| `relative_rpy_raw.npy` | `[T,3]` | `float32` | semantic mapping 后、低通前 `[roll,pitch,yaw]` |
| `neck_rpy.npy` | `[T,3]` | `float32` | 正式 filtered GT，rad；invalid 为 NaN |

`metadata.json` 记录 neutral frame、projection、relative formula、Euler convention、semantic mapping、filter/interpolation 参数、valid statistics 和 quality statistics。`feature_version = neck_pose_v1`。

## 10. Speech V1

### 10.1 Audio contract

输入：Clean clip。输出：

```text
datasets/zhubo_shuo_lianbo/features/speech_v1/
├── clips/<clip_id>/
│   ├── audio.wav
│   ├── transcript.json
│   └── metadata.json
├── metadata.jsonl
└── failed.jsonl
```

FFmpeg audio contract：

```text
sample_rate       = 16000 Hz
channels          = 1 (mono)
sample width      = signed 16-bit
codec/format      = pcm_s16le WAV
timeline          = clip-local t=0
```

`asetpts=PTS-STARTPTS` 只重置 clip-local audio PTS。不做 loudness normalization、denoise、VAD trim、silence trim、speed change。不要把 WAV 转成 MP3/AAC。Audio 与 video duration 容差为 0.10 s。

### 10.2 ASR contract

引擎是 **faster-whisper / CTranslate2**，不是 PyTorch Whisper pipeline。当前真实全量运行配置：

```text
model           = /home/jhl/projects/Project-Neck/dataset/models/faster-whisper-large-v3
language        = zh
task            = transcribe
beam_size       = 5
word_timestamps = true
vad_filter      = false
device          = cuda
compute_type    = float16
```

代码 CLI 的模型默认值仍是名称 `large-v3`；当前 artifacts 的 `metadata.json` 明确记录实际使用了本地模型路径。读取 artifact 时以其 metadata 为准。

### 10.3 Transcript schema

`transcript.json`：

```text
clip_id
language = "zh"
text                         full text
segments[]:
  id
  start, end                 seconds, clip-local
  text
  words[]:
    text
    start, end               seconds, clip-local
    probability              存在有效值时保存
words[]                      全局扁平 word list
  text
  start, end
  probability                可选
  segment_id
[detected_language]
[language_probability]
```

当前中文 Whisper transcript 已观察到少量 recognition errors、repetition、missing punctuation。Whisper segment 是声学/解码 segment，不等于严格书面句子。Fragment V1.2.1 不负责纠错，**禁止在 Fragment 阶段用 LLM 自动改写文本**。未来 ASR QA、text cleanup、punctuation restoration 必须作为独立、有版本的 stage，不能静默修改 Speech V1。

## 11. Fragment V1.2.1 — Canonical Sample Construction

### 11.1 定义

正式 sample 定义：

```text
fragment_type = spoken_utterance
feature_version = fragment_v1_2_1
```

Fragment 是 **variable-length spoken utterance**，不是固定 5 s/10 s window，也不是严格 linguistic sentence。设计优先级：

```text
natural speech boundary
> Whisper segment boundary
> duration safety fallback
```

正常情况下，一个非空 Whisper segment 是 minimum atomic speech unit，禁止 segment 内切分；多个 segments 可以组成一个 fragment。`segment != fragment`。

Speech core：

```text
first included valid word.start
→ last included valid word.end
```

Fragment 的 `source_start/source_end` 还可能包含 context padding，所以 `duration` 不一定等于纯 speech-core duration。

### 11.2 冻结参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `min_duration` | `2.0 s` | 短 utterance 优先向后 merge；不是无例外硬下限 |
| `strong_pause` | `0.45 s` | 正常 segment 间 natural boundary threshold |
| `soft_max_duration` | `20.0 s` | 开始 lookahead 的软阈值；**不是硬切点** |
| `absolute_max_duration` | `30.0 s` | 完整 segment grouping 的 safety limit |
| short residual long silence | `2.0 s` | 小于 2 s 时若下一 gap 已达此值，允许 short residual |
| `long_split_pause` | `0.35 s` | 仅用于单个 oversized segment 的内部 word fallback candidate |
| `max_context_padding` | `0.25 s`/side | 从相邻空白分配的最大上下文；按 half-gap 限制 |
| oversized split target | `18.5 s` | 单 segment >30 s 时内部 fallback 的目标附近 |
| absolute fallback target | `24.5 s` | 无自然边界时 segment-boundary fallback 的排序目标 |

当前 CLI 的 `--hard-max-duration`/`--max-duration` 是 `--soft-max-duration` 的 deprecated alias，不代表真实 hard cutoff。真实 safety limit 是 `--absolute-max-duration=30`。

### 11.3 Grouping 规则

- 若 current speech core `>=2 s` 且 next segment gap `>=0.45 s`：以 `strong_pause` 结束，`natural_boundary=true`。
- 若 current `<2 s`：优先继续 merge。
- 但若下一 gap `>=2.0 s`：允许结束为 short residual；它会标记 `duration_exception=true`、`duration_exception_reason=short_residual`。因此真实 summary 中存在 `<2 s` fragments，这是 contract 允许的例外，不要误删。
- 跨过 20 s 不会自动切分；继续 lookahead 并保留完整 Whisper segments。
- 若 20–30 s 内遇到 strong pause，仍为 `split_reason=strong_pause`、`natural_boundary=true`。
- `clip_end` 也是 natural boundary。
- 只有下一完整 segment 会使 utterance 超过 30 s 且找不到 natural boundary 时，才启用 absolute fallback。

### 11.4 Absolute / oversized fallback

普通 absolute fallback **只能选择 Whisper segment boundary**，优先级概念上为：

```text
largest historical inter-segment gap
→ punctuation boundary
→ ordinary segment boundary
```

对应常见 `split_reason`：

```text
absolute_max_pause_fallback
absolute_max_punctuation_fallback
absolute_max_segment_fallback
```

普通情况下禁止 word-level split。只有单个 Whisper segment 自身超过 30 s，才允许在该 segment 内按 word pause、punctuation、最后 ordinary word boundary fallback，产生 `oversized_segment_*` reason。当前全量数据的 `oversized_segment_word_fallback_count = 0`。

### 11.5 Context、Audio 与 Neck slicing

Grouping 后，代码最多从前后空白各取 `0.25 s` context，并以相邻 core gap 的一半为上限；context 不得使 fragment 超过 30 s。边界向外 snap 到 16 kHz PCM sample grid。

Fragment audio：从 Speech V1 WAV 按 sample index 精确 slicing，保持 16 kHz mono PCM s16；不 normalize、不 resample、不重新编码。

Fragment neck：按 Neck Pose V1 的真实 timestamps 选择区间并复制：

```text
neck_rpy.npy          unchanged Neck Pose V1 slice
neck_valid.npy        corresponding final-valid slice
neck_timestamps.npy   source neck timestamps - fragment source_start
```

除 clip end 特例外使用 `[source_start, source_end)`；clip end 可包含最后一帧。Fragment 阶段严禁：

- 重新 first-frame zero；
- 重新 neutral normalization；
- Euler subtraction；
- 重新 Butterworth；
- 重新 SLERP/interpolation；
- 重新 resample。

Fragment `neck_rpy` 继续继承整个 Clean clip 的 first-valid-frame reference，而不是 fragment-relative reference。

### 11.6 Fragment 时间和 transcript schema

每个 fragment 同时保存 clean-clip source timeline 与 fragment-local timeline：

```text
fragment source_start = 37.42
fragment source_end   = 44.81
fragment audio        = 0 ... 7.39 s

word:
  source_start/source_end
  local_start/local_end = source time - fragment source_start
```

`fragments/<fragment_id>/transcript.json`：

```text
fragment_id, clip_id, text,
fragment_type, segment_count, segment_ids,
source_start, source_end,
words[]:
  text
  local_start, local_end
  source_start, source_end
  probability       可选
  segment_id
```

不要用 local time 覆盖 source time，也不要把 context 边界误认为 speech core；`metadata.json` 中另有 `speech_start/speech_end`。

### 11.7 输出目录与 per-fragment schema

```text
datasets/zhubo_shuo_lianbo/fragments/fragment_v1/
├── fragments/<fragment_id>/
│   ├── audio.wav
│   ├── neck_rpy.npy             [Tf,3], float32, rad
│   ├── neck_timestamps.npy      [Tf], float64, fragment-local
│   ├── neck_valid.npy           [Tf], bool
│   ├── transcript.json
│   └── metadata.json
├── fragments.jsonl
├── clips.jsonl
├── failed.jsonl
└── summary.json
```

`metadata.json` 主要字段：

```text
fragment_id, clip_id, source_video_id, source_video_id_method,
source_start, source_end, speech_start, speech_end, duration,
fragment_type, segment_count, segment_ids, word_count,
boundary: {start_reason, end_reason, split_reason},
duration_exception, [duration_exception_reason],
natural_boundary, crossed_soft_threshold,
audio: {sample_rate, channels, sample_width_bits, format,
        sample_count, duration, source_start_sample, source_end_sample},
neck: {frame_count, valid_frames, valid_ratio,
       source_start_index, source_end_index_exclusive,
       one_frame_tolerance_seconds, slicing_interval, values},
quality_warning, [alignment_warnings], text,
feature_version, status
```

### 11.8 Manifest contracts

`fragments.jsonl` 每行一个 canonical sample：

```text
fragment_id, clip_id, source_video_id,
source_start, source_end, duration, text,
fragment_type, segment_count, segment_ids, word_count,
audio_path, neck_rpy_path, transcript_path,
neck_frame_count, neck_valid_ratio,
start_reason, end_reason, split_reason,
natural_boundary, crossed_soft_threshold,
duration_exception, duration_exception_reason,
quality_warning, feature_version, status
```

`clips.jsonl` 每行一个 clip 汇总：

```text
clip_id, source_video_id, source_video_id_method,
source_duration, fragment_count, fragment_ids,
total_fragment_duration, fragment_type,
segment_count, word_count,
normal_fragment_count, short_exception_count,
oversized_segment_word_fallback_count,
crossed_soft_threshold_count,
natural_boundary_count, fallback_boundary_count,
quality_warning_count, feature_version, status
```

注意 manifest 中 artifact paths 当前是绝对路径。不要无声重写路径；仓库搬迁需要显式 migration/versioning 决策。

### 11.9 当前 Fragment 全量规模

来自当前 `fragment_v1/summary.json`：

```text
completed clean clips       57
failed clips                 0
fragments                  471
total duration    5593.1379375 s
total hours          1.5536494271 h
median duration           10.31 s
min / max             0.8 / 29.7700625 s
natural boundaries          444 / 471 (94.2675%)
absolute-max fallback        27
```

## 12. Split V1

### 12.1 正式定义

目录：

```text
datasets/zhubo_shuo_lianbo/splits/split_v1/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── source_split.json
└── summary.json
```

目标比例与 seed：

```text
Train = 0.70
Test  = 0.20
Val   = 0.10
seed  = 42
split unit = source_video_id
feature_version = split_v1
```

算法先排序所有 unique source IDs，再用 `random.Random(42)` deterministic shuffle。整数数量按 largest remainder 分配，并列 remainder 按 `train → test → val`。N=51 时得到 `36/10/5`。

每个 split JSONL 保留原 fragment manifest 全部字段，额外增加：

```json
{"split": "train"}
```

行稳定排序为：`source_video_id`, `clip_id`, `source_start`, `fragment_id`。这些 JSONL 只引用 Fragment V1.2.1，不复制 audio、NPY、transcript 或 fragment directory。

### 12.2 当前真实结果

| split | source videos | clean clips | fragments | duration (s) | duration (h) |
|---|---:|---:|---:|---:|---:|
| Train | 36 | 37 | 314 | 3831.6205625 | 1.0643390451 |
| Test | 10 | 15 | 115 | 1308.7569375 | 0.3635435938 |
| Val | 5 | 5 | 42 | 452.7604375 | 0.1257667882 |
| **Total** | **51** | **57** | **471** | **5593.1379375** | **1.5536494271** |

实际比例：

| basis | Train | Test | Val |
|---|---:|---:|---:|
| source | 70.5882% | 19.6078% | 9.8039% |
| fragment | 66.6667% | 24.4161% | 8.9172% |
| duration | 68.5057% | 23.3993% | 8.0949% |

Grouped split 后 fragment/duration 不精确等于 70/20/10 是正常现象。禁止为了比例“更漂亮”破坏 source grouping。

`source_split.json` 是 `source_video_id → split` 的唯一权威映射，同时记录 seed、ratios 和各 split source lists。不要从行数或 fragment ID 重新推断映射。

### 12.3 Speaker 范围

Split V1 **不是 speaker-disjoint**。当前评估目标是：

```text
unseen source video / unseen speech content generalization
```

如果以后研究 unseen-speaker generalization，必须新增 `split_v2`，不得覆盖或修改 Split V1。

## 13. DATA LEAKAGE RULES — 不可违反

1. **永远不要 random fragment split。**
2. Split V1 的最小分组单位必须是 `source_video_id`。
3. 同一个 source video 下的所有 clean clips 和所有 fragments 必须进入同一个 split。
4. 不要重新随机生成 split 来替代 seed=42 的现有 Split V1。
5. Baseline 实验必须记录使用的 split version；当前应记录 `split_v1`。
6. 如果未来需要 speaker-disjoint，新增 Split V2；不得覆盖 V1。
7. 任何 Dataset、sampler、subsampling、cache 或 distributed loader 都不得跨 split 引入样本。
8. 训练统计量（例如 normalization mean/std）只能从 Train 估计；不得用 Val/Test 拟合 preprocessing 参数。

当前实际自动检查结果：

```text
source_overlap    = false
fragment_overlap  = false
missing_fragments = 0
```

## 14. Frozen Dataset Contracts

以下版本已经通过真实数据全量运行/验证，除非用户明确要求重新设计某层，否则视为冻结：

| Layer | Version / contract | 不可“顺手优化”的内容 |
|---|---|---|
| Clean | Clean V1 | scene/person/face/shoulder/scale 规则、3 FPS QA、固定 4:5 crop、原 FPS、H.264 参数 |
| Visual | `mediapipe_v1` | 帧对齐、NaN/valid 语义、478 landmarks、52 blendshapes、原生 `[T,4,4]` transform |
| Motion GT | `neck_pose_v1` | `R0.T @ R(t)`、first-valid neutral、RPY semantic axes/sign、0.20 s SLERP、1.5 Hz filter |
| Speech | `speech_v1` | 16 kHz mono PCM s16、clip-local timestamps、faster-whisper raw transcript contract |
| Samples | `fragment_v1_2_1` | spoken-utterance 定义、segment atomicity、2/0.45/20/30/2 s thresholds、unchanged neck slicing |
| Split | `split_v1` | source grouping、70/20/10、seed 42、现有 `source_to_split` mapping |

任何 contract 变更都应新建明确版本和输出目录，保留旧 artifact；不能原地重写并继续沿用旧 `feature_version`。

尤其禁止：

- 改 RPY axis order 或 sign；
- 把 clip neutral 改成 fragment neutral；
- 用 Euler subtraction 代替 relative rotation；
- 因机器人动作偏抖而加重 GT smoothing；
- 修改 Fragment 阈值或改成固定窗口；
- 改 seed 或重新分配现有 source mapping；
- 静默清理/改写 ASR text；
- 过滤低 `neck_valid_ratio`、短 duration 或 fallback fragments，除非用户要求独立 QA/filtering stage。

## 15. Ground Truth 与 Model Preprocessing 必须分离

Dataset Ground Truth 是冻结的 `neck_rpy.npy`。未来模型可能需要：

```text
normalization
Delta RPY / velocity / acceleration
fragment-relative representation
padding / mask
resampling for a specific architecture
loss-specific target transforms
```

这些只能在 Dataset/Model preprocessing 或训练 pipeline 中实现，并保存其训练配置。不得因此回头修改 Neck Pose V1 或 Fragment V1.2.1 artifact。

若产生 derived target：

- 保留原 `neck_rpy`, `neck_timestamps`, `neck_valid`；
- 明确记录 transform、单位、reference 和逆变换；
- normalization 统计仅由 train split 计算；
- invalid mask 与 padding mask 分开处理，不得把 NaN 当零而不提供 mask。

## 16. Future Dataset / Model 的安全消费方式

模型训练任务应以这些文件作为 canonical sample list：

```text
datasets/zhubo_shuo_lianbo/splits/split_v1/train.jsonl
datasets/zhubo_shuo_lianbo/splits/split_v1/val.jsonl
datasets/zhubo_shuo_lianbo/splits/split_v1/test.jsonl
```

不要直接扫描 `clean_v1/`、`speech_v1/`、`neck_pose_v1/` 或 `fragments/` 自行发现训练样本。每条 split row 已提供：

- `audio_path`
- `neck_rpy_path`
- `transcript_path`
- source/clip/fragment IDs
- duration 与 boundary/quality metadata
- `split`

`neck_timestamps.npy`、`neck_valid.npy` 与 `neck_rpy_path` 位于同一 fragment directory，可从 `Path(neck_rpy_path).parent` 定位。训练代码仍应验证必要文件存在和 shape/length 一致，但不得重建上游 artifact。

Fragment 是 variable-length。未来 DataLoader 应使用 padding、padding mask、neck-valid mask、bucketing 或 dynamic batching。**禁止为了 DataLoader 方便把数据集重新裁成固定 5 s/10 s fragments。** 若模型内部要求 fixed chunks，应把它定义为可复现的 model-side sampling，不得冒充新的 GT manifest，也不得跨 split。

## 17. 禁止隐式重算与数据写入规则

修改下游代码时，不要隐式运行以下 expensive stages：

```text
discover / validate / download / clean
extract-mediapipe
extract-neck-pose
extract-speech
build-fragments
build-split
```

实现 PyTorch Dataset 不应触发 MediaPipe、ASR、neck extraction、fragment build 或 split build。未经用户明确授权：

- 不修改、删除、移动 `datasets/` 中已有数据 artifact；
- 不使用 `--force` 重建冻结输出；
- 不自动下载模型或视频；
- 不修复所谓“异常”数据；
- 不把 analysis V0 当作训练标签。

可以安全做的通常是：读取 manifest/schema、写新下游模块、使用 synthetic fixtures 做 unit tests、运行 import/syntax/`--help` 检查。若任务确实要求改变已冻结层，应先说明版本迁移范围和是否需要重跑真实数据。

## 18. Research Context 与已知限制

### 18.1 One-to-many motion

同样的 speech/text 不一定对应唯一 neck motion，这是 one-to-many problem。当前 Dataset V1 只提供观测到的真实 paired supervision，不人为复制或生成多个 GT。

未来可研究 stochastic generation、multi-candidate generation、style-conditioned 或 context-conditioned generation；这些都不是当前 Dataset V1 已完成功能。

### 18.2 未来输入条件

未来 baseline 可能比较：

```text
audio only
text only
audio + text
```

更远期可能加入 context、affect、style、speaker individuality。目前都未在 Dataset V1 construction 中冻结。

### 18.3 Neck proxy 限制

当前 `neck_rpy` 是 clip-relative head orientation proxy，不是 mocap neck joint angle，也没有 torso correction。未来可能研究：

```text
R_neck = R_torso^-1 @ R_head
```

但 V1 没有 torso pose GT，不能把该方向写成已实现。

### 18.4 ASR 限制

中文 Whisper 有少量识别错误、重复和缺标点。不要在读取时静默“纠正”，否则 text 条件与 canonical artifact 不再可复现。需要清洗时新建显式 version/stage。

## 19. 当前真实进度

当前 canonical manifests 显示：

| Stage | 完成情况 |
|---|---:|
| Discovery candidates | 100 |
| Validation accepted / rejected | 100 / 0 |
| Downloaded / failed | 100 / 0 |
| Clean accepted clips / rejected shots | 57 / 94 |
| MediaPipe V1 completed / failed | 57 / 0 |
| Neck Pose V1 completed / failed | 57 / 0 |
| Speech V1 completed / failed | 57 / 0 |
| Fragment V1.2.1 clips / fragments / failed | 57 / 471 / 0 |
| Split V1 | 已生成，51 sources，471 fragments，无 leakage |

`features/speech_v1/clips/` 当前还存在一个 dot-prefixed temporary/staging residue；canonical `metadata.jsonl` 仍明确只有 57 个 completed clips。未来 Agent 不应把 dot directory 当作第 58 个样本，也不应在无授权情况下清理真实数据目录。

**Dataset construction V1 已完成。** 当前下一阶段是：

```text
PyTorch Dataset / DataLoader
+
baseline model input/output definition
```

尚未冻结：audio representation、text representation、audio+text fusion、neck target representation、normalization、padding/mask/batching、evaluation metrics。不要在本文件或未来实现中未经用户决策擅自冻结这些问题。

## 20. 当前目录树

```text
dataset/
├── AGENTS.md                         本文件：Agent 数据契约
├── .gitignore
├── README.md                         普通用户概览（可能比本文件简略）
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── youtube.yaml
│   └── mediapipe.yaml
├── models/
│   ├── mediapipe/face_landmarker.task
│   └── faster-whisper-large-v3/
├── datasets/zhubo_shuo_lianbo/
│   ├── download_archive.txt
│   ├── metadata/
│   │   ├── discovery/
│   │   ├── validation/
│   │   ├── videos.jsonl
│   │   └── download_failed.jsonl
│   ├── videos/
│   ├── clean_v1/
│   │   ├── clips/
│   │   ├── debug/
│   │   ├── metadata.jsonl
│   │   └── rejected.jsonl
│   ├── analysis/
│   │   ├── neck_pose_v0/             诊断，不是 GT
│   │   └── neck_smoothing_v0/        诊断，不是 GT
│   ├── features/
│   │   ├── mediapipe_v1/
│   │   ├── neck_pose_v1/
│   │   └── speech_v1/
│   ├── fragments/fragment_v1/
│   └── splits/split_v1/
├── src/
│   ├── cli.py
│   ├── crawler/
│   ├── cleaning/
│   ├── features/
│   ├── fragments/
│   └── splits/
└── tests/
```

`src/` 自身是顶层 Python package；项目 import 使用 `from src...`。不要额外创建 `src/project_data/` wrapper。代码默认路径应从 `Path(__file__).resolve()` 推导项目根目录，不要硬编码 `/home/jhl/...`。现有生成 manifest 中的绝对 artifact reference 已在本次目录迁移中同步到当前 checkout；schema、数据版本和内容语义未改变。

## 21. CLI 参考

从 `dataset/` 目录执行。只查看帮助是安全的；真实命令不得未经授权运行。

```bash
python -m src.cli --help

python -m src.cli discover [--query QUERY] [--limit N]
python -m src.cli validate
python -m src.cli download [--limit N]
python -m src.cli clean --input DIR [--limit N] [--force]

python -m src.cli extract-mediapipe \
  [--input DIR] [--output DIR] [--model FILE] [--limit N] [--force]

python -m src.cli validate-neck-pose \
  [--input DIR] [--video-input DIR] [--output DIR] [--limit N] [--force]

python -m src.cli compare-neck-smoothing \
  [--input DIR] [--video-input DIR] [--mediapipe-input DIR] \
  [--output DIR] [--limit N] [--force]

python -m src.cli extract-neck-pose \
  [--input DIR] [--output DIR] [--limit N] [--force]

python -m src.cli extract-speech \
  [--input DIR] [--output DIR] [--model MODEL] \
  [--device DEVICE] [--compute-type TYPE] [--limit N] [--force]

python -m src.cli build-fragments \
  [--speech-input DIR] [--neck-input DIR] [--clean-input DIR] \
  [--output DIR] [--limit N] [--force] \
  [--min-duration S] [--soft-max-duration S] \
  [--absolute-max-duration S] [--strong-pause S] \
  [--long-split-pause S] [--max-context-padding S]

python -m src.cli build-split \
  [--input DIR] [--output DIR] [--seed N] \
  [--train-ratio R] [--test-ratio R] [--val-ratio R] [--force]
```

主要默认路径：

```text
MediaPipe input/output : clean_v1/clips → features/mediapipe_v1
Neck V1 input/output   : features/mediapipe_v1/clips → features/neck_pose_v1
Speech input/output    : clean_v1/clips → features/speech_v1
Fragment inputs        : speech_v1/clips + neck_pose_v1/clips + clean_v1
Fragment output        : fragments/fragment_v1
Split input/output     : fragments/fragment_v1 → splits/split_v1
```

## 22. Future Agent 快速决策规则

任务开始时按以下顺序判断：

1. 如果是训练/评估任务，先读 `splits/split_v1/{train,val,test}.jsonl`。
2. 如果需要理解 sample，读 fragment manifest 和一个 fragment 的 metadata/transcript；不要扫描上游重新组样本。
3. 如果需要 neck semantics，以 `neck_pose_v1` metadata 和本文件 mapping 为准。
4. 如果修改仅属于 model representation，不修改 GT artifact。
5. 如果请求与冻结 contract 冲突，先向用户指出并建议新版本；不要偷偷兼容。
6. 真实数据操作、`--force`、网络下载和 expensive preprocessing 必须得到明确授权。
7. 新实验必须记录 dataset/fragment/split version，以及所有 model-side preprocessing。
