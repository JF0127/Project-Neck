# Project-Neck Dataset

`dataset/` is Project-Neck's offline dataset production subsystem. It is separate from the robot's real-time runtime; the online system remains under `modules/audio/`, `modules/algorithm/`, and `modules/motor/`.

It is responsible for:

- video discovery and download;
- raw data management;
- data cleaning;
- fragment construction;
- ASR, audio, MediaPipe, and neck-pose feature extraction;
- dataset splits and data-quality analysis;
- providing training, validation, and test data to the Algorithm Module.

## Pipeline

```text
YouTube Discovery
→ Validation
→ Download
→ Clean V1
→ MediaPipe / Neck Pose / Speech Features
→ Fragment Construction
→ Dataset Split
```

## CLI

Run the existing commands from the `dataset/` directory:

```bash
python -m src.cli discover
python -m src.cli validate
python -m src.cli download
python -m src.cli clean --input datasets/zhubo_shuo_lianbo/videos
python -m src.cli extract-mediapipe
python -m src.cli validate-neck-pose --limit 3
python -m src.cli compare-neck-smoothing --limit 3
python -m src.cli extract-neck-pose
python -m src.cli extract-speech
python -m src.cli build-fragments
python -m src.cli build-split
```

MediaPipe V1 saves frame-aligned raw landmarks, blendshapes, facial transformation matrices, timestamps, and valid masks under `features/mediapipe_v1/`. `validate-neck-pose` creates experimental ZYX RPY curves and debug videos under `analysis/neck_pose_v0/`; these are diagnostic outputs, not final labels. `compare-neck-smoothing` compares raw candidate Euler tracks with 1.0/1.5/2.0 Hz zero-phase Butterworth results under `analysis/neck_smoothing_v0/`. Formal baseline neck labels are generated separately under `features/neck_pose_v1/` by `extract-neck-pose`. Speech V1 uses `faster-whisper` (default `large-v3`) to produce clip-local Chinese segment and raw word timestamps under `features/speech_v1/`. Download the official [Face Landmarker task model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) to `models/mediapipe/face_landmarker.task`, or pass another location with `--model`.

Use `--help` after a command to view its existing options. Configuration files are in `configs/`. Speech extraction requires system FFmpeg/ffprobe; verify them with `ffmpeg -version` and `ffprobe -version`. The first `large-v3` run may download the faster-whisper model; `--model`, `--device`, and `--compute-type` can override ASR settings.

## Data

The concrete dataset is stored in `datasets/zhubo_shuo_lianbo/` relative to this project (from the repository root: `dataset/datasets/zhubo_shuo_lianbo/`). Discovery and validation records are under `datasets/zhubo_shuo_lianbo/metadata/`; downloaded videos are under `videos/`, Clean V1 outputs are under `clean_v1/`, and derived MediaPipe V1 features are under `features/mediapipe_v1/`. The existing `download_archive.txt` is preserved alongside the dataset metadata and videos.
