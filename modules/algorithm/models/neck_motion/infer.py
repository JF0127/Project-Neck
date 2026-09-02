#!/usr/bin/env python
"""推理脚本：加载 checkpoint + 词表，对单个 fragment JSON 生成相对 RPY 轨迹。

用法：
    python models/neck_motion/infer.py --checkpoint outputs/neck_motion/checkpoints/best.pt \
        --input sample.json --data-root data/source/response-net/processed_dataset

输入 JSON（与训练样本结构兼容）：
    {
      "role": "speaker" | "listener",
      "current_utterance": {
        "text": "...", "audio_path": "audio/<uid>.wav",
        "word_timestamps": [{"text": "...", "start_time": 0.0, "end_time": 0.2}]
      },
      "previous_listener_context" | "previous_speaker_context": {"text": "..."} | null,
      "num_frames": 289,            // 可选；缺省由 duration_sec 或词时间戳推算
      "duration_sec": 9.62,         // 可选
      "fps": 30,                    // 可选，默认 30
      "robot_actual_initial": [0.0, 0.0, 0.0]   // 可选；提供则额外输出旋转矩阵
    }

输出（--output-dir 下）：
    predicted_relative_rpy.npy                [N, 3] 弧度
    predicted_robot_rotation_matrices.npy     [N, 3, 3]（仅当提供 robot_actual_initial）
        R_robot[t] = R(robot_actual_initial) @ R(pred_relative_rpy[t])
        R = Ry(yaw) @ Rx(pitch) @ Rz(roll)（数据集旋转约定）

本脚本不负责：机器人坐标系转换、关节限位、速度/加速度/jerk 限制、Silent 状态、
以及向机器人下发控制指令（由执行层完成）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.neck_motion.cvae import build_model as build_model_any
from models.neck_motion.dataset import build_inference_batch
from models.neck_motion.rotations import rpy_to_matrix as rpy_to_rotation_matrix
from models.neck_motion.text_features import Vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="颈部运动生成——推理")
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径（best.pt / last.pt）")
    p.add_argument("--input", required=True, help="输入 JSON 文件路径（与训练样本结构兼容）")
    p.add_argument("--data-root", default=None, help="processed_dataset 根目录（默认取 checkpoint 配置）")
    p.add_argument("--output-dir", default=None, help="输出目录（默认 outputs/neck_motion/infer/<时间戳>）")
    p.add_argument("--device", default=None, help="cpu/cuda/mps（默认自动选择）")
    p.add_argument("--num-candidates", type=int, default=1, help="多候选模型：先验采样候选数（>1 时保存候选池）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 词表（checkpoint 记录的路径，相对则相对 checkpoint 目录）
    vocab_path = Path(ckpt["vocab_path"])
    if not vocab_path.exists():
        vocab_path = Path(args.checkpoint).parent / vocab_path.name
    vocab = Vocab.load(vocab_path)
    print(f"[infer] 词表: {vocab_path}（大小={len(vocab)}）")

    data_root = Path(args.data_root or cfg["data"]["data_root"])
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    from models.neck_motion.dataset import build_inference_batch
    batch = build_inference_batch(data, vocab, data_root, int(cfg["data"]["target_sr"]))
    N = batch["num_frames"]
    fps = float(data.get("fps", 30.0))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = build_model_any(cfg, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        if getattr(model, "is_cvae", False):
            pred = model(batch, use_posterior=False)  # 部署：先验 p(z|h) 采样一次
        elif getattr(model, "is_candidates", False):
            preds = model(batch, n_samples=args.num_candidates)  # [1, K, N, 3]
            pred = preds[0, 0]  # 默认取第一个候选（策略选择见 mvp.py）
        else:
            pred = model(batch)
    pred = pred[0].cpu().numpy().astype(np.float32)
    if not np.isfinite(pred).all():
        raise RuntimeError("模型输出包含 NaN/Inf")
    print(f"[infer] 生成完成: N={N} 帧（{N/fps:.2f}s @ {fps:.0f}fps），耗时 {time.time()-t0:.2f}s")

    out_dir = Path(args.output_dir or (Path(cfg["train"]["output_dir"]) / "infer" / time.strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(model, "is_candidates", False) and args.num_candidates > 1:
        cand_path = out_dir / "candidates.npy"
        np.save(cand_path, preds[0].cpu().numpy().astype(np.float32))
        print(f"[infer] 候选池已保存: {cand_path}（shape={preds[0].shape}，供候选策略 A/B 测试）")

    rel_path = out_dir / "predicted_relative_rpy.npy"
    np.save(rel_path, pred)
    print(f"[infer] 已保存: {rel_path}（shape={pred.shape}，弧度）")

    if data.get("robot_actual_initial") is not None:
        init = np.asarray(data["robot_actual_initial"], dtype=np.float64)
        if init.shape != (3,):
            raise ValueError("robot_actual_initial 必须是 [roll, pitch, yaw]")
        R_robot = rpy_to_rotation_matrix(init) @ rpy_to_rotation_matrix(pred)
        mat_path = out_dir / "predicted_robot_rotation_matrices.npy"
        np.save(mat_path, R_robot)
        print(f"[infer] 已保存: {mat_path}（shape={R_robot.shape}，R_robot[t]=R(init)@R(rel[t])）")
    else:
        print("[infer] 未提供 robot_actual_initial，跳过旋转矩阵输出")


if __name__ == "__main__":
    main()
