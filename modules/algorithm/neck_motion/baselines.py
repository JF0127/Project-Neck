#!/usr/bin/env python
"""基线指标：全零轨迹、仅 role 均值轨迹（在修正过滤后的 val 上评估）。

基线 1 全零：pred = 0（模型未学到任何信号时的下界参考）。
基线 2 仅 role：从 train 集按 role 统计归一化时间轴（0..1 → 180 帧）的平均相对
RPY 轨迹；对 val 每个样本插值回其 N 帧。预测只依赖 role 与时长，不含音频/文本。

用法：
    python neck_motion/baselines.py
    python neck_motion/baselines.py --threshold 0.1

输出：outputs/neck_motion/overlap_analysis/baselines.json + 控制台报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neck_motion.dataset import ROLE_TO_ID
from neck_motion.overlap_filter import load_fragments, load_true_overlap

UNIFORM_LEN = 180  # 归一化时间轴上的采样帧数


def interpolate_to_len(x: np.ndarray, n: int) -> np.ndarray:
    """x [L, 3] -> 线性插值到 [n, 3]。"""
    if n == x.shape[0]:
        return x.copy()
    t_in = np.linspace(0.0, 1.0, x.shape[0])
    t_out = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(3)], axis=-1)


def load_split_samples(data_root: Path, split: str, true_map: dict[str, float], thr: float) -> list[dict]:
    """读 split jsonl，应用 true_overlap 过滤（与 dataset.py 同一判据）。"""
    samples = []
    with open(data_root / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            uid = s["sample_id"].rsplit("_", 1)[0]
            if true_map.get(uid, 0.0) <= thr:
                samples.append(s)
    return samples


def load_target(data_root: Path, s: dict) -> np.ndarray:
    return np.load(str(data_root / s["target_rpy_path"])).astype(np.float32)


def compute_role_mean_trajectory(data_root: Path, samples: list[dict]) -> dict[str, np.ndarray]:
    """train 集按 role 统计归一化时间轴平均轨迹 [L, 3] 弧度。"""
    acc = {role: np.zeros((UNIFORM_LEN, 3)) for role in ("speaker", "listener")}
    cnt = {role: 0 for role in ("speaker", "listener")}
    for s in samples:
        rpy = load_target(data_root, s)
        if rpy.shape[0] < 2:
            continue
        acc[s["role"]] += interpolate_to_len(rpy, UNIFORM_LEN)
        cnt[s["role"]] += 1
    return {role: acc[role] / max(cnt[role], 1) for role in ("speaker", "listener")}


def eval_baseline(name: str, pred_fn, samples, data_root) -> dict:
    mae, rmse, vel, acc = [], [], [], []
    for s in samples:
        t = load_target(data_root, s)
        p = pred_fn(s, t.shape[0])
        d = p - t
        mae.append(np.abs(d).mean())
        rmse.append(np.sqrt((d ** 2).mean()))
        if len(t) >= 2:
            vel.append(np.abs((p[1:] - p[:-1]) - (t[1:] - t[:-1])).mean())
        if len(t) >= 3:
            pa = p[2:] - 2 * p[1:-1] + p[:-2]
            ta = t[2:] - 2 * t[1:-1] + t[:-2]
            acc.append(np.abs(pa - ta).mean())
    rad2deg = 180.0 / np.pi
    return {
        "name": name,
        "samples": len(samples),
        "mae_rad": float(np.mean(mae)),
        "mae_deg": float(np.mean(mae) * rad2deg),
        "rmse_rad": float(np.mean(rmse)),
        "rmse_deg": float(np.mean(rmse) * rad2deg),
        "vel_mae_deg_per_s": float(np.mean(vel) * rad2deg * 30),
        "acc_mae_deg_per_s2": float(np.mean(acc) * rad2deg * 30 * 30),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/source/response-net/processed_dataset")
    ap.add_argument("--true-overlap", default="outputs/neck_motion/overlap_analysis/true_overlap.json")
    ap.add_argument("--output", default="outputs/neck_motion/overlap_analysis/baselines.json")
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    true_map = load_true_overlap(args.true_overlap)
    thr = args.threshold

    train_samples = load_split_samples(data_root, "train", true_map, thr)
    val_samples = load_split_samples(data_root, "val", true_map, thr)
    print(f"[baselines] 过滤后 train={len(train_samples)} 条 | val={len(val_samples)} 条（阈值 {thr}s）")

    role_mean = compute_role_mean_trajectory(data_root, train_samples)

    results = []
    # 基线 1：全零
    results.append(eval_baseline("zero", lambda s, n: np.zeros((n, 3), dtype=np.float32),
                                 val_samples, data_root))
    # 基线 2：仅 role 均值轨迹（时间归一化）
    results.append(eval_baseline(
        "role_mean", lambda s, n: interpolate_to_len(role_mean[s["role"]], n).astype(np.float32),
        val_samples, data_root))
    # 基线 3：不分 role 的全局均值轨迹（参考）
    global_mean = 0.5 * (role_mean["speaker"] + role_mean["listener"])
    results.append(eval_baseline(
        "global_mean", lambda s, n: interpolate_to_len(global_mean, n).astype(np.float32),
        val_samples, data_root))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "results": results,
                   "role_mean_speaker": role_mean["speaker"].tolist(),
                   "role_mean_listener": role_mean["listener"].tolist()},
                  f, ensure_ascii=False, indent=1)

    print(f"\n{'基线':<12}{'MAE°':>10}{'RMSE°':>10}{'vel°/s':>10}{'acc°/s²':>10}")
    for r in results:
        print(f"{r['name']:<12}{r['mae_deg']:>10.3f}{r['rmse_deg']:>10.3f}"
              f"{r['vel_mae_deg_per_s']:>10.3f}{r['acc_mae_deg_per_s2']:>10.3f}")
    print(f"\n已保存: {args.output}")


if __name__ == "__main__":
    main()
