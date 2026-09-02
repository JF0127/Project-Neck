#!/usr/bin/env python
"""条件建模诊断：role-only 对比 + 条件错配测试 + 节奏对齐指标。

三种条件模式（同一 checkpoint 的 val 采样指标，按 Speaker/Listener 分开报告）：
    match         输入条件与样本一一对应（正常）
    shuffle_audio batch 内打乱音频（文本/上下文/role 保持）
    shuffle_text  batch 内打乱文本与上下文（音频/role 保持）

判读：
- 若候选池与输入无关（无条件动作先验），shuffle 后 best-of-K / 运动能量 /
  节奏对齐与 match 几乎无差异；
- 若候选由条件决定，shuffle 后 best-of-K 应变差、节奏对齐应明显下降
  （预测运动速度包络与音频能量包络的相关 → 0）。
- role-only 模型对 shuffle 应完全不敏感（sanity check）。

用法：
    python models/neck_motion/diagnose.py --checkpoint .../best.pt [--samples 8]
输出：<checkpoint 目录>/diagnose_<split>.json + 控制台报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.neck_motion.audio_features import interpolate_mel_to_frames
from models.neck_motion.cvae import build_model as build_model_any
from models.neck_motion.dataset import ResponseNetDataset, collate_neck_motion
from models.neck_motion.metrics import aggregate_metrics, candidate_metrics
from models.neck_motion.text_features import Vocab

RAD2DEG = 180.0 / np.pi


def rhythm_alignment(pred: torch.Tensor, mel_energy: torch.Tensor, n: int, smooth: int = 5) -> float:
    """预测运动速度包络与音频能量包络的 Pearson 相关（节奏对齐指标）。

    pred [n, 3]；mel_energy [T_mel]（log-Mel 帧能量）。两者插值到 n、滑动平均、
    z-score 后求相关。匹配条件应高于错配条件。
    """
    if n < 8 or mel_energy.numel() < 8:
        return 0.0
    # 速度包络：帧级速度范数
    vel = (pred[1:] - pred[:-1]).norm(dim=-1).float()  # [n-1]
    vel = torch.nn.functional.pad(vel, (0, 1))
    # 能量包络插值到 n 帧
    e = interpolate_mel_to_frames(mel_energy.unsqueeze(0).unsqueeze(0), [n])[0, 0]  # [n]

    def smooth_zscore(x: torch.Tensor, w: int) -> np.ndarray:
        x = x.float()
        if w > 1:
            kernel = torch.ones(w, dtype=x.dtype, device=x.device) / w
            x = torch.nn.functional.conv1d(x.view(1, 1, -1), kernel.view(1, 1, -1), padding=w // 2)[0, 0]
        x = x[: x.shape[0] - (x.shape[0] - n)] if x.shape[0] > n else x
        x = x[:n]
        std = x.std()
        if std < 1e-8:
            return np.zeros(n)
        return ((x - x.mean()) / std).cpu().numpy()

    v = smooth_zscore(vel, smooth)
    en = smooth_zscore(e, smooth)
    if v.std() < 1e-6 or en.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(v, en)[0, 1])


def shuffle_batch(batch: dict, key: str, gen: torch.Generator) -> dict:
    """batch 内打乱音频或文本条件（role/时长/target 保持不动）。"""
    b = dict(batch)
    B = batch["audio"].size(0)
    perm = torch.randperm(B, generator=gen)
    if key == "audio":
        b["audio"] = batch["audio"][perm]
        b["audio_lens"] = batch["audio_lens"][perm]
    elif key == "text":
        b["word_ids"] = [batch["word_ids"][i] for i in perm.tolist()]
        b["word_starts"] = [batch["word_starts"][i] for i in perm.tolist()]
        b["word_ends"] = [batch["word_ends"][i] for i in perm.tolist()]
        b["prev_token_ids"] = batch["prev_token_ids"][perm]
        b["prev_lens"] = batch["prev_lens"][perm]
    else:  # pragma: no cover
        raise ValueError(key)
    return b


@torch.no_grad()
def run_diagnose(model, loader, device, n_samples: int, seed: int = 1234) -> dict:
    records: dict[str, list[dict]] = {"match": [], "shuffle_audio": [], "shuffle_text": []}
    gen = torch.Generator().manual_seed(seed)
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        # 音频能量包络（模型内的 log-Mel，插值到每样本 N 帧）
        if model.encoder.use_audio:
            mel = model.encoder.mel(b["audio"])
        for mode in records:
            bb = b if mode == "match" else shuffle_batch(b, mode.split("_")[1], gen)
            samples = model.sample(bb, n_samples=n_samples)  # [B, K, N, 3]
            for i in range(samples.size(0)):
                n = int(bb["Ns"][i])
                m = candidate_metrics(samples[i, :, :n], b["rpy"][i, :n], bb["mask"][i, :n], b["role_names"][i])
                if m is None:
                    continue
                if model.encoder.use_audio:
                    e = mel[i].sum(dim=0)  # [T_mel]
                    m["rhythm_corr"] = rhythm_alignment(samples[i, :, :n].float().mean(dim=0), e, n)
                records[mode].append(m)
    return {mode: aggregate_metrics(rs) for mode, rs in records.items()}


def summarize(group_data: dict, n_samples: int) -> str:
    """group_data: mode -> 单个分组（overall/speaker/listener）的指标 dict。"""
    lines = []
    for mode, label in (("match", "匹配条件"), ("shuffle_audio", "音频错配"), ("shuffle_text", "文本错配")):
        m = group_data[mode]
        if not m:
            lines.append(f"[{label}] 无样本")
            continue
        rhythm = m.get("rhythm_corr")
        lines.append(f"[{label}] best-of-{n_samples} {m['best_of_k_deg']:.3f}° | "
                     f"多样性 {m['diversity_deg']:.2f}° | 幅度 {m['amp_deg']:.2f}° "
                     f"(目标 {m['target_amp_deg']:.2f}°) | vel {m['vel_deg_per_s']:.2f}°/s | "
                     f"acc {m['acc_deg_per_s2']:.0f}°/s² | "
                     f"节奏相关 {rhythm if rhythm is not None else float('nan'):+.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--true-overlap", default="outputs/neck_motion/overlap_analysis/true_overlap.json")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root or cfg["data"]["data_root"])

    vocab_path = Path(ckpt["vocab_path"])
    if not vocab_path.exists():
        vocab_path = Path(args.checkpoint).parent / vocab_path.name
    vocab = Vocab.load(vocab_path)

    ds = ResponseNetDataset(data_root, split=args.split, vocab=vocab,
                            filter_overlap_sec=cfg["data"].get("filter_overlap_sec"),
                            true_overlap_path=cfg["data"].get("true_overlap_path"),
                            target_sr=cfg["data"]["target_sr"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_neck_motion)

    model = build_model_any(cfg, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[diagnose] {args.split} n={len(ds)} | K={args.samples} | "
          f"role_only={getattr(model.encoder, 'role_only', False)}")
    out = run_diagnose(model, dl, device, args.samples, seed=args.seed)

    for group, key in (("overall", "全体"), ("speaker", "Speaker"), ("listener", "Listener")):
        print(f"=== {key} ===")
        print(summarize({mode: out[mode][group] for mode in out}, args.samples))

    out_path = Path(args.checkpoint).parent / f"diagnose_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
