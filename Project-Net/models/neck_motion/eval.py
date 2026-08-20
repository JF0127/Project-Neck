#!/usr/bin/env python
"""用 checkpoint 在 val/test 上评估。

回归模型：完整指标 + 轨迹诊断（|pred| / |t| / corr）。
CVAE 模型：后验路径 loss + 先验采样指标：
    - 多次采样的预测幅度 vs 目标幅度分布（是否脱离零退化）
    - 采样两两距离（多样性）
    - best-of-K 轨迹误差（覆盖度，oracle 指标，非部署期望）
    - 单样本平滑度（帧间速度/加速度）与关节安全范围（最大幅度分位）
    - 输出 json 到 <checkpoint 目录>/eval_<split>.json

用法：
    python models/neck_motion/eval.py --checkpoint .../best.pt [--split val] [--samples 10]
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

from models.neck_motion.cvae import build_model as build_model_any
from models.neck_motion.dataset import ResponseNetDataset, collate_neck_motion
from models.neck_motion.losses import NeckMotionLoss
from models.neck_motion.metrics import aggregate_metrics, format_metrics, sample_metrics
from models.neck_motion.text_features import Vocab

RAD2DEG = 180.0 / np.pi


def eval_regression(model, loader, loss_fn, device) -> dict:
    records = []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            pred = model(b)
            for i in range(pred.size(0)):
                m = sample_metrics(pred[i], b["rpy"][i], b["mask"][i], b["role_names"][i])
                if m is not None:
                    records.append(m)
    return aggregate_metrics(records)


@torch.no_grad()
def eval_sampling(model, loader, loss_fn, device, n_samples: int = 10, has_posterior: bool = False) -> dict:
    """采样式模型评估（CVAE / 多候选）：先验采样指标；CVAE 额外报后验路径 loss。"""
    records: list[dict] = []
    posterior_loss = 0.0
    n_samples_seen = 0

    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        if has_posterior:
            pred_post, aux = model(b, return_aux=True, use_posterior=True)
            total, _ = loss_fn(pred_post, b["rpy"], b["delta"], b["mask"],
                               kl=aux["kl"], beta_kl=1.0)
            posterior_loss += float(total) * pred_post.size(0)
            n_samples_seen += pred_post.size(0)

        # 先验采样 K 次（与部署一致）
        samples = model.sample(b, n_samples=n_samples)  # [B, K, Nmax, 3]
        B, K = samples.shape[0], samples.shape[1]
        for i in range(B):
            n = int(b["Ns"][i])
            s = samples[i, :, :n, :].float()          # [K, n, 3]
            t = b["rpy"][i, :n, :].float()            # [n, 3]
            role = b["role_names"][i]

            # 幅度：K 次采样的平均幅度（跨帧 mean），及其跨采样标准差
            amp_k = s.abs().mean(dim=(1, 2))          # [K]
            amp = float(amp_k.mean())
            amp_std_across_k = float(amp_k.std())     # 同一输入下采样的幅度波动

            # 多样性：K 条轨迹两两 MAE 距离
            d = s.unsqueeze(1) - s.unsqueeze(0)       # [K, K, n, 3]
            pair_mae = d.abs().mean(dim=(2, 3))       # [K, K]
            iu = torch.triu_indices(K, K, offset=1)
            diversity = float(pair_mae[iu[0], iu[1]].mean())

            # best-of-K（oracle 覆盖度）
            mae_k = (s - t.unsqueeze(0)).abs().mean(dim=(1, 2))  # [K]
            best_of_k = float(mae_k.min())

            # 平滑度与安全范围（取 K 次采样的平均轨迹质量）
            d1 = (s[:, 1:] - s[:, :-1]).abs().mean(dim=(1, 2))   # rad/帧 [K]
            d2 = (s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]).abs().mean(dim=(1, 2))
            max_amp = s.abs().amax(dim=(1, 2))                    # rad [K]

            records.append({
                "role": role, "frames": n,
                "amp_deg": amp * RAD2DEG,
                "amp_std_across_k_deg": amp_std_across_k * RAD2DEG,
                "target_amp_deg": float(t.abs().mean()) * RAD2DEG,
                "diversity_deg": diversity * RAD2DEG,
                "best_of_k_deg": best_of_k * RAD2DEG,
                "vel_deg_per_s": float(d1.mean()) * RAD2DEG * 30.0,
                "vel_p95_deg_per_s": float(torch.quantile(d1, 0.95)) * RAD2DEG * 30.0,
                "acc_deg_per_s2": float(d2.mean()) * RAD2DEG * 900.0,
                "max_amp_p95_deg": float(torch.quantile(max_amp, 0.95)) * RAD2DEG,
                "target_max_amp_p95_deg": float(torch.quantile(t.abs().amax(dim=1), 0.95)) * RAD2DEG,
            })

    agg = aggregate_metrics(records)
    if has_posterior:
        agg["overall"]["posterior_val_loss"] = posterior_loss / max(n_samples_seen, 1)
    return agg


def _summarize(agg: dict, n_samples: int) -> str:
    lines = [f"[CVAE 先验采样 K={n_samples}]"]
    for group, key in (("overall", "全体"), ("speaker", "Speaker"), ("listener", "Listener")):
        m = agg.get(group)
        if not m:
            lines.append(f"[{key}] 无样本")
            continue
        lines.append(
            f"[{key}] n={m['samples']} | 幅度 {m['amp_deg']:.2f}° (目标 {m['target_amp_deg']:.2f}°, "
            f"跨采样σ {m['amp_std_across_k_deg']:.2f}°) | 多样性 {m['diversity_deg']:.2f}° | "
            f"best-of-{n_samples} {m['best_of_k_deg']:.3f}° | vel {m['vel_deg_per_s']:.2f}°/s "
            f"(P95 {m['vel_p95_deg_per_s']:.1f}) | acc {m['acc_deg_per_s2']:.0f}°/s² | "
            f"max幅度P95 {m['max_amp_p95_deg']:.1f}° (目标 {m['target_max_amp_p95_deg']:.1f}°)"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--true-overlap", default="outputs/neck_motion/overlap_analysis/true_overlap.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--samples", type=int, default=10, help="CVAE 先验采样次数 K")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root or cfg["data"]["data_root"])
    model_type = cfg.get("model", {}).get("type", "regression")
    is_sampling = model_type in ("cvae", "candidates")

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
    print(f"[eval] split={args.split} n={len(ds)} model_type={model_type}")

    model = build_model_any(cfg, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss_fn = NeckMotionLoss(**cfg["loss"]) if model_type != "candidates" else _build_candidates_loss(cfg)

    if is_sampling:
        agg = eval_sampling(model, dl, loss_fn, device, n_samples=args.samples,
                            has_posterior=(model_type == "cvae"))
        print(_summarize(agg, args.samples))
        if agg["overall"].get("posterior_val_loss") is not None:
            print(f"[后验路径] val loss={agg['overall']['posterior_val_loss']:.5f}")
    else:
        agg = eval_regression(model, dl, loss_fn, device)
        print(format_metrics(agg))

    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)
    print(f"已保存: {out_path}")


def _build_candidates_loss(cfg: dict):
    from models.neck_motion.losses import MultiCandidateLoss
    c = cfg.get("candidates", {})
    return MultiCandidateLoss(
        k=int(c.get("k", 8)),
        w_pose=cfg["loss"]["w_pose"], w_velocity=cfg["loss"]["w_velocity"],
        w_acceleration=cfg["loss"]["w_acceleration"],
        pose=cfg["loss"].get("pose"), velocity=cfg["loss"].get("velocity"),
        acceleration=cfg["loss"].get("acceleration"),
        lambda_div=float(c.get("lambda_div", 1.0)), div_target=float(c.get("div_target", 0.02)),
        lambda_smooth=float(c.get("lambda_smooth", 0.5)), acc_ceiling=float(c.get("acc_ceiling", 0.004)),
        lambda_amp=float(c.get("lambda_amp", 1.0)), amp_ceiling=float(c.get("amp_ceiling", 0.25)),
    )


if __name__ == "__main__":
    main()
