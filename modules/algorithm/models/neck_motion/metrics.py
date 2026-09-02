"""验证指标（弧度/角度、速度/加速度、分角色、首帧误差）。

所有指标只对每个样本的有效帧计算，再按样本平均（避免变长序列池化偏差）。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def sample_metrics(
    pred: torch.Tensor,     # [T, 3]
    target: torch.Tensor,   # [T, 3]
    mask: torch.Tensor,     # [T] bool（有效帧为前缀连续）
    role: str,              # "speaker" | "listener"
    fps: float = 30.0,      # 速度/加速度换算用（°/帧 -> °/s）
) -> Optional[dict]:
    """单个样本的指标。无有效帧时返回 None。

    速度/加速度单位为 °/s 与 °/s²（差分按帧计算后乘 fps / fps²）。
    """
    n = int(mask.sum())
    if n == 0:
        return None
    p = pred[:n].float()
    t = target[:n].float()

    mae_rad = float((p - t).abs().mean())
    rmse_rad = float(((p - t) ** 2).mean().sqrt())

    vel_mae_deg_per_s = 0.0
    if n >= 2:
        vel_mae_deg_per_s = float(((p[1:] - p[:-1]) - (t[1:] - t[:-1])).abs().mean()) * 180.0 / math.pi * fps

    acc_mae_deg_per_s2 = 0.0
    if n >= 3:
        pa = p[2:] - 2.0 * p[1:-1] + p[:-2]
        ta = t[2:] - 2.0 * t[1:-1] + t[:-2]
        acc_mae_deg_per_s2 = float((pa - ta).abs().mean()) * 180.0 / math.pi * fps * fps

    first_frame_err = float((p[0] - t[0]).abs().mean())  # target[0] ≈ 0

    # 轨迹诊断（判断模型是否只是输出近零轨迹）
    pred_amp = float(p.abs().mean())
    target_amp = float(t.abs().mean())
    corr = 0.0
    if p.std() > 1e-8 and t.std() > 1e-8:
        corr = float(torch.corrcoef(torch.stack([p.flatten(), t.flatten()]))[0, 1])

    return {
        "role": role,
        "frames": n,
        "mae_rad": mae_rad,
        "mae_deg": math.degrees(mae_rad),
        "rmse_rad": rmse_rad,
        "rmse_deg": math.degrees(rmse_rad),
        "vel_mae_rad_per_s": vel_mae_deg_per_s * math.pi / 180.0,
        "vel_mae_deg_per_s": vel_mae_deg_per_s,
        "acc_mae_rad_per_s2": acc_mae_deg_per_s2 * math.pi / 180.0,
        "acc_mae_deg_per_s2": acc_mae_deg_per_s2,
        "first_frame_err_rad": first_frame_err,
        "first_frame_err_deg": math.degrees(first_frame_err),
        "pred_amp_deg": math.degrees(pred_amp),
        "target_amp_deg": math.degrees(target_amp),
        "corr": corr,
    }


def aggregate_metrics(records: list[dict]) -> dict:
    """把 sample_metrics 列表聚合为 overall / speaker / listener 三组。"""
    if not records:
        return {"overall": None, "speaker": None, "listener": None}
    keys = [k for k in records[0] if k not in ("role", "frames")]

    def mean_of(rs: list[dict]) -> Optional[dict]:
        if not rs:
            return None
        out = {"samples": len(rs)}
        for k in keys:
            out[k] = float(np.mean([r[k] for r in rs]))
        return out

    return {
        "overall": mean_of(records),
        "speaker": mean_of([r for r in records if r["role"] == "speaker"]),
        "listener": mean_of([r for r in records if r["role"] == "listener"]),
    }


def candidate_metrics(
    preds: torch.Tensor,    # [K, n, 3] 先验采样候选
    target: torch.Tensor,   # [n, 3]
    mask: torch.Tensor,     # [n] bool
    role: str,
    fps: float = 30.0,
) -> Optional[dict]:
    """多候选模型指标：best-of-K（oracle）、多样性、幅度、平滑度、安全范围。

    注意：best-of-K 是 oracle 覆盖度指标，不是部署期望误差。
    """
    n = int(mask.sum())
    if n == 0 or preds.size(0) == 0:
        return None
    p = preds[:, :n].float()   # [K, n, 3]
    t = target[:n].float()     # [n, 3]
    K = p.size(0)
    rad2deg = 180.0 / math.pi

    mae_k = (p - t).abs().mean(dim=(1, 2))                     # [K]
    best_of_k_deg = float(mae_k.min()) * rad2deg

    d = p.unsqueeze(1) - p.unsqueeze(0)                        # [K,K,n,3]
    pair = d.abs().mean(dim=(2, 3))                            # [K,K]
    iu = torch.triu_indices(K, K, offset=1)
    diversity_deg = float(pair[iu[0], iu[1]].mean()) * rad2deg

    amp_per_k = p.abs().mean(dim=(1, 2))                       # [K]
    amp_deg = float(amp_per_k.mean()) * rad2deg
    amp_std_across_k_deg = float(amp_per_k.std()) * rad2deg

    d1 = (p[:, 1:] - p[:, :-1]).abs().mean(dim=(1, 2))         # rad/帧 [K]
    d2 = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]).abs().mean(dim=(1, 2))
    max_amp = p.abs().amax(dim=(1, 2))                         # rad [K]

    return {
        "role": role,
        "frames": n,
        "best_of_k_deg": best_of_k_deg,
        "diversity_deg": diversity_deg,
        "amp_deg": amp_deg,
        "amp_std_across_k_deg": amp_std_across_k_deg,
        "target_amp_deg": float(t.abs().mean()) * rad2deg,
        "vel_deg_per_s": float(d1.mean()) * rad2deg * fps,
        "vel_p95_deg_per_s": float(torch.quantile(d1, 0.95)) * rad2deg * fps,
        "acc_deg_per_s2": float(d2.mean()) * rad2deg * fps * fps,
        "max_amp_p95_deg": float(torch.quantile(max_amp, 0.95)) * rad2deg,
        "target_max_amp_p95_deg": float(torch.quantile(t.abs().amax(dim=1), 0.95)) * rad2deg,
    }


def format_candidate_metrics(agg: dict, k: int) -> str:
    """多候选聚合指标 -> 多行可读字符串。"""
    lines = [f"[多候选采样 K={k}]（best-of-K 为 oracle 覆盖度指标）"]
    for group, key in (("overall", "全体"), ("speaker", "Speaker"), ("listener", "Listener")):
        m = agg.get(group)
        if not m:
            lines.append(f"[{key}] 无样本")
            continue
        lines.append(
            f"[{key}] n={m['samples']} | best-of-{k} {m['best_of_k_deg']:.3f}° | "
            f"多样性 {m['diversity_deg']:.2f}° | 幅度 {m['amp_deg']:.2f}° "
            f"(目标 {m['target_amp_deg']:.2f}°, 跨候选σ {m['amp_std_across_k_deg']:.2f}°) | "
            f"vel {m['vel_deg_per_s']:.2f}°/s (P95 {m['vel_p95_deg_per_s']:.1f}) | "
            f"acc {m['acc_deg_per_s2']:.0f}°/s² | max幅度P95 {m['max_amp_p95_deg']:.1f}° "
            f"(目标 {m['target_max_amp_p95_deg']:.1f}°)"
        )
    return "\n".join(lines)


def format_metrics(agg: dict) -> str:
    """聚合指标 -> 多行可读字符串。"""
    lines = []
    for group, key in (("overall", "全体"), ("speaker", "Speaker"), ("listener", "Listener")):
        m = agg.get(group)
        if not m:
            lines.append(f"[{key}] 无样本")
            continue
        lines.append(
            f"[{key}] n={m['samples']} | "
            f"MAE {m['mae_deg']:.3f}° ({m['mae_rad']:.5f} rad) | "
            f"RMSE {m['rmse_deg']:.3f}° | "
            f"vel MAE {m['vel_mae_deg_per_s']:.3f}°/s | "
            f"acc MAE {m['acc_mae_deg_per_s2']:.3f}°/s² | "
            f"首帧误差 {m['first_frame_err_deg']:.4f}° | "
            f"|pred|={m['pred_amp_deg']:.2f}° |t|={m['target_amp_deg']:.2f}° corr={m['corr']:.3f}"
        )
    return "\n".join(lines)
