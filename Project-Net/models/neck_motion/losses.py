"""训练损失：掩码 Smooth L1 / L1，只对有效帧计算，padding 不参与。

    L_total = 1.0 * L_pose + 0.5 * L_velocity + 0.1 * L_acceleration

每个分项可独立配置（弧度）：
- pose：Smooth L1，beta=0.1（train target |rpy| 的 P75≈0.069 / P90≈0.131）
- velocity：L1（train target 帧差分 |Δ| P50≈0.0015、P75≈0.0038 rad，量级太小，
  SmoothL1 二次区会把梯度再压弱；L1 保证梯度恒定）
- acceleration：L1（|Δ²| P50≈0.0009、P75≈0.0020 rad）
- 长度不足以计算差分项的样本安全跳过（不产生 NaN）；每项单独记录。
"""
from __future__ import annotations

import torch
import torch.nn as nn

VALID_MODES = ("l1", "smooth_l1")


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    """逐元素 Smooth L1，按有效帧 mask 归一化。mask [B, T] bool。"""
    d = (pred - target).abs()
    loss = torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)
    loss = loss * mask.unsqueeze(-1)
    count = mask.sum() * pred.shape[-1]
    return loss.sum() / count.clamp(min=1.0)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).abs() * mask.unsqueeze(-1)
    count = mask.sum() * pred.shape[-1]
    return loss.sum() / count.clamp(min=1.0)


def _per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: dict) -> torch.Tensor:
    """per-sample 掩码损失 [B]（MultiCandidateLoss 用，逐候选/逐样本归一化）。"""
    if cfg["mode"] == "l1":
        d = (pred - target).abs() * mask.unsqueeze(-1)
    else:
        d_abs = (pred - target).abs()
        beta = float(cfg["beta"])
        d = torch.where(d_abs < beta, 0.5 * d_abs * d_abs / beta, d_abs - 0.5 * beta)
        d = d * mask.unsqueeze(-1)
    cnt = mask.sum(dim=-1) * pred.shape[-1]
    return d.sum(dim=(1, 2)) / cnt.clamp(min=1.0)


class NeckMotionLoss(nn.Module):
    """第一版训练损失。输入全部为弧度。

    term 配置示例：dict(mode="smooth_l1", beta=0.1) 或 dict(mode="l1")。
    """

    def __init__(
        self,
        w_pose: float = 1.0,
        w_velocity: float = 0.5,
        w_acceleration: float = 0.1,
        pose: dict | None = None,
        velocity: dict | None = None,
        acceleration: dict | None = None,
    ):
        super().__init__()
        self.weights = {"pose": w_pose, "velocity": w_velocity, "acceleration": w_acceleration}
        self.terms = {
            "pose": self._norm_term(pose, {"mode": "smooth_l1", "beta": 0.1}),
            "velocity": self._norm_term(velocity, {"mode": "l1"}),
            "acceleration": self._norm_term(acceleration, {"mode": "l1"}),
        }

    @staticmethod
    def _norm_term(cfg: dict | None, default: dict) -> dict:
        cfg = dict(default if cfg is None else cfg)
        if cfg["mode"] not in VALID_MODES:
            raise ValueError(f"未知损失模式 {cfg['mode']}，可选 {VALID_MODES}")
        if cfg["mode"] == "smooth_l1" and float(cfg.get("beta", 0.0)) <= 0:
            raise ValueError("smooth_l1 需要 beta > 0")
        return cfg

    def _apply(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: dict) -> torch.Tensor:
        if cfg["mode"] == "l1":
            return _masked_l1(pred, target, mask)
        return _masked_smooth_l1(pred, target, mask, float(cfg["beta"]))

    def forward(
        self,
        pred: torch.Tensor,          # [B, T, 3]
        target_rpy: torch.Tensor,    # [B, T, 3] relative RPY
        target_delta: torch.Tensor,  # [B, T, 3] relative 帧间差分（delta[0]=0；t>=1 时 delta[t]=rpy[t]-rpy[t-1]）
        mask: torch.Tensor,          # [B, T] bool，True = 有效帧
        kl: torch.Tensor | None = None,   # CVAE 的 KL(q||p) 标量（已含 free-bits）
        beta_kl: float = 1.0,             # CVAE 的 KL 权重（含 warm-up 因子）
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        T = pred.size(1)
        comps: dict[str, torch.Tensor] = {}

        # L_pose
        comps["pose"] = self._apply(pred, target_rpy, mask, self.terms["pose"])

        # L_velocity：预测差分 vs 数据提供的 delta（等价于 target 差分）
        if T >= 2:
            pv = pred[:, 1:] - pred[:, :-1]
            tv = target_delta[:, 1:]  # delta[t+1] = rpy[t+1] - rpy[t]
            mv = mask[:, 1:] & mask[:, :-1]
            comps["velocity"] = self._apply(pv, tv, mv, self.terms["velocity"])
        else:
            comps["velocity"] = pred.new_zeros(())

        # L_acceleration：二阶差分
        if T >= 3:
            pa = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
            ta = target_delta[:, 2:] - target_delta[:, 1:-1]  # 与 rpy 二阶差分一致
            ma = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            comps["acceleration"] = self._apply(pa, ta, ma, self.terms["acceleration"])
        else:
            comps["acceleration"] = pred.new_zeros(())

        total = sum(self.weights[k] * comps[k] for k in ("pose", "velocity", "acceleration"))
        if kl is not None:
            comps["kl"] = kl
            total = total + beta_kl * kl
        return total, comps


class MultiCandidateLoss(nn.Module):
    """条件多候选轨迹模型的训练损失（直接对可部署先验采样训练，无后验）。

        L = min_k L_recon(ŷ_k, y)            # 轨迹级重构，选最接近标注的一条
          + λ_div · L_diversity              # 有上界：max(0, div_target - 候选两两距离)
          + λ_smooth · L_smooth              # 每条候选的加速度 hinge（防失控 jerk）
          + λ_amp · L_amp                    # 每条候选的幅度 hinge（防夸张动作刷多样性）

    preds 形状 [B, K, N, 3]；所有项均只对有效帧计算。
    """

    def __init__(
        self,
        k: int = 8,
        w_pose: float = 1.0,
        w_velocity: float = 0.5,
        w_acceleration: float = 0.1,
        pose: dict | None = None,
        velocity: dict | None = None,
        acceleration: dict | None = None,
        lambda_div: float = 1.0,
        div_target: float = 0.02,        # rad：候选两两距离达到该值后多样性项为 0（有上界）
        lambda_smooth: float = 0.5,
        acc_ceiling: float = 0.004,      # rad/帧²：二阶差分 hinge 阈值（≈200°/s²）
        lambda_amp: float = 1.0,
        amp_ceiling: float = 0.25,       # rad：单帧幅度 hinge 阈值（≈14.3°）
    ):
        super().__init__()
        self.weights = {"pose": w_pose, "velocity": w_velocity, "acceleration": w_acceleration}
        self.terms = {
            "pose": NeckMotionLoss._norm_term(pose, {"mode": "smooth_l1", "beta": 0.1}),
            "velocity": NeckMotionLoss._norm_term(velocity, {"mode": "l1"}),
            "acceleration": NeckMotionLoss._norm_term(acceleration, {"mode": "l1"}),
        }
        self.lambda_div = lambda_div
        self.div_target = div_target
        self.lambda_smooth = lambda_smooth
        self.acc_ceiling = acc_ceiling
        self.lambda_amp = lambda_amp
        self.amp_ceiling = amp_ceiling

    def forward(
        self,
        preds: torch.Tensor,           # [B, K, N, 3]
        target_rpy: torch.Tensor,      # [B, N, 3]
        target_delta: torch.Tensor,    # [B, N, 3]
        mask: torch.Tensor,            # [B, N] bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K, N, _ = preds.shape
        device = preds.device
        comps: dict[str, torch.Tensor] = {}

        # ---- 重构：每候选分项损失 -> min over K（轨迹级选择） ----
        recon = torch.zeros(B, K, device=device)
        for k in range(K):
            pk = preds[:, k]
            pose_k = _per_sample(pk, target_rpy, mask, self.terms["pose"])
            if N >= 2:
                mv = mask[:, 1:] & mask[:, :-1]
                vel_k = _per_sample(pk[:, 1:] - pk[:, :-1], target_delta[:, 1:], mv, self.terms["velocity"])
            else:
                vel_k = pk.new_zeros(B)
            if N >= 3:
                ma = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
                acc_k = _per_sample(pk[:, 2:] - 2 * pk[:, 1:-1] + pk[:, :-2],
                                    target_delta[:, 2:] - target_delta[:, 1:-1], ma, self.terms["acceleration"])
            else:
                acc_k = pk.new_zeros(B)
            recon[:, k] = (self.weights["pose"] * pose_k + self.weights["velocity"] * vel_k
                           + self.weights["acceleration"] * acc_k)
        recon_min = recon.min(dim=1).values  # [B]
        comps["recon_min"] = recon_min.mean()

        # ---- 多样性（有上界）：候选两两轨迹级 L1 距离，达到 div_target 即停 ----
        d = preds.unsqueeze(2) - preds.unsqueeze(1)          # [B,K,K,N,3]
        d_abs = d.abs() * mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        cnt = mask.sum(dim=-1).clamp(min=1.0) * 3            # [B]
        pair_dist = d_abs.sum(dim=(3, 4)) / cnt.unsqueeze(1).unsqueeze(1)  # [B,K,K]
        iu = torch.triu_indices(K, K, offset=1, device=device)
        div = (self.div_target - pair_dist[:, iu[0], iu[1]]).clamp(min=0.0)  # [B, pairs]
        comps["diversity"] = div.mean()

        # ---- 平滑：每条候选的二阶差分 hinge（超出 acc_ceiling 才惩罚） ----
        d2 = preds[:, :, 2:] - 2 * preds[:, :, 1:-1] + preds[:, :, :-2]  # [B,K,N-2,3]
        m2 = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).unsqueeze(1).unsqueeze(-1)  # [B,1,N-2,1]
        cnt2 = m2.sum(dim=(2, 3)).clamp(min=1.0)              # [B,K]
        smooth_pen = (d2.abs() - self.acc_ceiling).clamp(min=0.0) * m2
        comps["smooth"] = (smooth_pen.sum(dim=(2, 3)) / cnt2).mean()

        # ---- 幅度：每条候选的单帧幅度 hinge（超出 amp_ceiling 才惩罚） ----
        m1 = mask.unsqueeze(1).unsqueeze(-1)                  # [B,1,N,1]
        cnt1 = m1.sum(dim=(2, 3)).clamp(min=1.0)              # [B,K]
        amp_pen = (preds.abs() - self.amp_ceiling).clamp(min=0.0) * m1
        comps["amp"] = (amp_pen.sum(dim=(2, 3)) / cnt1).mean()

        total = (comps["recon_min"]
                 + self.lambda_div * comps["diversity"]
                 + self.lambda_smooth * comps["smooth"]
                 + self.lambda_amp * comps["amp"])
        return total, comps
