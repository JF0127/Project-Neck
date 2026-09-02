#!/usr/bin/env python
"""训练 / 验证 / dry-run 脚本（话语级颈部运动生成，第一版基线）。

用法示例：
    python neck_motion/train.py --dry-run
    python neck_motion/train.py --config neck_motion/config.yaml --epochs 30
    python neck_motion/train.py --resume outputs/neck_motion/checkpoints/last.pt

- 自动选择 CUDA / MPS / CPU；固定随机种子；输出全部写入 outputs/neck_motion/，
  不污染原始数据目录。
- 词表只从 train split 构建并保存；断点续训复用 checkpoint 中的词表路径。
- 支持 checkpoint 保存（last.pt + 最佳 val loss best.pt）。
- --dry-run 只跑一个 train batch（完整前向/反向/更新）+ 一个 val batch，
  并输出数据链路自检结果。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# 允许以 `python neck_motion/train.py` 方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neck_motion.cvae import build_model as build_model_any
from neck_motion.dataset import ResponseNetDataset, collate_neck_motion
from neck_motion.losses import MultiCandidateLoss, NeckMotionLoss
from neck_motion.metrics import (
    aggregate_metrics,
    candidate_metrics,
    format_candidate_metrics,
    format_metrics,
    sample_metrics,
)
from neck_motion.model import build_neck_motion_model
from neck_motion.overlap_filter import ensure_true_overlap
from neck_motion.text_features import Vocab

logger = logging.getLogger("neck_motion")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for k in ("audio", "rpy", "delta", "mask", "Ns", "roles", "prev_token_ids", "prev_lens"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def setup_logging(log_file: str | Path) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


# --------------------------------------------------------------------------- #
# 训练 / 验证
# --------------------------------------------------------------------------- #
def kl_beta(global_step: int, cfg: dict, steps_per_epoch: int) -> float:
    """CVAE 的 KL 权重（含 warm-up）；回归模型返回 1.0（不使用）。"""
    if cfg.get("model", {}).get("type", "regression") != "cvae":
        return 1.0
    c = cfg.get("cvae", {})
    warmup = max(int(c.get("kl_warmup_epochs", 2)) * steps_per_epoch, 1)
    return float(c.get("kl_weight", 0.001)) * min(1.0, global_step / warmup)


def forward_and_loss(model, batch: dict, loss_fn, beta_kl: float):
    """回归 / CVAE / 多候选统一的前向 + 损失。返回 (pred, total, comps)。"""
    if getattr(model, "is_cvae", False):
        pred, aux = model(batch, return_aux=True, use_posterior=True)
        total, comps = loss_fn(pred, batch["rpy"], batch["delta"], batch["mask"],
                               kl=aux["kl"], beta_kl=beta_kl)
        return pred, total, comps
    if getattr(model, "is_candidates", False):
        preds = model(batch)  # [B, K, N, 3]
        total, comps = loss_fn(preds, batch["rpy"], batch["delta"], batch["mask"])
        return preds, total, comps
    pred = model(batch)
    total, comps = loss_fn(pred, batch["rpy"], batch["delta"], batch["mask"])
    return pred, total, comps


def run_train_epoch(model, loader, loss_fn, optimizer, device, cfg, epoch: int, global_step: int) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    comps_sum: dict[str, float] = defaultdict(float)
    t0 = time.time()
    grad_clip = float(cfg["train"]["grad_clip"])
    log_every = int(cfg["train"]["log_every"])

    for bi, batch in enumerate(loader):
        batch = to_device(batch, device)
        beta = kl_beta(global_step, cfg, len(loader))
        pred, total, comps = forward_and_loss(model, batch, loss_fn, beta)
        if not torch.isfinite(total):
            raise RuntimeError(f"loss 非有限值: {comps}")

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        global_step += 1

        total_loss += float(total) * pred.size(0)
        for k, v in comps.items():
            comps_sum[k] += float(v) * pred.size(0)
        n_batches += pred.size(0)

        if (bi + 1) % log_every == 0:
            msg = " | ".join(f"{k}={v:.4f}" for k, v in comps.items())
            logger.info(
                "epoch %d batch %d/%d loss=%.4f (%s) %.1fs",
                epoch, bi + 1, len(loader), float(total), msg, time.time() - t0,
            )
        if cfg.get("_debug_max_batches") and bi + 1 >= cfg["_debug_max_batches"]:
            break

    n = max(n_batches, 1)
    out = {"loss": total_loss / n, "samples": n}
    for k, v in comps_sum.items():
        out[f"loss_{k}"] = v / n
    return out, global_step


@torch.no_grad()
def run_validation(model, loader, loss_fn, device, cfg, global_step: int = 0) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    comps_sum: dict[str, float] = defaultdict(float)
    records: list[dict] = []
    beta = kl_beta(global_step, cfg, max(len(loader), 1))
    is_cand = getattr(model, "is_candidates", False)
    if is_cand:
        torch.manual_seed(0)  # 验证时先验采样可复现

    for batch in loader:
        batch = to_device(batch, device)
        pred, total, comps = forward_and_loss(model, batch, loss_fn, beta)
        B = pred.size(0)
        total_loss += float(total) * B
        n_samples += B
        for k, v in comps.items():
            comps_sum[k] += float(v) * B
        if is_cand:
            for i in range(B):
                m = candidate_metrics(pred[i], batch["rpy"][i], batch["mask"][i], batch["role_names"][i])
                if m is not None:
                    records.append(m)
        else:
            for i in range(B):
                m = sample_metrics(pred[i], batch["rpy"][i], batch["mask"][i], batch["role_names"][i])
                if m is not None:
                    records.append(m)
        if cfg.get("_debug_max_batches") and len(records) >= cfg["_debug_max_batches"]:
            break

    agg = aggregate_metrics(records)
    agg["overall"]["val_loss"] = total_loss / max(n_samples, 1)
    for k, v in comps_sum.items():
        agg["overall"][f"val_loss_{k}"] = v / max(n_samples, 1)
    return agg


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, cfg: dict, vocab_path: Path, global_step: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "config": cfg,
            "vocab_path": str(vocab_path),
        },
        path,
    )
    logger.info("已保存 checkpoint: %s", path)


# --------------------------------------------------------------------------- #
# dry-run：一个 batch 全链路自检
# --------------------------------------------------------------------------- #
def run_dry_run(cfg, device, train_loader, val_loader, model, loss_fn, optimizer, vocab) -> None:
    if getattr(model, "is_candidates", False):
        return run_dry_run_candidates(cfg, device, train_loader, val_loader, model, loss_fn, optimizer, vocab)

    logger.info("=" * 70)
    logger.info("DRY-RUN：数据加载 -> 特征 -> forward -> loss -> backward -> optimizer step")
    logger.info("=" * 70)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---------- 1) 一个 train batch：完整训练链路 ----------
    train_batch = next(iter(train_loader))
    logger.info("[1/3] train batch: %d 条样本（speaker=%d, listener=%d）",
                len(train_batch["sample_ids"]),
                sum(1 for r in train_batch["role_names"] if r == "speaker"),
                sum(1 for r in train_batch["role_names"] if r == "listener"))
    batch = to_device(train_batch, device)

    # 1a) 音频已转 16 kHz mono（audio_lens 为填充前的真实长度）
    sr_ok = True
    for i in range(batch["audio"].size(0)):
        t_len = int(batch["audio_lens"][i])
        expected = round(train_batch["duration_sec"][i] * cfg["data"]["target_sr"])
        if abs(t_len - expected) > max(320, expected * 0.01):
            sr_ok = False
            logger.error("样本 %s 音频长度 %d != 期望 %d", train_batch["sample_ids"][i], t_len, expected)
    check("音频已重采样到 16kHz mono", sr_ok)

    # 1b) forward（CVAE 走后验路径；beta_kl 用 warm-up 完成态）
    is_cvae = getattr(model, "is_cvae", False)
    if is_cvae:
        pred, aux = model(batch, return_aux=True, use_posterior=True)
    else:
        pred, aux = model(batch, return_aux=True)
    beta_kl = kl_beta(10**9, cfg, max(len(train_loader), 1))  # warm-up 完成态

    # 1c) 文本帧对齐长度 == 目标 N，且无词覆盖帧使用可学习 <SILENCE> 向量
    n_ok = True
    for i, n in enumerate(batch["Ns"].tolist()):
        if aux["coverage"].shape[1] < n:
            n_ok = False
    sil = model.silence_embedding.detach()
    sil_ok = True
    with torch.no_grad():
        for i, n in enumerate(batch["Ns"].tolist()):
            uncov = aux["coverage"][i, :n, 0] < 0.5
            if uncov.any():
                sil_ok &= bool(torch.allclose(
                    aux["word_feat"][i, :n][uncov],
                    sil.unsqueeze(0).expand(int(uncov.sum()), -1), atol=1e-6))
    check("文本帧对齐长度等于目标 N", n_ok)
    check("无词覆盖帧使用 <SILENCE> 向量", sil_ok)
    cov_ratio = float(aux["coverage"][:, : batch["Ns"].max(), :].mean())

    # 1d) 输出 shape == target shape
    shape_ok = pred.shape == batch["rpy"].shape
    check("输出 shape == target [B, N, 3]", shape_ok, f"pred={tuple(pred.shape)} target={tuple(batch['rpy'].shape)}")

    # 1e) 首帧强制为零
    first_zero = pred[:, 0, :].abs().max().item() < 1e-6
    check("首帧预测严格为零", first_zero, f"max|pred[:,0]|={pred[:, 0, :].abs().max().item():.2e}")

    # 1f) Speaker / Listener 路由到独立 head
    with torch.no_grad():
        zeroed_s = aux["out_speaker"].clone(); zeroed_s[:, 0, :] = 0.0
        zeroed_l = aux["out_listener"].clone(); zeroed_l[:, 0, :] = 0.0
        spk_ok, lst_ok = True, True
        for i, role in enumerate(batch["role_names"]):
            if role == "speaker":
                spk_ok &= torch.allclose(pred[i], zeroed_s[i], atol=1e-6)
            else:
                lst_ok &= torch.allclose(pred[i], zeroed_l[i], atol=1e-6)
        heads_differ = not torch.allclose(aux["out_speaker"], aux["out_listener"], atol=1e-4)
    check("Speaker 路由到 SpeakerHead", spk_ok)
    check("Listener 路由到 ListenerHead", lst_ok)
    check("Speaker/Listener head 参数独立（输出不同）", heads_differ)

    # 1g) padding 不参与 loss：把 padding 帧的预测改成垃圾值，loss 应保持不变
    if is_cvae:
        total_masked, comps_masked = loss_fn(pred, batch["rpy"], batch["delta"], batch["mask"],
                                             kl=aux["kl"], beta_kl=beta_kl)
    else:
        total_masked, comps_masked = loss_fn(pred, batch["rpy"], batch["delta"], batch["mask"])
    n_min = int(batch["Ns"].min())
    if n_min < int(batch["Ns"].max()):
        pred_corrupt = pred.clone()
        for i in range(pred.size(0)):  # 只破坏每个样本自己的 padding 帧 [N_i, Nmax)
            pred_corrupt[i, int(batch["Ns"][i]):] += 1000.0
        if is_cvae:
            total_corrupt, _ = loss_fn(pred_corrupt, batch["rpy"], batch["delta"], batch["mask"],
                                       kl=aux["kl"], beta_kl=beta_kl)
        else:
            total_corrupt, _ = loss_fn(pred_corrupt, batch["rpy"], batch["delta"], batch["mask"])
        pad_ok = abs(float(total_masked.detach()) - float(total_corrupt.detach())) < 1e-6
        check("padding 不参与 loss", pad_ok,
              f"masked={float(total_masked):.4f} corrupt_padding={float(total_corrupt):.4f}")
    else:
        check("padding 不参与 loss", True, "本 batch 无 padding（所有 N 相同）")

    # 1h) delta 文件 == target 差分（只比较有效帧对，排除 padding 边界）
    with torch.no_grad():
        diff = batch["rpy"][:, 1:] - batch["rpy"][:, :-1]
        pair_valid = batch["mask"][:, 1:] & batch["mask"][:, :-1]
        delta_err = float(((diff - batch["delta"][:, 1:]).abs() * pair_valid.unsqueeze(-1)).max())
    check("delta 文件与 relative 差分一致", delta_err < 1e-5, f"max|diff|={delta_err:.2e}")

    # 1i) 数值检查
    finite_ok = (
        torch.isfinite(pred).all()
        and torch.isfinite(batch["rpy"]).all()
        and all(torch.isfinite(v).all() if torch.is_tensor(v) else True for v in comps_masked.values())
    )
    check("无 NaN / Inf（pred / target / loss）", bool(finite_ok))

    # 1j) backward + optimizer step
    optimizer.zero_grad(set_to_none=True)
    total_masked.backward()
    has_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check("backward 后所有参数梯度有限", bool(has_grad))
    optimizer.step()
    check("optimizer step 完成", True)

    # 1k) CVAE 先验采样链路（部署路径）
    if is_cvae:
        with torch.no_grad():
            samples = model.sample(batch, n_samples=3)
        samp_ok = (samples.shape[0] == pred.size(0) and samples.shape[1] == 3
                   and samples.shape[2:] == pred.shape[1:])
        check("CVAE 先验采样 shape [B, K, N, 3]", samp_ok, f"{tuple(samples.shape)}")
        check("CVAE 采样首帧为零且有限",
              bool(torch.isfinite(samples).all()) and float(samples[:, :, 0, :].abs().max()) < 1e-6)
        check("CVAE KL 有限", bool(torch.isfinite(aux["kl"]).all()),
              f"kl={float(aux['kl']):.4f} (warm-up 完成态 beta_kl={beta_kl:.5f})")
        z_eff = float(aux["z"].std())
        check("z 非退化（std>1e-3）", z_eff > 1e-3, f"z.std={z_eff:.4f}")

    logger.info("loss(pose=%.4f vel=%.4f acc=%.4f%s) total=%.4f | 词覆盖帧占比=%.1f%%",
                float(comps_masked["pose"]), float(comps_masked["velocity"]),
                float(comps_masked["acceleration"]),
                f" kl={float(comps_masked['kl']):.4f}" if "kl" in comps_masked else "",
                float(total_masked), cov_ratio * 100)

    # ---------- 2) 一个 val batch：forward + 指标 ----------
    logger.info("[2/3] val batch: %d 条样本", len(val_loader.dataset))
    val_batch = next(iter(val_loader))
    vbatch = to_device(val_batch, device)
    model.eval()
    with torch.no_grad():
        if is_cvae:
            vpred, vaux = model(vbatch, return_aux=True, use_posterior=True)
            vtotal, vcomps = loss_fn(vpred, vbatch["rpy"], vbatch["delta"], vbatch["mask"],
                                     kl=vaux["kl"], beta_kl=beta_kl)
        else:
            vpred = model(vbatch)
            vtotal, vcomps = loss_fn(vpred, vbatch["rpy"], vbatch["delta"], vbatch["mask"])
        vrec = [
            m for i in range(vpred.size(0))
            if (m := sample_metrics(vpred[i], vbatch["rpy"][i], vbatch["mask"][i], vbatch["role_names"][i])) is not None
        ]
    logger.info("val loss=%.4f | %s", float(vtotal), format_metrics(aggregate_metrics(vrec)).replace("\n", " | "))

    # ---------- 3) 汇总 ----------
    logger.info("[3/3] dry-run 自检结果：")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        logger.info("  [%s] %s%s", "PASS" if ok else "FAIL", name, f" ({detail})" if detail else "")
    logger.info("词表大小=%d | 模型参数量=%.2fM | device=%s", len(vocab), sum(p.numel() for p in model.parameters()) / 1e6, device)
    if not all_ok:
        raise SystemExit("DRY-RUN 存在失败项，请检查日志。")
    logger.info("DRY-RUN 全部通过。")


# --------------------------------------------------------------------------- #
# dry-run（多候选模型）
# --------------------------------------------------------------------------- #
def run_dry_run_candidates(cfg, device, train_loader, val_loader, model, loss_fn, optimizer, vocab) -> None:
    logger.info("=" * 70)
    logger.info("DRY-RUN（多候选 K=%d）：先验采样 -> forward -> loss -> backward -> optimizer step", model.k)
    logger.info("=" * 70)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    K = model.k
    # 1) 一个 train batch
    train_batch = next(iter(train_loader))
    logger.info("[1/3] train batch: %d 条样本（speaker=%d, listener=%d）",
                len(train_batch["sample_ids"]),
                sum(1 for r in train_batch["role_names"] if r == "speaker"),
                sum(1 for r in train_batch["role_names"] if r == "listener"))
    batch = to_device(train_batch, device)

    preds, aux = model(batch, return_aux=True)  # [B, K, N, 3]
    B, N = preds.shape[0], preds.shape[2]
    check("候选 shape [B, K, N, 3]", tuple(preds.shape) == (B, K, N, 3),
          f"{tuple(preds.shape)}")
    check("候选首帧严格为零", float(preds[:, :, 0, :].abs().max().detach()) < 1e-6,
          f"max={float(preds[:, :, 0, :].abs().max().detach()):.2e}")
    check("候选间初始多样性 > 0", bool((preds.unsqueeze(2) - preds.unsqueeze(1)).abs().mean().detach() > 1e-4),
          f"pairwise={float((preds.unsqueeze(2) - preds.unsqueeze(1)).abs().mean().detach()):.4f} rad")
    check("文本帧对齐长度等于目标 N", aux["coverage"].shape[1] >= N)
    check("无 NaN / Inf（preds / target）", bool(torch.isfinite(preds).all()) and bool(torch.isfinite(batch["rpy"]).all()))

    # loss（含 padding 不参与检查）
    total, comps = loss_fn(preds, batch["rpy"], batch["delta"], batch["mask"])
    check("loss 各分项有限", bool(torch.isfinite(total)) and all(torch.isfinite(v) for v in comps.values()),
          " | ".join(f"{k}={float(v):.4f}" for k, v in comps.items()))
    if int(batch["Ns"].min()) < int(batch["Ns"].max()):
        corrupt = preds.clone()
        for i in range(B):
            corrupt[i, :, int(batch["Ns"][i]):] += 1000.0
        total_c, _ = loss_fn(corrupt, batch["rpy"], batch["delta"], batch["mask"])
        check("padding 不参与 loss", abs(float(total.detach()) - float(total_c.detach())) < 1e-6,
              f"{float(total):.4f} vs {float(total_c):.4f}")
    else:
        check("padding 不参与 loss", True, "本 batch 无 padding")

    # backward + step
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    has_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check("backward 后所有参数梯度有限", bool(has_grad))
    optimizer.step()
    check("optimizer step 完成", True)

    # 部署接口一致性：sample() 与 forward() 相同路径
    with torch.no_grad():
        samples = model.sample(batch, n_samples=3)
    check("sample() 接口 shape 一致", tuple(samples.shape) == (B, 3, N, 3), f"{tuple(samples.shape)}")

    # 2) 一个 val batch：采样指标
    logger.info("[2/3] val batch: %d 条样本", len(val_loader.dataset))
    val_batch = next(iter(val_loader))
    vbatch = to_device(val_batch, device)
    model.eval()
    with torch.no_grad():
        vpreds = model(vbatch)
        vrec = [
            m for i in range(vpreds.size(0))
            if (m := candidate_metrics(vpreds[i], vbatch["rpy"][i], vbatch["mask"][i], vbatch["role_names"][i])) is not None
        ]
    logger.info("%s", format_candidate_metrics(aggregate_metrics(vrec), K).replace("\n", " | "))

    # 3) 汇总
    logger.info("[3/3] dry-run 自检结果：")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        logger.info("  [%s] %s%s", "PASS" if ok else "FAIL", name, f" ({detail})" if detail else "")
    logger.info("词表大小=%d | 模型参数量=%.2fM | device=%s", len(vocab), sum(p.numel() for p in model.parameters()) / 1e6, device)
    if not all_ok:
        raise SystemExit("DRY-RUN 存在失败项，请检查日志。")
    logger.info("DRY-RUN 全部通过。")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="话语级颈部运动生成——训练/验证")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"), help="YAML 配置文件")
    p.add_argument("--output-dir", default=None, help="输出目录（覆盖 config.train.output_dir）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu/cuda/mps（默认自动选择）")
    p.add_argument("--resume", default=None, help="checkpoint 路径（断点续训）")
    p.add_argument("--no-filter-overlap", action="store_true", help="关闭 overlap 过滤")
    p.add_argument("--ablate", choices=["none", "audio", "text"], default="none",
                   help="消融实验：none=完整模型；audio=去掉音频分支；text=去掉文本（词+上下文）")
    p.add_argument("--model-type", choices=["regression", "cvae", "candidates"], default=None,
                   help="覆盖 config.model.type")
    p.add_argument("--kl-weight", type=float, default=None, help="CVAE KL 权重（覆盖 config.cvae.kl_weight）")
    p.add_argument("--free-bits", type=float, default=None, help="CVAE free-bits nats/维（覆盖 config.cvae.free_bits）")
    p.add_argument("--kl-warmup-epochs", type=float, default=None, help="CVAE KL warm-up epoch 数")
    p.add_argument("--role-only", action="store_true",
                   help="role-only 多候选基线：移除音频/文本/上下文，仅保留 role+时间位置")
    p.add_argument("--dry-run", action="store_true", help="只跑一个 batch 验证全链路后退出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 命令行覆盖
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    for key, attr in (("epochs", "epochs"), ("batch_size", "batch_size"), ("learning_rate", "lr"),
                      ("num_workers", "num_workers"), ("seed", "seed")):
        val = getattr(args, attr)
        if val is not None:
            cfg["train"][key] = val
    if args.no_filter_overlap:
        cfg["data"]["filter_overlap_sec"] = None

    if args.model_type is not None:
        cfg.setdefault("model", {})["type"] = args.model_type
    if args.kl_weight is not None:
        cfg.setdefault("cvae", {})["kl_weight"] = args.kl_weight
    if args.free_bits is not None:
        cfg.setdefault("cvae", {})["free_bits"] = args.free_bits
    if args.kl_warmup_epochs is not None:
        cfg.setdefault("cvae", {})["kl_warmup_epochs"] = args.kl_warmup_epochs
    if args.role_only:
        cfg.setdefault("candidates", {})["role_only"] = True
        if args.output_dir is None:
            cfg["train"]["output_dir"] = str(Path(cfg["train"]["output_dir"]) / "roleonly")

    # 消融实验：覆盖模型标志，输出到独立子目录，避免覆盖主实验
    if args.ablate == "audio":
        cfg["model"]["use_audio"] = False
    elif args.ablate == "text":
        cfg["model"]["use_text"] = False
    if args.ablate != "none" and args.output_dir is None:
        cfg["train"]["output_dir"] = str(Path(cfg["train"]["output_dir"]) / f"ablate_{args.ablate}")

    out_dir = Path(cfg["train"]["output_dir"])
    setup_logging(out_dir / "train.log")
    logger.info("配置: %s", json.dumps(cfg, ensure_ascii=False, default=str)[:2000])

    device = resolve_device(args.device)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    logger.info("device=%s | seed=%d | 输出目录=%s", device, seed, out_dir)

    # ---------- 词表（只从 train 构建） ----------
    vocab_path = out_dir / "vocab.json"
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        vocab_path = Path(ckpt["vocab_path"])
        if not vocab_path.exists():  # 相对 checkpoint 目录解析
            vocab_path = Path(args.resume).parent / Path(ckpt["vocab_path"]).name
        logger.info("从 checkpoint 复用词表: %s", vocab_path)
        vocab = Vocab.load(vocab_path)
    elif vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        logger.info("复用已有词表: %s（大小=%d）", vocab_path, len(vocab))
    else:
        logger.info("从 train split 构建词表...")
        ds_vocab = ResponseNetDataset(cfg["data"]["data_root"], split="train", build_vocab=True,
                                      filter_overlap_sec=cfg["data"]["filter_overlap_sec"],
                                      true_overlap_path=cfg["data"].get("true_overlap_path"),
                                      target_sr=cfg["data"]["target_sr"])
        vocab = ds_vocab.vocab
        vocab.save(vocab_path)
        logger.info("词表已保存: %s（大小=%d）", vocab_path, len(vocab))

    # ---------- true_overlap 映射（过滤用，缺失则自动生成） ----------
    if cfg["data"].get("filter_overlap_sec") is not None:
        ensure_true_overlap(cfg["data"]["data_root"], cfg["data"]["true_overlap_path"])

    # ---------- 数据集 / DataLoader ----------
    kwargs = dict(data_root=cfg["data"]["data_root"], vocab=vocab,
                  filter_overlap_sec=cfg["data"]["filter_overlap_sec"],
                  true_overlap_path=cfg["data"].get("true_overlap_path"),
                  target_sr=cfg["data"]["target_sr"])
    train_ds = ResponseNetDataset(split="train", **kwargs)
    val_ds = ResponseNetDataset(split="val", **kwargs)
    logger.info("train=%d 条 | val=%d 条", len(train_ds), len(val_ds))

    nw = int(cfg["train"]["num_workers"]) if not args.dry_run else 0
    dl_kwargs = dict(collate_fn=collate_neck_motion, num_workers=nw, pin_memory=bool(cfg["train"]["pin_memory"]) and device.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]),
                              shuffle=True, drop_last=False, worker_init_fn=seed_worker,
                              generator=g, persistent_workers=nw > 0, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False, worker_init_fn=seed_worker, **dl_kwargs)

    # ---------- 模型 / 损失 / 优化器 ----------
    model_type = cfg.get("model", {}).get("type", "regression")
    model = build_model_any(cfg, vocab_size=len(vocab)).to(device)
    logger.info("模型类型: %s | 参数量=%.2fM", model_type, sum(p.numel() for p in model.parameters()) / 1e6)
    if model_type == "candidates":
        c = cfg.get("candidates", {})
        loss_fn = MultiCandidateLoss(
            k=int(c.get("k", 8)),
            w_pose=cfg["loss"]["w_pose"], w_velocity=cfg["loss"]["w_velocity"],
            w_acceleration=cfg["loss"]["w_acceleration"],
            pose=cfg["loss"].get("pose"), velocity=cfg["loss"].get("velocity"),
            acceleration=cfg["loss"].get("acceleration"),
            lambda_div=float(c.get("lambda_div", 1.0)), div_target=float(c.get("div_target", 0.02)),
            lambda_smooth=float(c.get("lambda_smooth", 0.5)), acc_ceiling=float(c.get("acc_ceiling", 0.004)),
            lambda_amp=float(c.get("lambda_amp", 1.0)), amp_ceiling=float(c.get("amp_ceiling", 0.25)),
        )
    else:
        loss_fn = NeckMotionLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["train"]["learning_rate"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = None
    if cfg["train"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))

    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val_loss"])
        global_step = int(ckpt.get("global_step", 0))
        if scheduler is not None:
            for _ in range(start_epoch - 1):
                scheduler.step()
        logger.info("已从 %s 恢复：epoch=%d, best_val_loss=%.4f", args.resume, ckpt["epoch"], best_val)

    save_config(cfg, out_dir / "config.yaml")
    ckpt_dir = out_dir / "checkpoints"

    # ---------- dry-run ----------
    if args.dry_run:
        run_dry_run(cfg, device, train_loader, val_loader, model, loss_fn, optimizer, vocab)
        logger.info("dry-run 完成，未保存任何 checkpoint。")
        return

    # ---------- 训练循环 ----------
    logger.info("开始训练：epochs=%d batch_size=%d lr=%.2e model_type=%s", int(cfg["train"]["epochs"]),
                int(cfg["train"]["batch_size"]), float(cfg["train"]["learning_rate"]), model_type)
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()
        tr, global_step = run_train_epoch(model, train_loader, loss_fn, optimizer, device, cfg, epoch, global_step)
        if scheduler is not None:
            scheduler.step()
        logger.info("epoch %d/%d 训练完成 loss=%.4f (%.0fs)",
                    epoch, int(cfg["train"]["epochs"]), tr["loss"], time.time() - t0)

        va = run_validation(model, val_loader, loss_fn, device, cfg, global_step)
        if model_type == "candidates":
            logger.info("epoch %d 验证:\n%s", epoch,
                        format_candidate_metrics(va, int(cfg.get("candidates", {}).get("k", 8))))
        else:
            logger.info("epoch %d 验证:\n%s", epoch, format_metrics(va))
        val_loss = va["overall"]["val_loss"]

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_val, cfg, vocab_path, global_step)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_val, cfg, vocab_path, global_step)
            logger.info("新的最佳 val loss: %.4f", best_val)

        va["epoch"] = epoch
        with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(va, f, ensure_ascii=False, indent=1)

    logger.info("训练结束。最佳 val loss=%.4f，checkpoint: %s", best_val, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
