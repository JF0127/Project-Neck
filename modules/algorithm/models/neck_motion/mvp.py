#!/usr/bin/env python
"""MVP 集成：Speaking / Listening / Silent 三态颈部动作调度（v3 角色动作先验）。

状态机：
    Speaking  → v3 speaker 动作先验（多候选采样 + 选择策略）
    Listening → v3 listener 动作先验（同上）
    Silent    → 规则化保持/回中（线性回中到会话初始姿态）

候选选择策略（best-of-K 需要真实标签、线上不可用，必须显式选择）：
    first         固定取第 0 个候选（可复现，等效固定 z 模板）
    random        随机采样一个候选（--seed 可复现）
    style_fixed   固定风格：使用固定 latent 模板（--style-z-path 或自动生成并保存
                  style_z.npy），取第 --style-idx 个模板——同一风格跨段/跨调用可复现
    energy_match  按期望动作强度选择：候选运动能量（°/s）最接近期望值者；
                  期望值来自 --expected-energy（°/s），缺省按 role 启发式
                  （speaking=候选能量中位数，listening=P25）

拼接语义：各段相对轨迹按“段起点姿态”合成到统一坐标系（相对
robot_actual_initial），段间连续，边界 blend_sec 内线性平滑。
Silent 段回中目标为 robot_neutral_pose（机器人标定中位，默认 [0,0,0]）；
robot_actual_initial 仅是每段开始时的实际姿态（坐标系基准），不是回中目标。
输出轨迹首帧为 [0,0,0]。

能力边界（研究结论）：动作平滑、多样，但**不具备**内容/语义/节奏响应能力；
本模块不负责关节限位、速度/加速度/jerk 限制、坐标标定与硬件下发（执行层职责）。

用法：
    python models/neck_motion/mvp.py --checkpoint outputs/neck_motion_v3/checkpoints/best.pt \
        --events events.json --data-root data/source/response-net/processed_dataset \
        --strategy energy_match

events.json:
    {
      "robot_actual_initial": [0.0, 0.0, 0.0],
      "robot_neutral_pose": [0.0, 0.0, 0.0],
      "events": [
        {"state": "speaking", "fragment": {与训练样本兼容的输入, "audio_path" 相对 data_root}},
        {"state": "silent", "duration": 1.5},
        {"state": "listening", "fragment": {...}},
        {"state": "silent", "duration": 2.0}
      ]
    }

输出（--output-dir）：
    unified_relative_rpy.npy   [T, 3] 弧度（相对 robot_actual_initial，首帧 0）
    states.npy                 [T] 0=silent 1=speaking 2=listening
    summary.json               每段元信息（状态/时长/策略/选中候选/幅度/能量）
    candidates_<i>.npy         每段候选池（供 A/B 测试）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.neck_motion.cvae import build_model as build_model_any
from models.neck_motion.dataset import build_inference_batch
from models.neck_motion.rotations import matrix_to_rpy, rpy_to_matrix
from models.neck_motion.text_features import Vocab

RAD2DEG = 180.0 / np.pi
STATE_ID = {"silent": 0, "speaking": 1, "listening": 2}


# --------------------------------------------------------------------------- #
# 候选选择策略
# --------------------------------------------------------------------------- #
def candidate_energy(pred: np.ndarray) -> float:
    """候选运动能量：帧间速度均值（°/s）。"""
    if pred.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(pred, axis=0)).mean()) * RAD2DEG * 30.0


def select_candidate(preds: np.ndarray, strategy: str, rng: np.random.Generator,
                     expected_energy: float | None = None, style_idx: int = 0) -> tuple[int, dict]:
    """从候选池 [K, N, 3] 选择一条。返回 (index, meta)。"""
    K = preds.shape[0]
    energies = np.array([candidate_energy(p) for p in preds])
    amps = np.array([float(np.abs(p).mean()) * RAD2DEG for p in preds])
    if strategy == "first":
        idx = 0
    elif strategy == "random":
        idx = int(rng.integers(K))
    elif strategy == "style_fixed":
        idx = style_idx % K
    elif strategy == "energy_match":
        if expected_energy is None:
            expected_energy = float(np.median(energies))
        idx = int(np.argmin(np.abs(energies - expected_energy)))
    else:  # pragma: no cover
        raise ValueError(f"未知策略: {strategy}")
    meta = {"candidate_idx": int(idx), "amp_deg": float(amps[idx]),
            "energy_deg_per_s": float(energies[idx]),
            "cand_energy_range_deg_per_s": [float(energies.min()), float(energies.max())]}
    return idx, meta


# --------------------------------------------------------------------------- #
# 段生成与拼接
# --------------------------------------------------------------------------- #
def generate_segment(model, batch: dict, strategy: str, rng, k: int,
                     expected_energy: float | None, style_idx: int, device,
                     fixed_z: torch.Tensor | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    with torch.no_grad():
        preds = model.sample(batch, n_samples=k, fixed_z=fixed_z)  # [1, K, N, 3]
    preds = preds[0].cpu().numpy().astype(np.float32)
    idx, meta = select_candidate(preds, strategy, rng, expected_energy, style_idx)
    return preds[idx], meta, preds


def silent_segment(n_frames: int) -> np.ndarray:
    """Silent 段：保持（相对段起点为零轨迹——回中由统一坐标合成体现）。"""
    return np.zeros((n_frames, 3), dtype=np.float32)


def generate(events_spec: dict, checkpoint: str, output_dir: str | None = None,
             data_root: str | None = None,
             device: str | None = None, strategy: str = "random",
             style_idx: int = 0, style_z_path: str | None = None,
             expected_energy: float | None = None, num_candidates: int = 8,
             seed: int = 42, blend_sec: float = 0.3,
             export_json: str | None = None):
    """MVP 三态轨迹生成（可 import 调用；main() 为 CLI 包装）。

    返回 (unified_relative_rpy [T,3] 弧度, states [T], summaries, doc|None)。
    仅生成不落盘；若 export_json 提供则 doc 非 None（调用方自行写盘）。
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("model", {}).get("type", "regression") != "candidates":
        raise SystemExit("[mvp] 需要多候选模型 checkpoint（model.type=candidates），"
                         "当前: %s" % cfg.get("model", {}).get("type", "regression"))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(data_root or cfg["data"]["data_root"])

    vocab_path = Path(ckpt["vocab_path"])
    if not vocab_path.exists():
        vocab_path = Path(checkpoint).parent / vocab_path.name
    vocab = Vocab.load(vocab_path)

    events = events_spec["events"]
    init_rpy = np.asarray(events_spec.get("robot_actual_initial", [0.0, 0.0, 0.0]), dtype=np.float64)
    if init_rpy.shape != (3,):
        raise ValueError("robot_actual_initial 必须是 [roll, pitch, yaw]")
    neutral_rpy = np.asarray(events_spec.get("robot_neutral_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
    if neutral_rpy.shape != (3,):
        raise ValueError("robot_neutral_pose 必须是 [roll, pitch, yaw]（机器人标定中位）")

    model = build_model_any(cfg, vocab_size=len(vocab)).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.manual_seed(seed)          # 固定候选采样（同 seed 跨运行可复现）
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    out_dir = Path(output_dir) if output_dir else \
        (Path(cfg["train"]["output_dir"]) / "mvp" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # style_fixed 的固定 latent 模板（同一风格跨段/跨调用可复现）
    style_z = None
    if strategy == "style_fixed":
        if style_z_path and Path(style_z_path).exists():
            style_z = np.load(style_z_path).astype(np.float32)
            if style_z.shape != (num_candidates, model.z_dim):
                raise ValueError(f"style latent 形状 {style_z.shape} != [K={num_candidates}, z_dim={model.z_dim}]")
            print(f"[mvp] 使用固定 latent 模板: {style_z_path}")
        else:
            style_z = np.random.default_rng(0).standard_normal((num_candidates, model.z_dim)).astype(np.float32)
            z_path = out_dir / "style_z.npy"
            np.save(z_path, style_z)
            print(f"[mvp] 已生成固定 latent 模板: {z_path}（seed=0，风格可复现）")
        style_z = torch.from_numpy(style_z)

    fps = 30.0
    unified: list[np.ndarray] = []      # 统一相对（相对会话初始姿态）
    states: list[np.ndarray] = []
    summaries: list[dict] = []
    candidates_all: list[np.ndarray] = []
    R_abs = rpy_to_matrix(init_rpy)     # 会话当前绝对姿态（段起点）
    R0 = R_abs.copy()                   # 会话初始（坐标系基准 = robot_actual_initial）
    R_neutral = rpy_to_matrix(neutral_rpy)          # Silent 回中目标（标定中位）
    neutral_unified = matrix_to_rpy(R0.T @ R_neutral)  # 统一坐标中的回中目标（(3,)）
    t_total = 0.0

    for i, ev in enumerate(events):
        st = ev["state"]
        if st in ("speaking", "listening"):
            frag = ev["fragment"]
            batch = build_inference_batch(frag, vocab, data_root, int(cfg["data"]["target_sr"]))
            N = batch["num_frames"]
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            expected = expected_energy
            if expected is None and strategy == "energy_match":
                # 启发式期望：speaking 取候选能量中位数；listening 取 P25（更安静）
                _, _, cands = generate_segment(model, batch, "first", rng,
                                               num_candidates, None, 0, dev, fixed_z=style_z)
                en = np.array([candidate_energy(c) for c in cands])
                expected = float(np.median(en) if st == "speaking" else np.percentile(en, 25))
                idx, meta = select_candidate(cands, strategy, rng, expected, style_idx)
                rel = cands[idx]
                meta["expected_energy_deg_per_s"] = expected
            else:
                rel, meta, cands = generate_segment(model, batch, strategy, rng,
                                                    num_candidates, expected, style_idx, dev,
                                                    fixed_z=style_z)
            # 段起点姿态合成：R_abs(t) = R_start @ R(rel(t))
            R_seg = R_abs @ rpy_to_matrix(rel)              # [N,3,3]
            unif = matrix_to_rpy(R0.T @ R_seg)              # [N,3]
            R_abs = R_seg[-1]
            n = len(rel)
        elif st == "silent":
            dur = float(ev.get("duration", 1.0))
            n = max(int(round(dur * fps)), 1)
            # 回中到 robot_neutral_pose（标定中位，统一坐标中的 neutral_unified）
            start_pose = matrix_to_rpy(R0.T @ R_abs) if i > 0 else neutral_unified
            unif = np.linspace(start_pose, neutral_unified, n).astype(np.float32)
            rel = np.zeros((n, 3), dtype=np.float32)
            R_abs = R_neutral  # 回中后绝对姿态 = 标定中位
            meta = {"duration_sec": dur}
            cands = np.zeros((0, 0, 3), dtype=np.float32)
        else:  # pragma: no cover
            raise ValueError(f"未知状态: {st}")

        # 与上一段边界混合（统一坐标空间）
        if unified and blend_sec > 0 and n > 1:
            m = min(int(round(blend_sec * fps)), len(unified[-1]) // 2, n // 2)
            if m > 1:
                w = np.linspace(0.0, 1.0, m)
                prev_tail = unified[-1][-m:]
                unif[:m] = prev_tail * (1 - w)[:, None] + unif[:m] * w[:, None]
                unified[-1] = unified[-1][:-m]   # 前段去掉被混合的尾部，避免重复
                states[-1] = states[-1][:-m]     # 状态序列同步截断
                summaries[-1]["duration_sec"] = round(len(unified[-1]) / fps, 3)

        unified.append(unif)
        states.append(np.full(n, STATE_ID[st], dtype=np.int8))
        candidates_all.append(cands)
        summaries.append({"index": i, "state": st, "start_sec": round(t_total, 3),
                          "duration_sec": round(n / fps, 3), **meta})
        t_total += n / fps

    traj = np.concatenate(unified, axis=0).astype(np.float32)
    state_seq = np.concatenate(states, axis=0)
    if not np.isfinite(traj).all():
        raise RuntimeError("输出轨迹包含 NaN/Inf")

    # 自描述轨迹 JSON 文档（不写盘，由调用方决定）
    doc = None
    if export_json is not None:
        max_frame_rate = float(np.abs(np.diff(traj, axis=0)).max(axis=1).max()) * RAD2DEG * fps if len(traj) > 1 else 0.0
        doc = {
            "format_version": "1.0",
            "trajectory": {
                "fps": fps,
                "units": "radian",
                "rpy_order": "roll,pitch,yaw",
                "rotation_convention": "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)",
                "reference": "unified relative to robot_actual_initial (per-segment start poses composed; frame 0 is [0,0,0])",
                "n_frames": int(len(traj)),
                "robot_actual_initial": init_rpy.tolist(),
                "robot_neutral_pose": neutral_rpy.tolist(),
                "states": state_seq.tolist(),
                "states_legend": {"0": "silent", "1": "speaking", "2": "listening"},
                "rpy": traj.tolist(),
            },
            "meta": {
                "model_type": cfg.get("model", {}).get("type"),
                "checkpoint": str(checkpoint),
                "strategy": strategy,
                "seed": seed,
                "num_candidates": num_candidates,
                "blend_sec": blend_sec,
                "max_frame_rate_deg_per_s": round(max_frame_rate, 3),
                "segments": summaries,
                "generated_by": "models/neck_motion/mvp.py",
            },
        }
    # 落盘（与旧 main() 行为一致）
    np.save(out_dir / "unified_relative_rpy.npy", traj)
    np.save(out_dir / "states.npy", state_seq)
    for i, c in enumerate(candidates_all):
        if c.size:
            np.save(out_dir / f"candidates_{i}.npy", c)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"strategy": strategy, "seed": seed,
                   "num_candidates": num_candidates,
                   "robot_actual_initial": init_rpy.tolist(),
                   "robot_neutral_pose": neutral_rpy.tolist(),
                   "total_sec": round(t_total, 3), "segments": summaries},
                  f, ensure_ascii=False, indent=1)
    print(f"[mvp] 策略={strategy} | 总时长 {t_total:.2f}s | {len(events)} 段")
    for s in summaries:
        extra = ""
        if "candidate_idx" in s:
            extra = (f"候选 {s['candidate_idx']} | 幅度 {s['amp_deg']:.2f}° | "
                     f"能量 {s['energy_deg_per_s']:.2f}°/s (池 {s['cand_energy_range_deg_per_s'][0]:.1f}–"
                     f"{s['cand_energy_range_deg_per_s'][1]:.1f})")
        if "expected_energy_deg_per_s" in s:
            extra += f" | 期望能量 {s['expected_energy_deg_per_s']:.1f}°/s"
        print(f"  [{s['index']}] {s['state']:<9} {s['duration_sec']:5.2f}s @ {s['start_sec']:6.2f}s  {extra}")
    print(f"[mvp] 已保存: {out_dir / 'unified_relative_rpy.npy'}（shape={traj.shape}，"
          f"相对 robot_actual_initial，首帧 [0,0,0]）")
    print(f"[mvp] 注意：best-of-K 为 oracle 指标，线上效果取决于候选策略与实机 A/B 测试。")
    return traj, state_seq, summaries, doc


def main() -> None:
    ap = argparse.ArgumentParser(description="MVP 三态颈部动作调度")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--events", required=True, help="事件序列 JSON")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--strategy", choices=["first", "random", "style_fixed", "energy_match"],
                    default="random")
    ap.add_argument("--style-idx", type=int, default=0)
    ap.add_argument("--style-z-path", default=None,
                    help="style_fixed 的固定 latent 模板 [K, z_dim] npy；缺省自动生成并保存到输出目录")
    ap.add_argument("--expected-energy", type=float, default=None, help="°/s，energy_match 用")
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blend-sec", type=float, default=0.3, help="段边界线性混合时长")
    ap.add_argument("--export-json", default=None,
                    help="输出自描述轨迹 JSON（供外部校验层 loadNeckTrajectoryJson 测试）")
    args = ap.parse_args()
    with open(args.events, "r", encoding="utf-8") as f:
        spec = json.load(f)

    traj, state_seq, summaries, doc = generate(
        spec, args.checkpoint, args.output_dir, args.data_root, args.device, args.strategy,
        args.style_idx, args.style_z_path, args.expected_energy,
        args.num_candidates, args.seed, args.blend_sec, args.export_json)

    # 导出自描述轨迹 JSON（若要求）
    if doc is not None:
        json_path = Path(args.export_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"[mvp] 轨迹 JSON 已导出: {json_path}（n_frames={len(traj)}，供校验层测试）")


if __name__ == "__main__":
    main()
