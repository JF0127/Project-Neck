#!/usr/bin/env python
"""条件预测信号探针 v2（可证伪实验）：Speaker 与 Listener 分开评估。

任务定义（产品语义对齐）：
    speaker   输入当前话语（同步） -> 预测说话期间 Speaker 运动标签（同步标签）
    listener  输入对方完整话语 -> 预测"对方最后一个词结束后 0.2s 起的反应窗口"
              内 Listener 运动标签（后反应标签，窗口 [end+0.2, min(end+2.0, 窗尾)]，
              实际可用约 0.3s——数据管线只保留末词后 0.5s 的轨迹）
              - 只保留反应窗口内无他人发言干扰的样本（同 dialogue 其他 fragment
                词区间与反应窗口重叠 > 0.05s 即剔除）
              - 反应段用绝对轨迹重新相对化（R(段起点)^T @ R_abs[t]，首帧 ≈ 0）

条件组（低维统计表征——结论仅限"现有低维表征未检测到增量"）：
    role_only  时长（+role，listener 任务中 role 为常数）
    audio      当前音频 RMS 能量包络统计
    text       词数/词率 + 词向量均值
    other      对方头部 RPY 统计（部署时相机可观测）
    full       全部

多随机种子（--seeds）训练/评估，报告 mean ± std。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.neck_motion.audio_features import load_audio_waveform_cached
from models.neck_motion.overlap_filter import load_fragments, load_true_overlap, word_span_abs
from models.neck_motion.rotations import relative_segment
from models.neck_motion.text_features import Vocab, encode_word_timestamps

TASKS = ["nod", "shake", "direction", "energy"]
GROUPS = ["role_only", "audio", "text", "other", "full"]
REACT_START_DELAY = 0.2   # 反应窗口起点：末词结束后 0.2s
REACT_MAX_LEN = 2.0       # 反应窗口目标长度上限
REACT_MIN_LEN = 0.3       # 反应窗口最低有效长度（秒）


# --------------------------------------------------------------------------- #
# 特征 / 标签
# --------------------------------------------------------------------------- #
def rms_envelope_stats(wav: np.ndarray, hop: int = 160) -> np.ndarray:
    """波形 RMS 包络统计（@16kHz，hop=10ms）：[mean, std, 峰率, rms]"""
    wav = wav.astype(np.float32)
    n = len(wav)
    if n < hop * 2:
        return np.zeros(4, dtype=np.float32)
    frames = wav[: (n // hop) * hop].reshape(-1, hop)
    env = np.sqrt((frames ** 2).mean(axis=1))
    env = env / (env.max() + 1e-8)
    thr = env.mean() + env.std()
    peaks = 0
    for i in range(1, len(env) - 1):
        if env[i] > env[i - 1] and env[i] > env[i + 1] and env[i] > thr:
            peaks += 1
    dur_sec = n / 16000.0
    return np.array([env.mean(), env.std(), peaks / max(dur_sec, 1e-6), float(np.sqrt((wav ** 2).mean()))], dtype=np.float32)


def rpy_stats(rpy: np.ndarray) -> np.ndarray:
    """RPY 轨迹统计：[mean|rpy|, std|rpy|, energy, pitch过零, yaw过零,
    yaw净位移, pitch净位移, max|Δpitch|]"""
    rpy = rpy.astype(np.float32)
    if rpy.shape[0] < 2:
        return np.zeros(8, dtype=np.float32)
    d = np.diff(rpy, axis=0)
    def crossings(x):
        x = x - x.mean()
        return int(((x[:-1] * x[1:]) < 0).sum())
    return np.array([
        float(np.abs(rpy).mean()), float(rpy.std()),
        float(np.abs(d).mean()),
        crossings(rpy[:, 1]), crossings(rpy[:, 2]),
        float(rpy[-1, 2] - rpy[0, 2]), float(rpy[-1, 1] - rpy[0, 1]),
        float(np.abs(d[:, 1]).max()),
    ], dtype=np.float32)


def extract_labels(rpy: np.ndarray, nod_thr_rad: float = 0.045, min_zc: int = 2) -> dict:
    """轨迹 -> 标签：nod / shake / direction(左0 右1 静止2) / energy(log rad/帧)"""
    rpy = rpy.astype(np.float32)
    if rpy.shape[0] < 2:
        rpy = np.zeros((2, 3), dtype=np.float32)
    def nod_like(x):
        x = x - x.mean()
        zc = int(((x[:-1] * x[1:]) < 0).sum())
        return int(zc >= min_zc and (x.max() - x.min()) > nod_thr_rad)
    dy = rpy[-1, 2] - rpy[0, 2]
    dirc = 0 if dy < -0.035 else (1 if dy > 0.035 else 2)
    energy = float(np.abs(np.diff(rpy, axis=0)).mean())
    return {"nod": nod_like(rpy[:, 1]), "shake": nod_like(rpy[:, 2]),
            "direction": dirc, "energy": float(np.log(energy + 1e-6))}


def load_split_samples(data_root: Path, split: str, true_map: dict[str, float], thr: float) -> list[dict]:
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


def build_pose_and_word_maps(data_root: Path):
    """返回 (dialogue_frags, listener_pose, next_word_start)。

    - dialogue_frags: dialogue_id -> [frag, ...]（按窗口起点排序）
    - listener_pose: uid -> listener 绝对 RPY [N,3]（供拼接）
    - next_word_start: uid -> 同 dialogue 中 > 本 fragment 末词结束的最小词开始（绝对秒）
    """
    frags = load_fragments(data_root)
    by_dlg: dict[str, list[dict]] = defaultdict(list)
    for f in frags:
        by_dlg[f["dialogue_id"]].append(f)
    for lst in by_dlg.values():
        lst.sort(key=lambda f: (f["window_start_original_sec"], f["turn_index"], f["fragment_index"]))

    word_start_abs: dict[str, np.ndarray] = {}
    last_end_abs: dict[str, float] = {}
    for f in frags:
        wts = f["current_utterance"].get("word_timestamps") or []
        if not wts:
            continue
        w0 = f["window_start_original_sec"]
        starts = np.array([w0 + float(w["start_time"]) for w in wts])
        ends = np.array([w0 + float(w["end_time"]) for w in wts])
        word_start_abs[f["utterance_id"]] = starts
        last_end_abs[f["utterance_id"]] = float(ends.max())

    next_word_start: dict[str, float] = {}
    for f in frags:
        uid = f["utterance_id"]
        if uid not in last_end_abs:
            continue
        le = last_end_abs[uid]
        nxt = np.inf
        for g in by_dlg[f["dialogue_id"]]:
            if g["utterance_id"] == uid:
                continue
            ss = word_start_abs.get(g["utterance_id"])
            if ss is None:
                continue
            cand = ss[ss > le + 1e-6]
            if cand.size:
                nxt = min(nxt, float(cand.min()))
        next_word_start[uid] = nxt

    listener_pose: dict[str, np.ndarray] = {}
    for f in frags:
        p = data_root / f"poses/{f['utterance_id']}_listener_rpy.npy"
        if p.exists():
            listener_pose[f["utterance_id"]] = np.load(p).astype(np.float64)
    return by_dlg, listener_pose, next_word_start


def interpolate_listener_pose(by_dlg, listener_pose, uid: str, t0: float, t1: float) -> np.ndarray | None:
    """从同 dialogue 各 fragment 窗口的 listener 绝对姿态拼接出 [t0, t1] 段（30fps）。

    重叠区取先出现的窗口值；若姿态覆盖不足则返回截断后的段（可能短于请求）。
    """
    dlg = None
    for f in by_dlg.values():
        if any(x["utterance_id"] == uid for x in f):
            dlg = f
            break
    if dlg is None:
        return None
    # 拼接时间轴（去重叠，取先出现）
    ts_all: list[np.ndarray] = []
    rp_all: list[np.ndarray] = []
    for f in dlg:
        p = listener_pose.get(f["utterance_id"])
        if p is None or p.shape[0] == 0:
            continue
        w0 = f["window_start_original_sec"]
        ts = w0 + np.arange(p.shape[0]) / 30.0
        ts_all.append(ts)
        rp_all.append(p)
    if not ts_all:
        return None
    T = np.concatenate(ts_all)
    R = np.concatenate(rp_all, axis=0)
    order = np.argsort(T)
    T, R = T[order], R[order]
    # 去重（保留每个时间点第一个值）
    keep = np.concatenate([[True], T[1:] > T[:-1] + 1e-9])
    T, R = T[keep], R[keep]
    # 目标网格
    n = max(int(round((t1 - t0) * 30.0)), 1)
    grid = t0 + np.arange(n) / 30.0
    ok = (grid >= T[0]) & (grid <= T[-1])
    n_ok = int(ok.sum())
    if n_ok < 3:
        return None
    seg = np.stack([np.interp(grid, T, R[:, c]) for c in range(3)], axis=-1)
    return seg[:n_ok]  # 截断到姿态覆盖范围


def reaction_window(s: dict) -> tuple[int, int, str] | None:
    """（保留：旧版 0.5s padding 限制版，已由连续姿态拼接取代，仅作参考）"""
    wts = s["current_utterance"].get("word_timestamps") or []
    if not wts:
        return None, None, "no_words"
    last_end = max(float(w["end_time"]) for w in wts)
    dur = float(s.get("duration_sec", s["num_frames"] / 30.0))
    N = int(s["num_frames"])
    t0 = last_end + REACT_START_DELAY
    t1 = min(last_end + REACT_MAX_LEN, dur)
    f0 = int(np.ceil(t0 * 30.0))
    f1 = min(int(round(t1 * 30.0)), N)
    if f1 - f0 < 3:
        return None, None, "window_too_short"
    return f0, f1, "ok"


def build_interfere_map(data_root: Path) -> dict[str, list[tuple[float, float]]]:
    """（保留：旧版干扰过滤，已由 next_word_start 限制取代）"""
    frags = load_fragments(data_root)
    by_dlg: dict[str, list[dict]] = {}
    for f in frags:
        by_dlg.setdefault(f["dialogue_id"], []).append(f)
    out: dict[str, list[tuple[float, float]]] = {}
    for f in frags:
        others = []
        sp = word_span_abs(f)
        if sp is None:
            continue
        for g in by_dlg[f["dialogue_id"]]:
            if g["utterance_id"] == f["utterance_id"]:
                continue
            gs = word_span_abs(g)
            if gs is not None:
                others.append(gs)
        out[f["utterance_id"]] = others
    return out


# --------------------------------------------------------------------------- #
# 单任务探针
# --------------------------------------------------------------------------- #
class ProbeModel(nn.Module):
    def __init__(self, num_feat: int, use_text_embed: bool, vocab_size: int, task: str, embed_dim: int = 64):
        super().__init__()
        self.use_text_embed = use_text_embed
        self.text_embed = nn.Embedding(vocab_size, embed_dim) if use_text_embed else None
        d = num_feat + (embed_dim if use_text_embed else 0)
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.head = nn.Linear(64, 3 if task == "direction" else 1)

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None):
        if self.use_text_embed:
            emb = self.text_embed(ids)
            m = (ids != 0).unsqueeze(-1).float()
            emb = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            x = torch.cat([x, emb], dim=-1)
        return self.head(self.net(x))


def train_probe(task, num_feat, use_text_embed, X, ids, lab, vocab_size, epochs=8, lr=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    model = ProbeModel(num_feat, use_text_embed, vocab_size, task)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = lab[task]
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(X[idx])
            ids_b = torch.from_numpy(ids[idx]) if use_text_embed else None
            out = model(xb, ids_b)
            if task == "direction":
                loss = nn.functional.cross_entropy(out, y[idx])
            elif task == "energy":
                loss = nn.functional.mse_loss(out.squeeze(-1), y[idx])
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_probe(model, task, num_feat, use_text_embed, X, ids, y):
    model.eval()
    outs = []
    for i in range(0, len(X), 512):
        xb = torch.from_numpy(X[i:i + 512])
        ids_b = torch.from_numpy(ids[i:i + 512]) if use_text_embed else None
        outs.append(model(xb, ids_b))
    out = torch.cat(outs)
    y = y.numpy()

    if task == "direction":
        p = out.softmax(-1).numpy()
        yp = p.argmax(-1)
        rec = [np.mean(yp[y == c] == c) if (y == c).sum() else 0.0 for c in range(3)]
        return float(np.mean(rec))
    if task == "energy":
        p = out.squeeze(-1).numpy()
        ss_res = float(((p - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return float(1.0 - ss_res / max(ss_tot, 1e-9))
    # nod / shake：AUROC（Mann-Whitney U）
    p = torch.sigmoid(out.squeeze(-1)).numpy()
    yb = y.astype(bool)
    n1, n2 = yb.sum(), (~yb).sum()
    if n1 == 0 or n2 == 0:
        return 0.5
    order = np.argsort(-p)
    rank = np.empty(len(p)); rank[order] = np.arange(1, len(p) + 1)
    u = n1 * n2 + n1 * (n1 + 1) / 2 - rank[yb].sum()
    return float(u / (n1 * n2))


@torch.no_grad()
def eval_nod_logits(model, num_feat, use_text_embed, X, ids) -> np.ndarray:
    """返回 nod logits（numpy，用于 bootstrap CI）。"""
    model.eval()
    outs = []
    for i in range(0, len(X), 512):
        xb = torch.from_numpy(X[i:i + 512])
        ids_b = torch.from_numpy(ids[i:i + 512]) if use_text_embed else None
        outs.append(model(xb, ids_b))
    return torch.cat(outs).squeeze(-1).numpy()


def auroc_from_scores(p: np.ndarray, y: np.ndarray) -> float:
    yb = y.astype(bool)
    n1, n2 = yb.sum(), (~yb).sum()
    if n1 == 0 or n2 == 0:
        return 0.5
    order = np.argsort(-p)
    rank = np.empty(len(p)); rank[order] = np.arange(1, len(p) + 1)
    u = n1 * n2 + n1 * (n1 + 1) / 2 - rank[yb].sum()
    return float(u / (n1 * n2))


def block_bootstrap_ci(dlg_ids: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray],
                       n_iter: int = 1000, seed: int = 123) -> dict:
    """按 dialogue_id block bootstrap 的 nod AUROC 差值（full − role_only）置信区间。

    scores: group -> [n_val, n_seeds]（nod 概率/logits，已 sigmoid）。
    返回 mean / 95% CI / P(Δ>0)。
    """
    rng = np.random.default_rng(seed)
    dlgs = np.unique(dlg_ids)
    n_dlg = len(dlgs)
    diffs = np.empty(n_iter * scores["role_only"].shape[1])
    for it in range(n_iter):
        idx = np.isin(dlg_ids, rng.choice(dlgs, size=n_dlg, replace=True))
        for s in range(scores["role_only"].shape[1]):
            a = auroc_from_scores(scores["role_only"][idx, s], y[idx])
            b = auroc_from_scores(scores["full"][idx, s], y[idx])
            diffs[it * scores["role_only"].shape[1] + s] = b - a
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "p_gt_0": float((diffs > 0).mean()), "n_boot": int(len(diffs))}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_feature_columns(group: str, role: np.ndarray, dur: np.ndarray,
                          audio: np.ndarray, text_num: np.ndarray,
                          other: np.ndarray) -> np.ndarray:
    role1 = (role == 1).astype(np.float32)
    base = [role1, dur.astype(np.float32)]
    def cat(extra):
        return np.concatenate([np.stack(base, 1)] + [e.astype(np.float32) for e in extra], 1)
    if group == "role_only":
        return np.stack(base, 1)
    if group == "audio":
        return cat([audio])
    if group == "text":
        return cat([text_num])
    if group == "other":
        return cat([other])
    if group == "full":
        return cat([audio, text_num, other])
    raise ValueError(group)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["speaker", "listener"], required=True,
                    help="speaker=同步标签；listener=后反应标签（重建）")
    ap.add_argument("--data-root", default="data/source/response-net/processed_dataset")
    ap.add_argument("--true-overlap", default="outputs/neck_motion/overlap_analysis/true_overlap.json")
    ap.add_argument("--vocab", default="outputs/neck_motion/vocab.json")
    ap.add_argument("--cache", default="outputs/neck_motion/probe_{task}_v2.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5, help="随机种子数（报告 mean±std）")
    args = ap.parse_args()

    root = Path(args.data_root)
    true_map = load_true_overlap(args.true_overlap)
    vocab = Vocab.load(args.vocab)
    cache = Path(args.cache.format(task=args.task))
    task = args.task

    if cache.exists():
        print(f"[probe] 加载特征缓存 {cache}")
        c = np.load(cache, allow_pickle=True)
        feats, labs, meta = c["feats"].item(), c["labs"].item(), c["meta"].item()
    else:
        t0 = time.time()
        by_dlg, listener_pose, next_word_start = build_pose_and_word_maps(root)
        feats, labs, meta = {"train": {}, "val": {}}, {"train": {}, "val": {}}, {}
        for split in ("train", "val"):
            samples = load_split_samples(root, split, true_map, 0.1)
            if task == "listener":
                samples = [s for s in samples if s["role"] == "listener"]
            else:
                samples = [s for s in samples if s["role"] == "speaker"]
            F = {"role": [], "duration": [], "audio": [], "text_num": [], "other": [], "ids": [], "dialogue": []}
            L = {"nod": [], "shake": [], "direction": [], "energy": []}
            n_skip = {"no_words": 0, "no_pose": 0, "window_too_short": 0, "interfere": 0}
            for si, s in enumerate(samples):
                uid = s["sample_id"].rsplit("_", 1)[0]
                other_role = "listener" if s["role"] == "speaker" else "speaker"
                me = np.load(root / s["target_rpy_path"]).astype(np.float32)
                other = np.load(root / f"poses/{uid}_{other_role}_relative_rpy.npy").astype(np.float32)
                N = me.shape[0]
                dlg_id = s.get("dialogue_id") or uid.rsplit("_", 2)[0] + "_" + uid.rsplit("_", 2)[1]

                if task == "listener":
                    # 可变长度反应窗口（原始连续姿态，不受 +0.5s padding 限制）
                    wts = s["current_utterance"].get("word_timestamps") or []
                    if not wts:
                        n_skip["no_words"] += 1
                        continue
                    w0 = float(s.get("window_start_original_sec", 0.0))
                    last_end = w0 + max(float(w["end_time"]) for w in wts)
                    t_start = last_end + REACT_START_DELAY
                    t_end = min(t_start + REACT_MAX_LEN, next_word_start.get(uid, np.inf))
                    seg = interpolate_listener_pose(by_dlg, listener_pose, uid, t_start, t_end)
                    if seg is None or len(seg) < 3 or (len(seg) / 30.0) < 0.1:
                        n_skip["no_pose"] += 1
                        continue
                    if len(seg) / 30.0 < REACT_MIN_LEN:
                        n_skip["window_too_short"] += 1
                        continue
                    lab_traj = relative_segment(seg, 0, len(seg))
                    labels = extract_labels(lab_traj, nod_thr_rad=0.03, min_zc=1)
                    cond_len = N  # 条件 = 对方完整话语
                else:
                    labels = extract_labels(me)
                    cond_len = N

                wav = load_audio_waveform_cached(str(root / s["current_utterance"]["audio_path"])).numpy()[0]
                wav_cond = wav[: int(cond_len / 30.0 * 16000)]
                if len(wav_cond) < 1600:
                    wav_cond = wav
                wts = s["current_utterance"].get("word_timestamps") or []
                ids = vocab.encode_text(s["current_utterance"]["text"])
                F["role"].append(1.0 if s["role"] == "speaker" else 0.0)
                F["duration"].append(cond_len / 30.0)
                F["audio"].append(rms_envelope_stats(wav_cond))
                F["text_num"].append(np.array([len(wts), len(wts) / max(cond_len / 30.0, 1e-6)], dtype=np.float32))
                F["other"].append(rpy_stats(other[:cond_len]))
                F["ids"].append(np.asarray(ids, dtype=np.int64))
                F["dialogue"].append(dlg_id)
                for k, v in labels.items():
                    L[k].append(v)
                if (si + 1) % 3000 == 0:
                    print(f"  [{split}] {si+1}/{len(samples)}，{time.time()-t0:.0f}s")
            feats[split] = {k: (np.stack(v) if k not in ("ids", "dialogue") else v) for k, v in F.items()}
            labs[split] = {k: torch.tensor(np.asarray(v), dtype=torch.long if k == "direction" else torch.float32)
                           for k, v in L.items()}
            print(f"[probe:{task}] {split}: 保留 {len(samples)-sum(n_skip.values())}/{len(samples)} 条"
                  f"（剔除: {n_skip}），{time.time()-t0:.0f}s")
        np.savez(cache, feats=feats, labs=labs, meta=meta)
        print(f"[probe] 特征已缓存: {cache}")

    print("\n标签分布（train）: nod=%.1f%% shake=%.1f%% dir[左/右/静]=%s" % (
        labs["train"]["nod"].mean() * 100, labs["train"]["shake"].mean() * 100,
        [int((labs["train"]["direction"] == c).sum()) for c in range(3)]))

    max_ids = max(len(a) for a in feats["train"]["ids"] + feats["val"]["ids"])
    ids_tr_all = np.stack([np.pad(a, (0, max_ids - len(a))) for a in feats["train"]["ids"]])
    ids_va_all = np.stack([np.pad(a, (0, max_ids - len(a))) for a in feats["val"]["ids"]])

    results: dict[str, dict[str, list[float]]] = {}
    for g in GROUPS:
        Xtr = build_feature_columns(g, feats["train"]["role"], feats["train"]["duration"],
                                    feats["train"]["audio"], feats["train"]["text_num"], feats["train"]["other"])
        Xva = build_feature_columns(g, feats["val"]["role"], feats["val"]["duration"],
                                    feats["val"]["audio"], feats["val"]["text_num"], feats["val"]["other"])
        use_embed = g in ("text", "full")
        for task_name in TASKS:
            vals = []
            for seed in range(args.seeds):
                model = train_probe(task_name, Xtr.shape[1], use_embed, Xtr,
                                    ids_tr_all if use_embed else None, labs["train"], len(vocab),
                                    epochs=args.epochs, seed=seed)
                vals.append(eval_probe(model, task_name, Xtr.shape[1], use_embed, Xva,
                                       ids_va_all if use_embed else None, labs["val"][task_name]))
            results[f"{g}/{task_name}"] = vals
        v = results[f"{g}/nod"]
        print(f"[{task}/{g:<10}] nod AUROC {np.mean(v):.3f}±{np.std(v):.3f} | "
              f"shake {np.mean(results[f'{g}/shake']):.3f}±{np.std(results[f'{g}/shake']):.3f} | "
              f"dir {np.mean(results[f'{g}/direction']):.3f}±{np.std(results[f'{g}/direction']):.3f} | "
              f"energy R² {np.mean(results[f'{g}/energy']):.3f}±{np.std(results[f'{g}/energy']):.3f}")

    ro = {t: (np.mean(results[f"role_only/{t}"]), np.std(results[f"role_only/{t}"])) for t in TASKS}
    print(f"\n=== [{task}] 相对 role_only 的提升（mean±std，n_seeds={args.seeds}） ===")
    for g in GROUPS[1:]:
        parts = []
        for t in TASKS:
            m = np.mean(results[f"{g}/{t}"]) - ro[t][0]
            s = np.hypot(np.std(results[f"{g}/{t}"]), ro[t][1])
            parts.append(f"{t} {m:+.3f}±{s:.3f}")
        print(f"  {g:<10} " + " | ".join(parts))

    # ---------- dialogue block bootstrap：nod AUROC 差值（full − role_only） ----------
    if task == "listener":
        dlg_ids = np.asarray(feats["val"]["dialogue"])
        y_nod = labs["val"]["nod"].numpy()
        scores = {}
        for g in ("role_only", "full"):
            Xv = build_feature_columns(g, feats["val"]["role"], feats["val"]["duration"],
                                       feats["val"]["audio"], feats["val"]["text_num"], feats["val"]["other"])
            use_embed = g == "full"
            s_mat = np.zeros((len(Xv), args.seeds))
            for seed in range(args.seeds):
                Xt = build_feature_columns(g, feats["train"]["role"], feats["train"]["duration"],
                                           feats["train"]["audio"], feats["train"]["text_num"], feats["train"]["other"])
                model = train_probe("nod", Xt.shape[1], use_embed, Xt,
                                    ids_tr_all if use_embed else None, labs["train"], len(vocab),
                                    epochs=args.epochs, seed=seed)
                s_mat[:, seed] = 1.0 / (1.0 + np.exp(-eval_nod_logits(model, Xt.shape[1], use_embed, Xv,
                                                                      ids_va_all if use_embed else None)))
            scores[g] = s_mat
        ci = block_bootstrap_ci(dlg_ids, y_nod, scores, n_iter=1000, seed=1234)
        print(f"\n=== [listener] dialogue block bootstrap：nod AUROC 差值 (full−role_only) ===")
        print(f"  mean={ci['mean']:+.4f} | 95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}] | "
              f"P(Δ>0)={ci['p_gt_0']:.3f}（n_boot={ci['n_boot']}）")

    out_path = cache.with_suffix(".results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"task": task, "seeds": args.seeds, "results": {k: v for k, v in results.items()}},
                  f, ensure_ascii=False, indent=1)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
