#!/usr/bin/env python
"""Listener 反应窗口原始时间轴审计。

问题：数据管线只保留"末词后 +0.5s"的 fragment 窗口，导致反应窗口被压缩到
0.3s。本脚本从原始时间轴出发：

1. 每段话语（fragment）末词结束 -> 同 dialogue 中下一个词开始（任何人开口）
   的真实空隙分布；
2. 可变长度反应窗口：start = 末词结束 + 0.2s；
   end = min(start + 2.0s, 下一人开口前)；最低有效长度 0.3s；
3. 姿态可用性：同 dialogue 中 listener 绝对姿态轨迹（各 fragment 窗口内，
   时间上相邻可拼接）对反应窗口的覆盖长度——不受 +0.5s padding 限制。

输出：outputs/neck_motion/reaction_audit.json + 控制台报告。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neck_motion.overlap_filter import load_fragments

START_DELAY = 0.2
MAX_LEN = 2.0
MIN_LEN = 0.3


def main() -> None:
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/source/response-net/processed_dataset")
    frags = load_fragments(data_root)

    # split 归属（train/val/test.jsonl 的 uid 集合）
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        ids = set()
        with open(data_root / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["sample_id"].rsplit("_", 1)[0])
        split_ids[split] = ids

    by_dlg: dict[str, list[dict]] = defaultdict(list)
    for f in frags:
        by_dlg[f["dialogue_id"]].append(f)

    # 每个 fragment：绝对词区间 / 末词结束 / 下一词开始
    word_start_abs: dict[str, np.ndarray] = {}   # uid -> 该 fragment 所有词的绝对 start
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

    gaps: list[tuple[str, float, float]] = []   # (split, gap_sec, 可用反应长度)
    for f in frags:
        uid = f["utterance_id"]
        if uid not in last_end_abs:
            continue
        le = last_end_abs[uid]
        # 同 dialogue 中下一个词开始（排除自己 fragment 的词）
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
        gap = nxt - le
        start = le + START_DELAY
        end = min(start + MAX_LEN, nxt)
        avail = max(end - start, 0.0)
        split = next((s for s in ("train", "val", "test") if uid in split_ids[s]), "?")
        gaps.append((split, gap, avail))

    edges = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, np.inf]
    labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", "1.0-2.0", ">2.0/无下一词"]
    print("=== 话语结束后到下一人开口的真实空隙分布（全部 fragment） ===")
    g_all = np.array([g for _, g, _ in gaps])
    hist, _ = np.histogram(g_all, bins=edges)
    for lab, h in zip(labels, hist):
        print(f"  空隙 {lab:>12}s : {h:>5} ({h/len(gaps)*100:5.1f}%)")
    print(f"  空隙 ≥0.2s（可构建反应窗口）: {(g_all>=0.2).sum()} ({(g_all>=0.2).mean()*100:.1f}%)")

    print("\n=== 可变长度反应窗口（start=末词+0.2s, end=min(start+2.0, 下一人开口)） ===")
    for split in ("train", "val"):
        av = np.array([a for s, _, a in gaps if s == split])
        n_ok = int((av >= MIN_LEN).sum())
        print(f"  {split}: 话语数={len(av)}，窗口长度 mean={av.mean():.2f}s median={np.median(av):.2f}s，"
              f"≥{MIN_LEN}s 可用={n_ok} ({n_ok/len(av)*100:.1f}%)")
        for lo, hi, lab in ((0.3, 0.5, "0.3-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"), (2.0, 2.01, "=2.0(完整)")):
            n = int(((av >= lo) & (av < hi)).sum())
            print(f"      [{lab}s]: {n}")

    # 保存
    out = {
        "start_delay": START_DELAY, "max_len": MAX_LEN, "min_len": MIN_LEN,
        "gap_histogram": {lab: int(h) for lab, h in zip(labels, hist)},
        "gap_ge_0.2": int((g_all >= 0.2).sum()),
        "window_by_split": {s: {
            "utterances": int((np.array([a for sp, _, a in gaps if sp == s]) >= 0).sum()),
            "usable_ge_min": int((np.array([a for sp, _, a in gaps if sp == s]) >= MIN_LEN).sum()),
        } for s in ("train", "val")},
    }
    out_path = data_root.parent.parent.parent / "outputs" / "neck_motion" / "reaction_audit.json"
    # 直接用绝对路径输出到项目 outputs
    out_path = Path("outputs/neck_motion/reaction_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
