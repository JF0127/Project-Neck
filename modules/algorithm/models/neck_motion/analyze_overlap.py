#!/usr/bin/env python
"""分析重叠分布并生成 true_overlap 过滤映射。

输出到 --output-dir：
    overlap_summary.json    窗口重叠（旧定义）分布与边界特征
    filtered_fragments.csv  旧定义过滤的 fragment 明细
    true_overlap.json       uid -> true_overlap 秒（新过滤映射，不含窗口 padding）
    true_overlap_summary.json  新旧过滤对比统计
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.neck_motion.overlap_filter import compute_true_overlap, load_fragments, save_true_overlap

def load_split_ids(data_root: Path) -> dict[str, set[str]]:
    """split -> utterance_id 集合（train/val/test.jsonl 中的样本去重）。"""
    out: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        ids: set[str] = set()
        with open(data_root / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["sample_id"].rsplit("_", 1)[0])
        out[split] = ids
    return out


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/source/response-net/processed_dataset")
    ap.add_argument("--output-dir", default="outputs/neck_motion/overlap_analysis")
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thr = args.threshold

    frags = load_fragments(root)
    split_ids = load_split_ids(root)
    frag_split = {}
    for split, ids in split_ids.items():
        for uid in ids:
            frag_split[uid] = split

    ov = np.array([f.get("overlap_other_speaker_sec") or 0.0 for f in frags])
    dur = np.array([f["duration_sec"] for f in frags])

    # ---- 按 dialogue 内 (turn_index, fragment_index) 排序，找"下一个 fragment" ----
    by_dlg: dict[str, list[dict]] = defaultdict(list)
    for f in frags:
        by_dlg[f["dialogue_id"]].append(f)
    for lst in by_dlg.values():
        lst.sort(key=lambda f: (f["turn_index"], f["fragment_index"]))

    def next_info(f: dict):
        lst = by_dlg[f["dialogue_id"]]
        i = lst.index(f)
        if i + 1 >= len(lst):
            return None, None, None, None
        nxt = lst[i + 1]
        gap = nxt["window_start_original_sec"] - f["window_end_original_sec"]
        same_speaker = nxt["current_utterance"]["speaker_id"] == f["current_utterance"]["speaker_id"]
        return nxt, gap, same_speaker, (nxt["turn_index"] != f["turn_index"])

    # ---- 汇总统计 ----
    def describe(name: str, arr: np.ndarray) -> dict:
        return {
            "name": name,
            "count": int(len(arr)),
            "mean": fmt(float(arr.mean())),
            "median": fmt(float(np.median(arr))),
            "p90": fmt(float(np.percentile(arr, 90))),
            "p95": fmt(float(np.percentile(arr, 95))),
            "p99": fmt(float(np.percentile(arr, 99))),
            "max": fmt(float(arr.max())),
        }

    summary = {
        "threshold_sec": thr,
        "total_fragments": len(frags),
        "filtered_fragments": int((ov > thr).sum()),
        "filtered_ratio": fmt(float((ov > thr).mean())),
        "all": describe("全部 fragment", ov),
        "kept": describe("保留 (<=阈值)", ov[ov <= thr]),
        "filtered": describe("过滤 (>阈值)", ov[ov > thr]),
    }

    # 分 bin 直方图
    bins = [0.0, 0.01, 0.05, thr, 0.2, 0.5, 1.0, 2.0, 5.0, max(5.0, float(ov.max()) + 1e-6)]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    hist, _ = np.histogram(ov, bins=bins)
    summary["histogram"] = {lab: int(h) for lab, h in zip(labels, hist)}

    # ---- 过滤集：边界特征 ----
    rows = []
    turn_last_share = 0
    gap_to_next: list[float] = []
    gap_lt_0_8 = 0
    n_with_next = 0
    dur_corr = None
    filt_dur = []
    for f, o in zip(frags, ov):
        if o <= thr:
            continue
        nxt, gap, same_speaker, new_turn = next_info(f)
        is_turn_last = nxt is None or new_turn
        if nxt is not None:
            n_with_next += 1
            gap_to_next.append(float(gap))
            # 下一说话者开头词约在 next.window_start + 0.3s（pre-pad）：
            # 若本窗口后垫 0.5s 覆盖到它 -> 边界发言而非同时说话
            if gap < 0.8:
                gap_lt_0_8 += 1
        if is_turn_last:
            turn_last_share += 1
        filt_dur.append(float(f["duration_sec"]))
        rows.append({
            "utterance_id": f["utterance_id"],
            "split": frag_split.get(f["utterance_id"], "?"),
            "dialogue_id": f["dialogue_id"],
            "turn_index": f["turn_index"],
            "fragment_index": f["fragment_index"],
            "turn_last": int(is_turn_last),
            "overlap_other_speaker_sec": round(o, 4),
            "duration_sec": round(float(f["duration_sec"]), 3),
            "window_start_original_sec": f["window_start_original_sec"],
            "window_end_original_sec": f["window_end_original_sec"],
            "next_window_gap_sec": round(gap, 3) if nxt is not None else None,
            "next_same_speaker": None if nxt is None else int(same_speaker),
        })
    if gap_to_next:
        g = np.array(gap_to_next)
        summary["filtered_turn_boundary"] = {
            "turn_last_fraction": fmt(turn_last_share / len(rows)),
            "n_with_next_fragment": n_with_next,
            "next_window_gap": describe("到下一窗口间隔", g),
            "next_gap_lt_0_8s_fraction": fmt(gap_lt_0_8 / n_with_next),
        }
    if len(filt_dur) > 1:
        dur_corr = float(np.corrcoef(filt_dur, [r["overlap_other_speaker_sec"] for r in rows])[0, 1])
        summary["filtered_overlap_vs_duration_corr"] = fmt(dur_corr)

    # 各 split 过滤量（旧定义）
    per_split: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        ids = split_ids[split]
        if ids:
            o_s = np.array([f.get("overlap_other_speaker_sec") or 0.0 for f in frags if f["utterance_id"] in ids])
            per_split[split] = {
                "fragments": int(len(ids)),
                "filtered": int((o_s > thr).sum()),
                "filtered_ratio": fmt(float((o_s > thr).mean())),
            }
    summary["per_split"] = per_split

    # ================================================================== #
    # 新定义：true_overlap（实际词区间交集，不含窗口 padding）
    # ================================================================== #
    true_map = compute_true_overlap(frags)
    true_path = out / "true_overlap.json"
    save_true_overlap(true_path, true_map)
    print(f"true_overlap 映射已保存: {true_path}")

    to = np.array([true_map[f["utterance_id"]] for f in frags])
    old_filtered = ov > thr
    new_filtered = to > thr
    both = old_filtered & new_filtered
    old_only = old_filtered & ~new_filtered
    new_only = ~old_filtered & new_filtered

    # 新定义下各 split 过滤量
    per_split_new: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        ids = split_ids[split]
        if ids:
            idx = np.array([f["utterance_id"] in ids for f in frags])
            n_split = int(idx.sum())
            n_filt = int((new_filtered & idx).sum())
            per_split_new[split] = {
                "fragments": n_split,
                "filtered": n_filt,
                "filtered_ratio": fmt(n_filt / n_split),
            }
    # 新定义过滤集的 true_overlap 分布
    true_summary = {
        "threshold_sec": thr,
        "total_fragments": len(frags),
        "old_definition": {
            "filtered": int(old_filtered.sum()),
            "ratio": fmt(float(old_filtered.mean())),
            "per_split": per_split,
        },
        "new_definition": {
            "filtered": int(new_filtered.sum()),
            "ratio": fmt(float(new_filtered.mean())),
            "per_split": per_split_new,
            "true_overlap_dist": describe("true_overlap 过滤集", to[new_filtered]),
        },
        "old_vs_new": {
            "both": int(both.sum()),
            "old_only_restored": int(old_only.sum()),   # 旧误删，新定义保留
            "new_only": int(new_only.sum()),            # 旧漏删，新定义删除
        },
    }
    with open(out / "true_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(true_summary, f, ensure_ascii=False, indent=1)

    # ---- 打印 ----
    print(f"=== overlap_other_speaker_sec 分析（阈值 {thr}s）===")
    print(f"总 fragment: {summary['total_fragments']}，旧定义过滤: {summary['filtered_fragments']} "
          f"({summary['filtered_ratio']}%)")
    print(f"各 split 过滤(旧): " + " | ".join(f"{k}: {v['filtered']}/{v['fragments']} ({v['filtered_ratio']}%)"
                                              for k, v in per_split.items()))

    # ---- 导出 ----
    with open(out / "overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(out / "filtered_fragments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 打印 ----
    print(f"=== overlap_other_speaker_sec 分析（阈值 {thr}s）===")
    print(f"总 fragment: {summary['total_fragments']}，过滤: {summary['filtered_fragments']} "
          f"({summary['filtered_ratio']}%)")
    print(f"各 split 过滤: " + " | ".join(f"{k}: {v['filtered']}/{v['fragments']} ({v['filtered_ratio']}%)"
                                          for k, v in per_split.items()))
    print("\n直方图（全部 4682 个 fragment）:")
    for lab, h in summary["histogram"].items():
        print(f"  overlap {lab:>10}s : {h:>5}")
    print(f"\n过滤集 overlap 分布: {describe('', ov[ov > thr])}")
    tb = summary.get("filtered_turn_boundary", {})
    print(f"\n边界特征（旧定义过滤集）:")
    print(f"  轮次末尾 fragment 占比: {tb.get('turn_last_fraction', 'N/A')}")
    print(f"  到下一窗口间隔 < 0.8s（padding 窗口重叠）占比: {tb.get('next_gap_lt_0_8s_fraction', 'N/A')}")
    print(f"  过滤集 overlap 与 duration 相关系数: {summary.get('filtered_overlap_vs_duration_corr', 'N/A')}")

    print(f"\n=== true_overlap（新定义，不含窗口 padding）对比 ===")
    o, n = true_summary["old_definition"], true_summary["new_definition"]
    print(f"旧定义过滤: {o['filtered']} ({o['ratio']}%) -> 新定义过滤: {n['filtered']} ({n['ratio']}%)")
    print(f"各 split 过滤(新): " + " | ".join(f"{k}: {v['filtered']}/{v['fragments']} ({v['filtered_ratio']}%)"
                                              for k, v in n["per_split"].items()))
    vn = true_summary["old_vs_new"]
    print(f"新旧一致(都过滤): {vn['both']} | 旧误删-新保留: {vn['old_only_restored']} | 新增过滤(旧漏删): {vn['new_only']}")
    print(f"新过滤集 true_overlap 分布: {n['true_overlap_dist']}")
    print(f"\n已导出: {out / 'overlap_summary.json'} | {out / 'filtered_fragments.csv'} | "
          f"{out / 'true_overlap.json'} | {out / 'true_overlap_summary.json'}")


if __name__ == "__main__":
    main()
