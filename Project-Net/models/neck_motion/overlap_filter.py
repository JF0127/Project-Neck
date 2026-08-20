"""真实语音重叠（true_overlap）计算与过滤映射。

定义（不含窗口 padding）：
    true_overlap(fragment) =
        当前 fragment 的实际词区间（首词 start ~ 末词 end，绝对时间轴）
        ∩ 同一 dialogue 中对方角色所有 fragment 的实际词区间
        的交集总时长（秒）

只过滤 true_overlap > 阈值 的样本。窗口的 0.5s 后垫 / 0.3s 前垫不参与判定：
轮次边界处"对方在 padding 区内开口"的正常相邻发言不会被误删。

说明：轮内插话若未形成保留的 fragment（过短被并入/删除），其词区间不可见，
此类样本 true_overlap 记为 0（保守保留，不误删）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_fragments(data_root: str | Path) -> list[dict]:
    """读取 aligned_utterances.jsonl（fragment 级权威数据）。"""
    path = Path(data_root) / "aligned_utterances.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，无法计算 true_overlap")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_span_abs(frag: dict) -> tuple[float, float] | None:
    """fragment 实际词区间（绝对秒）。无词时间戳时返回 None。"""
    wts = frag.get("current_utterance", {}).get("word_timestamps") or []
    if not wts:
        return None
    starts = [float(w["start_time"]) for w in wts]
    ends = [float(w["end_time"]) for w in wts]
    if not starts:
        return None
    w0 = frag["window_start_original_sec"] + min(starts)
    w1 = frag["window_start_original_sec"] + max(ends)
    return w0, w1


def compute_true_overlap(frags: list[dict]) -> dict[str, float]:
    """uid -> true_overlap 秒。对同一 dialogue 内对方角色 fragment 求词区间交集总时长。"""
    by_dlg: dict[str, list[dict]] = {}
    for f in frags:
        by_dlg.setdefault(f["dialogue_id"], []).append(f)

    true_overlap: dict[str, float] = {}
    n_undecidable = 0
    for dlg, lst in by_dlg.items():
        spans: list[tuple[dict, tuple[float, float]]] = []
        for f in lst:
            sp = word_span_abs(f)
            if sp is None:
                n_undecidable += 1
            spans.append((f, sp))
        for f, sp in spans:
            if sp is None:
                true_overlap[f["utterance_id"]] = 0.0
                continue
            ov = 0.0
            for g, gsp in spans:
                if g["utterance_id"] == f["utterance_id"]:
                    continue
                if g["current_utterance"]["speaker_id"] == f["current_utterance"]["speaker_id"]:
                    continue  # 只统计对方角色
                if gsp is None:
                    continue
                lo = max(sp[0], gsp[0])
                hi = min(sp[1], gsp[1])
                if hi > lo:
                    ov += hi - lo
            true_overlap[f["utterance_id"]] = ov
    logger.info("true_overlap 计算完成：%d 个 fragment，%d 个无词信息（记为 0）",
                len(true_overlap), n_undecidable)
    return true_overlap


def load_true_overlap(path: str | Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


def save_true_overlap(path: str | Path, true_overlap: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(true_overlap, f, ensure_ascii=False, sort_keys=True, indent=1)


def ensure_true_overlap(data_root: str | Path, out_path: str | Path) -> dict[str, float]:
    """加载 true_overlap.json；缺失时自动计算并保存（不写原始数据目录）。"""
    out_path = Path(out_path)
    if out_path.exists():
        return load_true_overlap(out_path)
    logger.info("true_overlap.json 不存在，正在从 aligned_utterances.jsonl 计算...")
    frags = load_fragments(data_root)
    m = compute_true_overlap(frags)
    save_true_overlap(out_path, m)
    logger.info("已保存: %s", out_path)
    return m
