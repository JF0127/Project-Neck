"""文本特征：词表、tokenization、词级时间戳到 30 Hz 帧对齐。

- 词表只从 train split 构建（严禁从 val/test 构建）。
- 基础 tokenization：Unicode NFKC 规范化 + 小写 + 按词切分。
- 保留特殊 token：<PAD> <UNK> <SILENCE> <START_OF_DIALOGUE>。
- 每个词按 word_timestamps.start_time/end_time 对齐到其覆盖的 30 Hz 帧；
  同一帧多个词覆盖时平均池化；无词覆盖的帧使用可学习 <SILENCE> 向量
  （对齐本身在 model.py 中调用 align_words_to_frames 完成）。
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

PAD = "<PAD>"
UNK = "<UNK>"
SILENCE = "<SILENCE>"
START_OF_DIALOGUE = "<START_OF_DIALOGUE>"
SPECIAL_TOKENS = [PAD, UNK, SILENCE, START_OF_DIALOGUE]

_TOKEN_RE = re.compile(r"[\w']+")


def tokenize(text: str | None) -> list[str]:
    """NFKC 规范化 -> 小写 -> 按词切分（保留词内撇号）。"""
    if not text:
        return []
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return _TOKEN_RE.findall(t)


class Vocab:
    """词表。word2idx 中特殊 token 恒在前 4 位。"""

    def __init__(self, word2idx: dict[str, int]):
        self.word2idx = dict(word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        for tok in SPECIAL_TOKENS:
            assert tok in self.word2idx, f"缺少特殊 token: {tok}"

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIAL_TOKENS))]
        word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for w in words:
            word2idx[w] = len(word2idx)
        return cls(word2idx)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["word2idx"])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ #
    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx[UNK]
        return [self.word2idx.get(t, unk) for t in tokens]

    def encode_text(self, text: str | None) -> list[int]:
        """文本 -> token id 列表。None/空文本 -> 空列表。"""
        return self.encode(tokenize(text))

    def __len__(self) -> int:
        return len(self.word2idx)


def encode_word_timestamps(
    vocab: Vocab,
    word_timestamps: list[dict],
    num_frames: int,
    fps: float = 30.0,
) -> tuple[list[int], list[int], list[int], int]:
    """把词级时间戳（相对窗口起点，秒）编码为 (词 id, 帧区间) 列表，严格 1:1。

    帧 t 覆盖时间 [t/fps, (t+1)/fps)；词覆盖 floor(start*fps) 到 ceil(end*fps)。
    结果夹紧到 [0, num_frames]；完全落在窗口外或时间戳无效的词被跳过。

    Returns:
        (word_ids, starts, ends, n_skipped)：每词一条，[start_frame, end_frame)。
    """
    ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    n_skipped = 0
    for w in word_timestamps or []:
        toks = tokenize(w.get("text") or "")
        if not toks:
            n_skipped += 1
            continue
        s = float(w.get("start_time", 0.0))
        e = float(w.get("end_time", s))
        if not math.isfinite(s) or not math.isfinite(e) or e <= s:
            n_skipped += 1
            continue
        f0 = int(math.floor(s * fps))
        f1 = int(math.ceil(e * fps))
        f1 = max(f1, f0 + 1)
        if f1 <= 0 or f0 >= num_frames:  # 完全在窗口外
            n_skipped += 1
            continue
        ids.append(vocab.encode([toks[0]])[0])
        starts.append(max(f0, 0))
        ends.append(min(f1, num_frames))
    return ids, starts, ends, n_skipped


def align_words_to_frames(
    embedding: nn.Embedding,
    silence: torch.Tensor,
    word_ids: list[list[int]],
    starts: list[list[int]],
    ends: list[list[int]],
    n_frames: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """把每个样本的词向量对齐到其覆盖的 30 Hz 帧（批内循环，B 通常 ≤ 16）。

    - 同一帧多个词覆盖 -> 平均池化。
    - 无词覆盖的帧 -> 可学习 <SILENCE> 向量。
    - 额外返回二值 coverage 特征 [B, Nmax, 1]。

    Returns:
        word_feat [B, Nmax, word_embed_dim]
        coverage  [B, Nmax, 1]（float 0/1）
    """
    device = embedding.weight.device
    B = len(word_ids)
    n_frames = [int(n) for n in n_frames]
    n_max = max(n_frames) if n_frames else 0
    embed_dim = embedding.weight.shape[1]

    word_feat = torch.zeros(B, n_max, embed_dim, device=device)
    count = torch.zeros(B, n_max, device=device)

    for i in range(B):
        ids, ss, ee = word_ids[i], starts[i], ends[i]
        if not ids:
            continue
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        emb = embedding(ids_t)  # [W, D]
        n_i = n_frames[i]
        for j in range(len(ids_t)):
            s, e = ss[j], ee[j]
            if e <= s or s >= n_i:
                continue
            word_feat[i, s:e] += emb[j]
            count[i, s:e] += 1.0

    coverage = (count > 0).unsqueeze(-1).float()  # [B, Nmax, 1]
    denom = count.clamp(min=1.0).unsqueeze(-1)
    word_feat = word_feat / denom  # 平均池化
    # 无词覆盖的帧替换为可学习 <SILENCE> 向量
    word_feat = torch.where(coverage > 0, word_feat, silence.view(1, 1, embed_dim))
    return word_feat, coverage
