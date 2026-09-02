"""ResponseNet 话语级数据集加载器。

- 加载 train/val/test.jsonl + relative/delta RPY + 当前话语音频。
- 按 aligned_utterances.jsonl 中的 overlap_other_speaker_sec 过滤（> 阈值剔除，
  按 fragment 过滤，speaker/listener 两条样本一起剔除）；日志输出过滤数量。
- 音频统一 mono + 16 kHz（内存中完成，不改写 WAV）。
- 路径相对 processed_dataset 解析。
- 返回张量 + metadata（initial_pose 仅作元数据，不进入网络）。
- 词表只在 build_vocab=True（train split）时构建。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from neck_motion.audio_features import load_audio_waveform, load_audio_waveform_cached
from neck_motion.overlap_filter import load_true_overlap
from neck_motion.text_features import (
    START_OF_DIALOGUE,
    Vocab,
    encode_word_timestamps,
    tokenize,
)

logger = logging.getLogger(__name__)

ROLE_TO_ID = {"speaker": 0, "listener": 1}


def sample_to_utterance_id(sample: dict) -> str:
    """sample_id = '<utterance_id>_speaker|listener' -> utterance_id。"""
    return sample["sample_id"].rsplit("_", 1)[0]


class ResponseNetDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        vocab: Vocab | None = None,
        build_vocab: bool = False,
        filter_overlap_sec: float | None = 0.1,
        true_overlap_path: str | Path | None = None,
        target_sr: int = 16000,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.vocab = vocab
        self.target_sr = target_sr
        self.true_overlap_path = true_overlap_path

        self.samples = self._load_samples(filter_overlap_sec)
        if build_vocab:
            if self.vocab is not None:
                raise ValueError("build_vocab=True 时不得同时传入 vocab")
            self.vocab = self._build_vocab_from_train()
            logger.info("已从 %s split 构建词表，大小=%d", split, len(self.vocab))
        if self.vocab is None:
            raise ValueError("必须提供 vocab（或对 train split 设置 build_vocab=True）")

    # ------------------------------------------------------------------ #
    def _load_samples(self, filter_overlap_sec: float | None) -> list[dict]:
        path = self.data_root / f"{self.split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        if filter_overlap_sec is None:
            return raw

        overlap_map = self._load_true_overlap_map()
        n_before = len(raw)
        kept = [s for s in raw if overlap_map.get(sample_to_utterance_id(s), 0.0) <= filter_overlap_sec]
        n_filtered = n_before - len(kept)
        if n_filtered > 0:
            logger.info(
                "split=%s 真实语音重叠过滤：剔除 %d/%d 条样本（true_overlap > %.2fs，不含窗口 padding）",
                self.split, n_filtered, n_before, filter_overlap_sec,
            )
        return kept

    def _load_true_overlap_map(self) -> dict[str, float]:
        """uid -> true_overlap 秒（由 overlap_filter 从词区间计算，不含窗口 padding）。"""
        if self.true_overlap_path is None or not Path(self.true_overlap_path).exists():
            raise FileNotFoundError(
                f"缺少 true_overlap 映射文件: {self.true_overlap_path}\n"
                "请先运行: python neck_motion/analyze_overlap.py\n"
                "（train.py 会在训练前自动生成该文件）"
            )
        return load_true_overlap(self.true_overlap_path)

    def _build_vocab_from_train(self) -> Vocab:
        texts: list[str] = []
        for s in self.samples:
            cur = s.get("current_utterance") or {}
            if cur.get("text"):
                texts.append(cur["text"])
            prev = s.get("previous_listener_context") or s.get("previous_speaker_context")
            if prev and prev.get("text"):
                texts.append(prev["text"])
        # min_freq / max_size 从当前包默认取；train.py 负责持久化。
        return Vocab.build(texts)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def _load_target(self, rel_path: str, expected_n: int) -> np.ndarray:
        arr = np.load(str(self.data_root / rel_path)).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"RPY 文件格式异常: {rel_path} shape={arr.shape}")
        if arr.shape[0] != expected_n:
            logger.warning("RPY 长度 %d != num_frames %d: %s", arr.shape[0], expected_n, rel_path)
            arr = arr[:expected_n]
        return arr

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cur = s["current_utterance"]
        fps = float(s.get("fps", 30.0))
        N = int(s.get("num_frames", 0))
        if N <= 0:  # pragma: no cover
            raise ValueError(f"num_frames 无效: {s['sample_id']}")

        # 监督标签（relative RPY 与 delta，均为弧度）
        rpy = torch.from_numpy(self._load_target(s["target_rpy_path"], N))
        delta = torch.from_numpy(self._load_target(s["target_delta_rpy_path"], N))

        # 当前话语音频 -> mono 16k
        audio = load_audio_waveform_cached(str(self.data_root / cur["audio_path"]), self.target_sr)

        # 当前文本 -> 词 id + 帧区间（ids 与 starts/ends 严格 1:1）
        word_ids, starts, ends, n_skipped = encode_word_timestamps(
            self.vocab, cur.get("word_timestamps") or [], N, fps
        )
        if n_skipped:  # 仅在异常时提示，正常数据不会触发
            logger.debug("sample=%s 跳过 %d 个无效/越界词", s["sample_id"], n_skipped)

        # 上一片段文本（仅语义上下文；音频不使用）。首片段 -> [<START_OF_DIALOGUE>]
        prev = s.get("previous_listener_context") or s.get("previous_speaker_context")
        prev_text = (prev or {}).get("text") if prev else None
        prev_token_ids = self.vocab.encode_text(prev_text) if prev_text else [self.vocab.word2idx[START_OF_DIALOGUE]]

        return {
            "sample_id": s["sample_id"],
            "role": s["role"],
            "role_id": ROLE_TO_ID[s["role"]],
            "N": N,
            "fps": fps,
            "duration_sec": float(s.get("duration_sec", N / fps)),
            "initial_pose": np.asarray(s.get("initial_pose", [0.0, 0.0, 0.0]), dtype=np.float32),
            "audio": audio,                # [1, T] @16k mono
            "audio_len": int(audio.shape[1]),
            "word_ids": word_ids,          # list[int]
            "word_starts": starts,         # list[int] 帧区间 [s, e)
            "word_ends": ends,
            "prev_token_ids": prev_token_ids,  # list[int]
            "rpy": rpy,                    # [N, 3] relative
            "delta": delta,                # [N, 3]
        }


def build_inference_batch(data: dict, vocab: Vocab, data_root: str | Path,
                          target_sr: int = 16000, fps_default: float = 30.0) -> dict:
    """把与训练样本结构兼容的输入 JSON 构建为单样本 batch dict（CPU 张量）。

    infer.py 与 mvp.py 共用。帧数 N 由 num_frames / duration_sec / 词时间戳推算。
    """
    data_root = Path(data_root)
    role = data["role"]
    if role not in ROLE_TO_ID:
        raise ValueError(f"role 必须是 speaker/listener，得到: {role}")
    cur = data["current_utterance"]
    wts = cur.get("word_timestamps") or []
    fps = float(data.get("fps", fps_default))

    if data.get("num_frames"):
        N = int(data["num_frames"])
    elif data.get("duration_sec"):
        N = max(2, round(float(data["duration_sec"]) * fps))
    elif wts:
        last_end = max(float(w.get("end_time", 0.0)) for w in wts)
        N = max(2, round((last_end + 0.5) * fps))  # 与数据管线窗口后垫 0.5s 一致
    else:
        raise ValueError("无法确定帧数：请提供 num_frames 或 duration_sec，或带时间戳的词列表")

    prev = (data.get("previous_listener_context") or data.get("previous_speaker_context")
            or data.get("previous_context"))
    prev_text = (prev or {}).get("text") if prev else None
    prev_ids = vocab.encode_text(prev_text) if prev_text else [vocab.word2idx[START_OF_DIALOGUE]]

    audio = load_audio_waveform(data_root / cur["audio_path"], target_sr)
    word_ids, starts, ends, n_skipped = encode_word_timestamps(vocab, wts, N, fps)
    if n_skipped:
        logger.debug("inference: 跳过 %d 个无效/越界词", n_skipped)

    return {
        "audio": audio.unsqueeze(0),                      # [1,1,T]
        "Ns": torch.tensor([N], dtype=torch.long),
        "mask": torch.ones(1, N, dtype=torch.bool),
        "roles": torch.tensor([ROLE_TO_ID[role]], dtype=torch.long),
        "prev_token_ids": torch.tensor([prev_ids], dtype=torch.long),
        "prev_lens": torch.tensor([len(prev_ids)], dtype=torch.long),
        "word_ids": [word_ids],
        "word_starts": [starts],
        "word_ends": [ends],
        "role_name": role,
        "num_frames": N,
    }


def collate_neck_motion(batch: list[dict]) -> dict:
    """把样本列表打包为 batch dict（变长帧 -> padding + mask）。"""
    B = len(batch)
    n_max = max(b["N"] for b in batch)
    t_max = max(b["audio"].shape[1] for b in batch)
    l_max = max(len(b["prev_token_ids"]) for b in batch)

    audio = torch.zeros(B, 1, t_max)
    audio_lens = torch.zeros(B, dtype=torch.long)
    rpy = torch.zeros(B, n_max, 3)
    delta = torch.zeros(B, n_max, 3)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    prev_token_ids = torch.zeros(B, l_max, dtype=torch.long)
    prev_lens = torch.zeros(B, dtype=torch.long)
    Ns = torch.zeros(B, dtype=torch.long)
    roles = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["N"]
        audio[i, :, : b["audio"].shape[1]] = b["audio"]
        audio_lens[i] = b["audio_len"]
        rpy[i, :n] = b["rpy"]
        delta[i, :n] = b["delta"]
        mask[i, :n] = True
        Ns[i] = n
        roles[i] = b["role_id"]
        pl = len(b["prev_token_ids"])
        prev_token_ids[i, :pl] = torch.tensor(b["prev_token_ids"], dtype=torch.long)
        prev_lens[i] = pl

    return {
        "audio": audio,
        "audio_lens": audio_lens,
        "rpy": rpy,
        "delta": delta,
        "mask": mask,
        "Ns": Ns,
        "roles": roles,
        "prev_token_ids": prev_token_ids,
        "prev_lens": prev_lens,
        "word_ids": [b["word_ids"] for b in batch],
        "word_starts": [b["word_starts"] for b in batch],
        "word_ends": [b["word_ends"] for b in batch],
        # metadata（不进入网络）
        "sample_ids": [b["sample_id"] for b in batch],
        "role_names": [b["role"] for b in batch],
        "initial_poses": [b["initial_pose"] for b in batch],
        "duration_sec": [b["duration_sec"] for b in batch],
    }
