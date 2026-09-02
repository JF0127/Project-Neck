"""共享条件编码器 + 序列解码器（确定性回归版与 CVAE 版共用）。

结构（详见 README.md）：

    当前音频 -> log-Mel(80) -> 两层 Conv1d+GELU 音频编码器 ──────────┐
    当前文本 + 词时间戳 -> 词向量 -> 30 Hz 帧对齐（平均池化/SILENCE）├─ 逐帧
    上一片段文本 -> Embedding+GRU -> context 向量（广播到每帧）      ├─ concat
    role embedding（广播） + 归一化时间特征 + coverage 特征          │
                                                                    ├─> Linear 投影
                                                                    └─> 逐帧条件特征 h [B, N, D]

    SequenceDecoder: 位置编码 + Transformer + SpeakerHead/ListenerHead
    NeckMotionModel = ConditionEncoder + SequenceDecoder（第一版确定性回归）
    FragmentCVAE    = ConditionEncoder + 后验/先验/解码（v2，见 cvae.py）

约束：
- initial_pose（数据人物的绝对坐姿）不进入网络，仅保留在元数据中。
- 模型直接预测完整 relative RPY（不是逐帧累加 delta），首帧强制为 [0,0,0]。
- 两个 head 独立；按样本 role 路由。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from neck_motion.audio_features import LogMelExtractor, interpolate_mel_to_frames
from neck_motion.text_features import PAD, align_words_to_frames

PAD_IDX = 0  # 与 text_features.SPECIAL_TOKENS 顺序一致（<PAD> 恒为 0）


# --------------------------------------------------------------------------- #
# 子模块
# --------------------------------------------------------------------------- #
class AudioConvEncoder(nn.Module):
    """两层 Conv1d + GELU 音频时序编码器。输入 [B, n_mels, T] -> 输出 [B, T, out_dim]。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_dim: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden_channels, out_dim, kernel_size, padding=pad)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x.transpose(1, 2)  # [B, T, out_dim]


class PositionalEncoding(nn.Module):
    """可学习或正弦位置编码（batch-first 输入 [B, T, D]）。"""

    def __init__(self, d_model: int, max_len: int, kind: str = "sinusoidal"):
        super().__init__()
        self.kind = kind
        if kind == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)  # [max_len, D]
        elif kind == "learned":
            self.pe = nn.Embedding(max_len, d_model)
        else:  # pragma: no cover
            raise ValueError(f"未知 pos_encoding: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if self.kind == "sinusoidal":
            return x + self.pe[:T].unsqueeze(0)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return x + self.pe(pos)


class ContextTextEncoder(nn.Module):
    """上一片段文本编码器：Embedding + 双向 GRU -> 全局 context 向量 [B, ctx_dim]。

    首片段（previous context 为 null）输入序列为 [<START_OF_DIALOGUE>]。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = token_ids.size(0)
        emb = self.embedding(token_ids)  # [B, L, D]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # [B, L, H]
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(B, device=out.device), idx]  # 正向最后一帧 [B, H]
        if self.gru.bidirectional:
            first = out[torch.arange(B, device=out.device), 0]  # 反向第一帧 [B, H]
            h = self.gru.hidden_size
            return torch.cat([last[:, :h], first[:, h:]], dim=-1)  # [B, 2h]
        return last


class _MeanPoolContextEncoder(nn.Module):
    """Embedding + 平均池化 的全局 context 编码器（与 GRU 版本接口一致）。"""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.out_dim = embed_dim

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)  # [B, L, D]
        mask = (torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        return (emb * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()  # [B, D]


class SpeakerHead(nn.Module):
    """Speaker 专用输出头：hidden_dim -> 128 -> 3（直接预测 relative RPY）。"""

    def __init__(self, hidden_dim: int, head_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ListenerHead(nn.Module):
    """Listener 专用输出头（与 SpeakerHead 结构相同但参数独立）。"""

    def __init__(self, hidden_dim: int, head_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 条件编码器（音频 + 文本 + 词时间戳 + 上下文 + role -> 逐帧特征 h）
# --------------------------------------------------------------------------- #
class ConditionEncoder(nn.Module):
    """共享条件编码器。batch 字段见 NeckMotionModel 注释。

    role_only=True 时仅保留 role embedding + 时间位置特征（role-only 多候选基线）。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        mel_kwargs: dict | None = None,
        use_audio: bool = True,
        use_text: bool = True,
        role_only: bool = False,
        audio_encoder_out_dim: int = 128,
        audio_encoder_kernel: int = 5,
        word_embed_dim: int = 128,
        context_encoder: str = "gru",
        context_hidden_dim: int = 128,
        context_num_layers: int = 1,
        context_bidirectional: bool = True,
        role_embed_dim: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_audio = use_audio and not role_only
        self.use_text = use_text and not role_only
        self.role_only = role_only

        # --- 音频分支 ---
        if use_audio:
            self.mel = LogMelExtractor(**(mel_kwargs or {}))
            self.audio_encoder = AudioConvEncoder(
                in_channels=self.mel.filterbank.shape[0],
                hidden_channels=max(64, audio_encoder_out_dim),
                out_dim=audio_encoder_out_dim,
                kernel_size=audio_encoder_kernel,
            )
            self.audio_out_dim = audio_encoder_out_dim
        else:
            self.mel = None
            self.audio_encoder = None
            self.audio_out_dim = 0

        # --- 文本分支（词表与上下文共用同一 embedding 表） ---
        if use_text:
            self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=PAD_IDX)
            self.silence_embedding = nn.Parameter(torch.randn(word_embed_dim) * 0.02)
            self.word_embed_dim = word_embed_dim
        else:
            self.word_embedding = None
            self.silence_embedding = None
            self.word_embed_dim = 0

        if use_text:
            if context_encoder == "gru":
                self.context_encoder = ContextTextEncoder(
                    vocab_size, word_embed_dim, context_hidden_dim, context_num_layers, context_bidirectional
                )
                ctx_dim = self.context_encoder.out_dim
            elif context_encoder == "mean_pool":
                self.context_encoder = _MeanPoolContextEncoder(vocab_size, word_embed_dim)
                ctx_dim = word_embed_dim
            else:  # pragma: no cover
                raise ValueError(f"未知 context encoder: {context_encoder}")
        else:
            self.context_encoder = None
            ctx_dim = 0
        self.ctx_dim = ctx_dim

        # --- role ---
        self.role_embedding = nn.Embedding(2, role_embed_dim)
        self.role_embed_dim = role_embed_dim

        # --- 融合投影 ---
        fusion_dim = (
            self.audio_out_dim + self.word_embed_dim + 1 + 1 + self.ctx_dim + role_embed_dim
        )
        if self.role_only:  # 仅 role + 时间位置
            fusion_dim = 1 + role_embed_dim
        self.fusion_proj = nn.Linear(fusion_dim, hidden_dim)

    def forward(self, batch: dict) -> dict:
        device = self.word_embedding.weight.device if self.word_embedding is not None else batch["audio"].device
        audio = batch["audio"]  # [B, 1, T]
        Ns = [int(n) for n in batch["Ns"]]
        B = audio.size(0)
        n_max = max(Ns) if Ns else 0

        # 音频：log-Mel -> 插值到各自 N 帧 -> Conv1d
        if self.use_audio:
            mel = self.mel(audio)  # [B, n_mels, T_mel]
            mel = interpolate_mel_to_frames(mel, Ns)  # [B, n_mels, Nmax]
            audio_feat = self.audio_encoder(mel)  # [B, Nmax, A]
        else:
            audio_feat = audio.new_zeros(B, n_max, 0)  # 消融/role-only：无音频特征

        # 文本：词向量按时间戳对齐到 30 Hz 帧 + coverage 二值特征
        if self.use_text:
            word_feat, coverage = align_words_to_frames(
                self.word_embedding,
                self.silence_embedding,
                batch["word_ids"],
                batch["word_starts"],
                batch["word_ends"],
                Ns,
            )  # [B, Nmax, D], [B, Nmax, 1]
        else:
            word_feat = audio.new_zeros(B, n_max, 0)  # 消融/role-only：无文本特征
            coverage = audio.new_zeros(B, n_max, 1)
        if self.role_only:
            coverage = audio.new_zeros(B, n_max, 0)  # role-only：不保留 coverage 维度

        # 归一化时间位置特征（-1..1）
        time_feat = torch.zeros(B, n_max, 1, device=device)
        for i, n in enumerate(Ns):
            if n > 0:
                time_feat[i, :n, 0] = torch.linspace(-1.0, 1.0, n, device=device)

        # 上一片段文本 -> 全局 context 向量 -> 广播到每帧
        if self.use_text:
            ctx = self.context_encoder(batch["prev_token_ids"], batch["prev_lens"])  # [B, C]
            ctx_feat = ctx.unsqueeze(1).expand(B, n_max, -1)
        else:
            ctx_feat = audio.new_zeros(B, n_max, 0)

        # role embedding -> 广播到每帧
        role_feat = self.role_embedding(batch["roles"]).unsqueeze(1).expand(B, n_max, -1)

        h = torch.cat([audio_feat, word_feat, coverage, time_feat, ctx_feat, role_feat], dim=-1)
        h = self.fusion_proj(h)  # [B, Nmax, hidden]
        return {
            "h": h,
            "audio_feat": audio_feat,
            "word_feat": word_feat,
            "coverage": coverage,
            "time_feat": time_feat,
            "ctx_feat": ctx_feat,
            "role_feat": role_feat,
            "Ns": Ns,
            "n_max": n_max,
        }


# --------------------------------------------------------------------------- #
# 序列解码器（位置编码 + Transformer + 双头）
# --------------------------------------------------------------------------- #
class SequenceDecoder(nn.Module):
    """Transformer 解码 + Speaker/Listener 双头。输入逐帧特征 h [B, N, D]。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        transformer_dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
        max_len: int = 2048,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(hidden_dim, max_len, kind=pos_encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.speaker_head = SpeakerHead(hidden_dim, head_hidden_dim, head_dropout)
        self.listener_head = ListenerHead(hidden_dim, head_hidden_dim, head_dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B, N, D] -> (out_speaker, out_listener) 均为 [B, N, 3]。"""
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=~mask)  # True = 忽略该帧
        return self.speaker_head(h), self.listener_head(h)


def route_and_zero_first(out_speaker: torch.Tensor, out_listener: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    """按样本 role 路由，并强制首帧为零（匹配数据定义 relative_rpy[0] ≈ 0）。"""
    is_speaker = (roles == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1]
    pred = torch.where(is_speaker, out_speaker, out_listener)
    pred = pred.clone()
    pred[:, 0, :] = 0.0
    return pred


# --------------------------------------------------------------------------- #
# 第一版确定性回归模型
# --------------------------------------------------------------------------- #
class NeckMotionModel(nn.Module):
    """共享编码器 + 双 head 的话语级颈部运动生成模型（确定性回归）。

    batch 字段（由 dataset.collate_neck_motion 产生）：
        audio [B,1,T] @16k、Ns [B]、mask [B,Nmax] bool、
        word_ids/word_starts/word_ends（list[list[int]]）、
        prev_token_ids [B,L]、prev_lens [B]、roles [B] long（0=speaker, 1=listener）。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        mel_kwargs: dict | None = None,
        use_audio: bool = True,
        use_text: bool = True,
        audio_encoder_out_dim: int = 128,
        audio_encoder_kernel: int = 5,
        word_embed_dim: int = 128,
        context_encoder: str = "gru",
        context_hidden_dim: int = 128,
        context_num_layers: int = 1,
        context_bidirectional: bool = True,
        role_embed_dim: int = 16,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        transformer_dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
        max_len: int = 2048,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = ConditionEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            mel_kwargs=mel_kwargs,
            use_audio=use_audio,
            use_text=use_text,
            audio_encoder_out_dim=audio_encoder_out_dim,
            audio_encoder_kernel=audio_encoder_kernel,
            word_embed_dim=word_embed_dim,
            context_encoder=context_encoder,
            context_hidden_dim=context_hidden_dim,
            context_num_layers=context_num_layers,
            context_bidirectional=context_bidirectional,
            role_embed_dim=role_embed_dim,
        )
        self.decoder = SequenceDecoder(
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            transformer_dropout=transformer_dropout,
            pos_encoding=pos_encoding,
            max_len=max_len,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        )
        # 便捷属性（dry-run 诊断等直接访问）
        self.silence_embedding = self.encoder.silence_embedding

    def forward(self, batch: dict, return_aux: bool = False):
        feats = self.encoder(batch)
        out_speaker, out_listener = self.decoder(feats["h"], batch["mask"])
        pred = route_and_zero_first(out_speaker, out_listener, batch["roles"])
        if return_aux:
            return pred, {**feats, "out_speaker": out_speaker, "out_listener": out_listener,
                          "mask": batch["mask"], "roles": batch["roles"]}
        return pred


def build_neck_motion_model(cfg: dict, vocab_size: int) -> NeckMotionModel:
    """从配置 dict 构建确定性回归模型。"""
    m = cfg["model"]
    return NeckMotionModel(
        vocab_size=vocab_size,
        hidden_dim=m["hidden_dim"],
        mel_kwargs=cfg["audio"],
        use_audio=bool(m.get("use_audio", True)),
        use_text=bool(m.get("use_text", True)),
        audio_encoder_out_dim=m["audio_encoder"]["out_dim"],
        audio_encoder_kernel=m["audio_encoder"]["kernel_size"],
        word_embed_dim=cfg["text"]["word_embed_dim"],
        context_encoder=cfg["text"]["context"]["encoder"],
        context_hidden_dim=cfg["text"]["context"]["hidden_dim"],
        context_num_layers=cfg["text"]["context"]["num_layers"],
        context_bidirectional=cfg["text"]["context"]["bidirectional"],
        role_embed_dim=m["role_embed_dim"],
        transformer_layers=m["transformer"]["layers"],
        attention_heads=m["transformer"]["heads"],
        feedforward_dim=m["transformer"]["ffn_dim"],
        transformer_dropout=m["transformer"]["dropout"],
        pos_encoding=m["transformer"]["pos_encoding"],
        max_len=m["transformer"]["max_len"],
        head_hidden_dim=m["head"]["hidden_dim"],
        head_dropout=m["head"]["dropout"],
    )
