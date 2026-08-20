"""片段级条件 CVAE（v2 概率生成模型）。

    ConditionEncoder:  h = 音频 + 对齐文本 + 上下文 + role（逐帧特征 [B, N, D]）
    h_cond = mean-pool(h)                                [B, D]
    训练：q(z | h_cond, target) -> N(μ_q, σ_q)          z ~ q，重参数化
    部署：p(z | h_cond) -> N(μ_p, σ_p)                  z ~ p
    解码：concat([逐帧 h, z 广播]) -> z_proj -> SequenceDecoder -> 双头 -> pred

- z 对整段 fragment 共享（片段级，非逐帧噪声），保证风格/动作连贯。
- 训练目标 = 重构（pose/velocity/acceleration 分项，同回归版）+ β_kl·KL(q||p)。
- 必须启用 KL warm-up 与 free-bits，否则后验会忽略 z（posterior collapse）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.neck_motion.model import (
    ConditionEncoder,
    SequenceDecoder,
    route_and_zero_first,
)

LOGVAR_MIN, LOGVAR_MAX = -10.0, 2.0


class _TargetEncoder(nn.Module):
    """target RPY 轨迹 -> 全局向量 [B, H_t]（后验 q 的输入编码）。"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            target, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        B = target.size(0)
        idx = (lengths - 1).clamp(min=0)
        return out[torch.arange(B, device=out.device), idx]  # [B, H_t]


class FragmentCVAE(nn.Module):
    """片段级条件 CVAE：共享编码器 + 后验/先验 + 序列解码器。"""

    def __init__(
        self,
        vocab_size: int,
        z_dim: int = 16,
        hidden_dim: int = 256,
        target_encoder_hidden: int = 128,
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
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.is_cvae = True

        self.encoder = ConditionEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim, mel_kwargs=mel_kwargs,
            use_audio=use_audio, use_text=use_text,
            audio_encoder_out_dim=audio_encoder_out_dim,
            audio_encoder_kernel=audio_encoder_kernel,
            word_embed_dim=word_embed_dim,
            context_encoder=context_encoder,
            context_hidden_dim=context_hidden_dim,
            context_num_layers=context_num_layers,
            context_bidirectional=context_bidirectional,
            role_embed_dim=role_embed_dim,
        )
        self.silence_embedding = self.encoder.silence_embedding

        # 后验 q(z | h_cond, target)
        self.target_encoder = _TargetEncoder(input_dim=3, hidden_dim=target_encoder_hidden)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + target_encoder_hidden, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        # 先验 p(z | h_cond)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        # 解码：逐帧条件特征 + 片段级 z
        self.z_proj = nn.Linear(hidden_dim + z_dim, hidden_dim)
        self.decoder = SequenceDecoder(
            hidden_dim=hidden_dim, transformer_layers=transformer_layers,
            attention_heads=attention_heads, feedforward_dim=feedforward_dim,
            transformer_dropout=transformer_dropout, pos_encoding=pos_encoding,
            max_len=max_len, head_hidden_dim=head_hidden_dim, head_dropout=head_dropout,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor, logvar_q: torch.Tensor,
        mu_p: torch.Tensor, logvar_p: torch.Tensor,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """KL(q||p) [B, z_dim]（逐维），free-bits 后按样本求平均 -> 标量。"""
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl_per_dim = 0.5 * (
            logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
        )
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        return kl_per_dim.sum(dim=-1).mean()

    # ------------------------------------------------------------------ #
    def encode_condition(self, batch: dict) -> tuple[dict, torch.Tensor]:
        feats = self.encoder(batch)
        h_cond = feats["h"].mean(dim=1)  # [B, D] 片段级条件向量
        return feats, h_cond

    def posterior_params(self, h_cond: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor):
        t_enc = self.target_encoder(target, lengths)  # [B, H_t]
        return self._split_params(self.posterior(torch.cat([h_cond, t_enc], dim=-1)))

    def prior_params(self, h_cond: torch.Tensor):
        return self._split_params(self.prior(h_cond))

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict, return_aux: bool = False, use_posterior: bool = True):
        """训练/评估路径（默认 use_posterior=True，需要 batch['rpy']）。

        Returns:
            pred [B, Nmax, 3]（首帧为零）；return_aux=True 时返回 (pred, aux)。
        """
        feats, h_cond = self.encode_condition(batch)
        if use_posterior:
            mu_q, logvar_q = self.posterior_params(h_cond, batch["rpy"], batch["Ns"])
            mu_p, logvar_p = self.prior_params(h_cond)
            z = self.reparameterize(mu_q, logvar_q)
            kl = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
        else:
            mu_p, logvar_p = self.prior_params(h_cond)
            z = self.reparameterize(mu_p, logvar_p)
            kl = h_cond.new_zeros(())

        B, n_max, _ = feats["h"].shape
        z_b = z.unsqueeze(1).expand(B, n_max, -1)  # 片段级 z 广播到每帧
        h = self.z_proj(torch.cat([feats["h"], z_b], dim=-1))
        out_speaker, out_listener = self.decoder(h, batch["mask"])
        pred = route_and_zero_first(out_speaker, out_listener, batch["roles"])

        if return_aux:
            aux = {**feats, "out_speaker": out_speaker, "out_listener": out_listener,
                   "mask": batch["mask"], "roles": batch["roles"],
                   "kl": kl, "mu_q": mu_q, "logvar_q": logvar_q,
                   "mu_p": mu_p, "logvar_p": logvar_p, "z": z}
            return pred, aux
        return pred

    @torch.no_grad()
    def sample(self, batch: dict, n_samples: int = 10, temperature: float = 1.0) -> torch.Tensor:
        """部署路径：先验采样 K 次。返回 [B, K, Nmax, 3]（首帧为零）。"""
        feats, h_cond = self.encode_condition(batch)
        mu_p, logvar_p = self.prior_params(h_cond)
        B, n_max, _ = feats["h"].shape
        preds = []
        for _ in range(n_samples):
            z = self.reparameterize(mu_p, logvar_p, temperature)
            z_b = z.unsqueeze(1).expand(B, n_max, -1)
            h = self.z_proj(torch.cat([feats["h"], z_b], dim=-1))
            out_s, out_l = self.decoder(h, batch["mask"])
            preds.append(route_and_zero_first(out_s, out_l, batch["roles"]).unsqueeze(1))
        return torch.cat(preds, dim=1)  # [B, K, Nmax, 3]


def build_fragment_cvae(cfg: dict, vocab_size: int) -> FragmentCVAE:
    """从配置 dict 构建 CVAE 模型。"""
    m = cfg["model"]
    c = cfg.get("cvae", {})
    return FragmentCVAE(
        vocab_size=vocab_size,
        z_dim=int(c.get("z_dim", 16)),
        hidden_dim=m["hidden_dim"],
        target_encoder_hidden=int(c.get("target_encoder_hidden", 128)),
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


def build_model(cfg: dict, vocab_size: int) -> nn.Module:
    """按 config 的 model.type 构建回归 / CVAE / 多候选模型。"""
    t = cfg.get("model", {}).get("type", "regression")
    if t == "cvae":
        return build_fragment_cvae(cfg, vocab_size)
    if t == "candidates":
        from models.neck_motion.candidates import build_multi_candidate_model
        return build_multi_candidate_model(cfg, vocab_size)
    return _build_regression(cfg, vocab_size)


def _build_regression(cfg: dict, vocab_size: int):
    from models.neck_motion.model import build_neck_motion_model
    return build_neck_motion_model(cfg, vocab_size)
