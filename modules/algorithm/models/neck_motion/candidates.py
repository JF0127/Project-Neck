"""条件多候选轨迹模型（v2.1）。

训练与部署完全一致：直接先验采样（无后验，杜绝部署时后验不可用的作弊路径）。

    h = ConditionEncoder(batch)                    # 逐帧条件特征 [B, N, D]
    z₁...zK ~ p(z|h) = N(0, I)（固定标准正态先验，z 整段共享）
    ŷ₁...ŷK = 解码器(h, z₁...zK)                   # batch 维度拼接，一次 Transformer

训练损失（MultiCandidateLoss）：
    L = min_k L_recon(ŷ_k, y) + λ_div·L_div + λ_smooth·L_smooth + λ_amp·L_amp

- min_k：轨迹级重构损失，选择最接近标注的一条（梯度只流向被选候选）；
- L_div 有上界（候选两两距离达到 div_target 即停），防止候选相同，也防止仅靠
  夸张大幅动作刷多样性（配合幅度 hinge）；
- 每条候选都计算平滑度（二阶差分 hinge）与幅度 hinge。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.neck_motion.model import ConditionEncoder, SequenceDecoder, route_and_zero_first


class MultiCandidateModel(nn.Module):
    """条件多候选模型。sample() 与 forward() 相同（均为先验采样，可部署）。"""

    def __init__(
        self,
        vocab_size: int,
        k: int = 8,
        z_dim: int = 16,
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
        self.k = k
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.is_candidates = True

        self.encoder = ConditionEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim, mel_kwargs=mel_kwargs,
            use_audio=use_audio, use_text=use_text, role_only=role_only,
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
        # 解码：逐帧条件特征 + 片段级 z
        self.z_proj = nn.Linear(hidden_dim + z_dim, hidden_dim)
        self.decoder = SequenceDecoder(
            hidden_dim=hidden_dim, transformer_layers=transformer_layers,
            attention_heads=attention_heads, feedforward_dim=feedforward_dim,
            transformer_dropout=transformer_dropout, pos_encoding=pos_encoding,
            max_len=max_len, head_hidden_dim=head_hidden_dim, head_dropout=head_dropout,
        )

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict, n_samples: int | None = None, return_aux: bool = False,
                fixed_z: torch.Tensor | None = None):
        """先验采样 K 个候选 -> [B, K, N, 3]（首帧为零）。训练/部署同一路径。

        fixed_z: [K, z_dim] 或 [B, K, z_dim]——固定 latent 模板（style_fixed 策略用，
        保证同一风格跨调用/跨段可复现）。
        """
        K = int(n_samples or self.k)
        feats = self.encoder(batch)
        B, n_max, _ = feats["h"].shape
        device = feats["h"].device

        if fixed_z is not None:
            z = fixed_z.to(device=device, dtype=torch.float32)
            if z.dim() == 2:
                z = z.unsqueeze(0).expand(B, -1, -1)  # [B, K, z_dim]
            z = z.reshape(B, K, self.z_dim)
        else:
            z = torch.randn(B, K, self.z_dim, device=device)  # 固定标准正态先验

        # 候选拼入 batch 维度，一次解码
        h = feats["h"].unsqueeze(1).expand(B, K, n_max, -1).reshape(B * K, n_max, -1)
        z_b = z.reshape(B * K, 1, self.z_dim).expand(B * K, n_max, -1)
        h_in = self.z_proj(torch.cat([h, z_b], dim=-1))
        mask_k = batch["mask"].unsqueeze(1).expand(B, K, -1).reshape(B * K, n_max)
        roles_k = batch["roles"].unsqueeze(1).expand(B, K).reshape(B * K)
        out_s, out_l = self.decoder(h_in, mask_k)
        pred = route_and_zero_first(out_s, out_l, roles_k)  # [B*K, N, 3]
        preds = pred.reshape(B, K, n_max, 3)

        if return_aux:
            return preds, {**feats, "z": z}
        return preds

    @torch.no_grad()
    def sample(self, batch: dict, n_samples: int = 10, temperature: float = 1.0,
               fixed_z: torch.Tensor | None = None) -> torch.Tensor:
        """部署路径：与 forward 相同（先验采样）。temperature 保留接口（candidates 未用）。"""
        return self.forward(batch, n_samples=n_samples, fixed_z=fixed_z)


def build_multi_candidate_model(cfg: dict, vocab_size: int) -> MultiCandidateModel:
    m = cfg["model"]
    c = cfg.get("candidates", {})
    return MultiCandidateModel(
        vocab_size=vocab_size,
        k=int(c.get("k", 8)),
        z_dim=int(c.get("z_dim", 16)),
        hidden_dim=m["hidden_dim"],
        mel_kwargs=cfg["audio"],
        use_audio=bool(m.get("use_audio", True)),
        use_text=bool(m.get("use_text", True)),
        role_only=bool(c.get("role_only", False)),
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
