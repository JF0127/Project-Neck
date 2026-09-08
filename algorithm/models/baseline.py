"""Small deterministic audio-text Transformer baseline."""

from __future__ import annotations

import math

import torch
from torch import nn


class ContinuousTimeEncoding(nn.Module):
    """Fixed sinusoidal encoding evaluated at physical timestamps in seconds."""

    def __init__(
        self,
        hidden_dim: int,
        min_frequency_hz: float = 0.1,
        max_frequency_hz: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for sinusoidal time encoding")
        if not 0 < min_frequency_hz <= max_frequency_hz:
            raise ValueError("time frequencies must satisfy 0 < min <= max")
        frequencies = torch.logspace(
            math.log10(min_frequency_hz),
            math.log10(max_frequency_hz),
            hidden_dim // 2,
            dtype=torch.float32,
        )
        self.register_buffer("frequencies_hz", frequencies)

    def forward(
        self, neck_timestamps: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        if neck_timestamps.ndim != 2 or sequence_mask.shape != neck_timestamps.shape:
            raise ValueError("neck_timestamps and sequence_mask must have matching shape [B, T]")
        if sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must have bool dtype")
        if not torch.isfinite(neck_timestamps[sequence_mask]).all():
            raise ValueError("real neck timestamps must be finite")

        # Padding timestamps are NaN in the collated Dataset batch. Replacing only
        # those values avoids propagating NaNs before they are masked to zero.
        safe_times = torch.where(sequence_mask, neck_timestamps, torch.zeros_like(neck_timestamps))
        angles = (
            2.0
            * math.pi
            * safe_times.to(dtype=self.frequencies_hz.dtype).unsqueeze(-1)
            * self.frequencies_hz
        )
        encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoding * sequence_mask.unsqueeze(-1)


class BaselineModel(nn.Module):
    """Fuse aligned conditions and predict fragment-relative RPY offsets."""

    def __init__(
        self,
        audio_dim: int = 64,
        text_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.time_encoding = ContinuousTimeEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.rpy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            # A final bias would cancel exactly in raw(t) - raw(t0).
            nn.Linear(hidden_dim // 2, 3, bias=False),
        )

    def forward(
        self,
        aligned_audio: torch.Tensor,
        aligned_text: torch.Tensor,
        neck_timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if aligned_audio.ndim != 3 or aligned_audio.shape[-1] != self.audio_dim:
            raise ValueError(
                f"aligned_audio must have shape [B, T, {self.audio_dim}], "
                f"got {tuple(aligned_audio.shape)}"
            )
        if aligned_text.ndim != 3 or aligned_text.shape[-1] != self.text_dim:
            raise ValueError(
                f"aligned_text must have shape [B, T, {self.text_dim}], "
                f"got {tuple(aligned_text.shape)}"
            )
        if aligned_audio.shape[:2] != aligned_text.shape[:2]:
            raise ValueError("aligned audio and text must share [B, T]")
        if neck_timestamps.shape != aligned_audio.shape[:2]:
            raise ValueError("neck_timestamps must have shape [B, T]")
        if sequence_mask.shape != aligned_audio.shape[:2] or sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must be bool with shape [B, T]")
        if not torch.all(sequence_mask[:, 0]):
            raise ValueError("the first timestep must be real for every fragment")

        fused = self.fusion(torch.cat((aligned_audio, aligned_text), dim=-1))
        hidden = fused + self.time_encoding(neck_timestamps, sequence_mask)
        # Dataset sequence_mask uses True=real; PyTorch uses True=padding.
        hidden = self.transformer(hidden, src_key_padding_mask=~sequence_mask)
        raw_prediction = self.rpy_head(hidden)
        prediction = raw_prediction - raw_prediction[:, :1, :]
        # Padding predictions have no physical meaning and are explicitly zero.
        return prediction * sequence_mask.unsqueeze(-1)
