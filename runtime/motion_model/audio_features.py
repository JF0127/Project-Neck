"""音频加载与 log-Mel 特征提取。

- 所有音频统一转 mono、重采样到 16 kHz（内存中完成，不修改原始 WAV）。
- 读取使用 soundfile（轻量、无 ffmpeg/torchcodec 依赖）；缺失时给出明确报错。
- 重采样使用 torchaudio.functional.resample（纯 torch 实现）。
- 80 维 log-Mel：手动三角滤波器组 + torch.stft，无大型依赖。
- 音频特征最终按样本插值到目标的 N 个 30 Hz RPY 帧（见 interpolate_mel_to_frames）。
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def load_audio_waveform(path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """读取 WAV -> mono float32 -> 重采样到 target_sr。返回 [1, T] 张量。

    Raises:
        ImportError: 缺少 soundfile / torchaudio 时给出可操作的安装提示。
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "缺少 soundfile 依赖，无法读取 WAV 音频。"
            "请安装：pip install soundfile"
        ) from exc

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    if data.shape[1] == 0:  # pragma: no cover
        raise ValueError(f"音频文件为空或通道数为 0: {path}")
    wav = torch.from_numpy(data)
    if wav.shape[1] > 1:  # downmix 到 mono
        wav = wav.mean(dim=1, keepdim=True)
    wav = wav[:, 0]  # [T]

    if sr != target_sr:
        try:
            import torchaudio.functional as TAF
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "缺少 torchaudio 依赖，无法执行 16 kHz 重采样。"
                "请安装：pip install torchaudio"
            ) from exc
        wav = TAF.resample(wav, sr, target_sr)

    if not torch.isfinite(wav).all():  # pragma: no cover
        raise ValueError(f"音频包含 NaN/Inf: {path}")
    return wav.unsqueeze(0)  # [1, T]


@functools.lru_cache(maxsize=256)
def _load_audio_cached(path: str, target_sr: int) -> torch.Tensor:
    """带 LRU 缓存的音频加载（每个 worker 进程独立缓存，跨 epoch 复用）。"""
    return load_audio_waveform(path, target_sr)


def load_audio_waveform_cached(path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """带缓存的音频加载，供 Dataset 使用。"""
    return _load_audio_cached(str(path), target_sr)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int, f_min: float, f_max: float) -> torch.Tensor:
    """HTK 风格三角 mel 滤波器组 [n_mels, n_fft // 2 + 1]。"""
    def hz_to_mel(f: float) -> float:
        return 2595.0 * float(np.log10(1.0 + f / 700.0))

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if right > left:
            fb[i, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
            fb[i, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return torch.from_numpy(fb)


class LogMelExtractor(nn.Module):
    """80 维 log-Mel 特征提取器（纯 torch）。

    输入 [B, 1, T] @16kHz -> 输出 [B, n_mels, T_mel]（约 100 帧/秒）。
    """

    def __init__(
        self,
        n_mels: int = 80,
        sr: int = TARGET_SR,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        log_eps: float = 1e-5,
        normalize: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_eps = log_eps
        self.normalize = normalize
        self.register_buffer("filterbank", _mel_filterbank(n_mels, n_fft, sr, f_min, f_max))
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]
        spec = torch.stft(
            wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )  # [B, n_fft//2+1, T_mel]
        power = spec.abs().pow(2)
        mel = torch.matmul(self.filterbank, power)  # [B, n_mels, T_mel]
        log_mel = torch.log(torch.clamp(mel, min=self.log_eps))
        if self.normalize:
            mean = log_mel.mean(dim=2, keepdim=True)
            std = log_mel.std(dim=2, keepdim=True) + 1e-6
            log_mel = (log_mel - mean) / std
        return log_mel


def interpolate_mel_to_frames(mel: torch.Tensor, n_frames: list[int]) -> torch.Tensor:
    """把 log-Mel 按样本线性插值到每个样本的 N 个 30 Hz RPY 帧。

    Args:
        mel: [B, n_mels, T_mel]
        n_frames: 每个样本的目标帧数 [B]

    Returns:
        [B, n_mels, Nmax]，Nmax = max(n_frames)；超出各自 N 的部分补零（被 mask 忽略）。
    """
    B = mel.shape[0]
    if B == 0:  # pragma: no cover
        return mel
    n_frames = [int(n) for n in n_frames]
    n_max = max(n_frames)
    out = mel.new_zeros(B, mel.shape[1], n_max)
    for i, n in enumerate(n_frames):
        if n <= 0:
            continue
        seg = F.interpolate(mel[i : i + 1], size=n, mode="linear", align_corners=False)
        out[i, :, :n] = seg
    return out
