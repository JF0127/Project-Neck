"""Learnable audio/text features aligned to canonical neck timestamps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NO_WORD_TOKEN = "<no_word>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, NO_WORD_TOKEN)


def tokenize_word(text: str) -> list[str]:
    """Tokenize one timestamped ASR word entry into Unicode characters."""
    return list(text)


class CharacterVocabulary:
    """Small deterministic character vocabulary built from train words only."""

    def __init__(self, tokens: Sequence[str]) -> None:
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"vocabulary must start with {SPECIAL_TOKENS}")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary tokens must be unique")
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def no_word_id(self) -> int:
        return self.token_to_id[NO_WORD_TOKEN]

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokenize_word(text)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterVocabulary":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(value["tokens"])


def build_vocab(
    train_dataset: Iterable[Mapping[str, Any]], min_frequency: int = 1
) -> CharacterVocabulary:
    """Build a deterministic vocabulary; rejects a dataset marked as val/test."""
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    split = getattr(train_dataset, "split", "train")
    if split != "train":
        raise ValueError(f"vocabulary may only be built from train split, got {split!r}")

    counts: Counter[str] = Counter()
    iter_word_entries = getattr(train_dataset, "iter_word_entries", None)
    word_batches = (
        iter_word_entries()
        if callable(iter_word_entries)
        else (sample["words"] for sample in train_dataset)
    )
    for words in word_batches:
        for word in words:
            counts.update(tokenize_word(word["text"]))
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(sorted(token for token, count in counts.items() if count >= min_frequency and token not in SPECIAL_TOKENS))
    return CharacterVocabulary(tokens)


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(0.0)),
        _hz_to_mel(torch.tensor(sample_rate / 2)),
        n_mels + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    lower = hz_edges[:-2].unsqueeze(1)
    center = hz_edges[1:-1].unsqueeze(1)
    upper = hz_edges[2:].unsqueeze(1)
    rising = (frequencies.unsqueeze(0) - lower) / (center - lower)
    falling = (upper - frequencies.unsqueeze(0)) / (upper - center)
    return torch.clamp(torch.minimum(rising, falling), min=0.0)


class AudioFrontend(nn.Module):
    """80-bin log-Mel frontend with unambiguous center=False frame times."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        log_epsilon: float = 1e-10,
    ) -> None:
        super().__init__()
        if n_fft != win_length:
            raise ValueError("the first frontend requires n_fft == win_length for explicit frame timing")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_epsilon = log_epsilon
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer("mel_filters", _mel_filterbank(sample_rate, n_fft, n_mels))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[M, n_mels]`` and physical frame-center seconds ``[M]``."""
        if waveform.ndim != 1:
            raise ValueError(f"waveform must have shape [A], got {tuple(waveform.shape)}")
        if waveform.numel() < self.win_length:
            raise ValueError(
                f"waveform has {waveform.numel()} samples, fewer than win_length={self.win_length}"
            )
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel_power = self.mel_filters @ power
        log_mel = torch.log(mel_power.clamp_min(self.log_epsilon)).transpose(0, 1)
        frame_count = log_mel.shape[0]
        # Frame m consumes the half-open analysis interval
        # [m*hop, m*hop+win_length) in sample units. Its timestamp is that
        # interval's center; with the defaults this is 0.0125 + 0.01*m seconds.
        timestamps = (
            torch.arange(frame_count, device=waveform.device, dtype=torch.float64)
            * self.hop_length
            + self.win_length / 2
        ) / self.sample_rate
        return log_mel, timestamps


class AudioEncoder(nn.Module):
    """Small learnable time-preserving log-Mel encoder."""

    def __init__(self, n_mels: int = 80, feature_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(n_mels, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim != 2:
            raise ValueError(f"log_mel must have shape [M, D], got {tuple(log_mel.shape)}")
        return self.network(log_mel.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)


def interpolate_at_timestamps(
    native_features: torch.Tensor,
    native_timestamps: torch.Tensor,
    query_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Linear time interpolation with edge-hold outside native center coverage."""
    if native_features.ndim != 2:
        raise ValueError("native_features must have shape [M, D]")
    if native_timestamps.ndim != 1 or len(native_timestamps) != len(native_features):
        raise ValueError("native_timestamps must have shape [M] matching native_features")
    if query_timestamps.ndim != 1:
        raise ValueError("query_timestamps must have shape [Q]")
    if not len(native_timestamps):
        raise ValueError("cannot interpolate an empty native feature sequence")
    if len(native_timestamps) > 1 and not torch.all(native_timestamps[1:] > native_timestamps[:-1]):
        raise ValueError("native_timestamps must be strictly increasing")
    if not torch.isfinite(query_timestamps).all():
        raise ValueError("query_timestamps must be finite")
    if len(native_timestamps) == 1:
        return native_features[0].expand(len(query_timestamps), -1)

    queries = query_timestamps.to(device=native_timestamps.device, dtype=native_timestamps.dtype)
    right = torch.searchsorted(native_timestamps, queries, right=False)
    right = right.clamp(1, len(native_timestamps) - 1)
    left = right - 1
    left_time = native_timestamps[left]
    right_time = native_timestamps[right]
    weight = ((queries - left_time) / (right_time - left_time)).clamp(0.0, 1.0)
    weight = weight.to(dtype=native_features.dtype).unsqueeze(1)
    return native_features[left] + weight * (native_features[right] - native_features[left])


class FeatureEncoder(nn.Module):
    """Produce audio/text conditions on exactly the padded neck timestamp grid."""

    def __init__(
        self,
        vocab: CharacterVocabulary,
        audio_feature_dim: int = 64,
        text_feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.audio_frontend = AudioFrontend()
        self.audio_encoder = AudioEncoder(self.audio_frontend.n_mels, audio_feature_dim)
        self.text_embedding = nn.Embedding(
            len(vocab), text_feature_dim, padding_idx=vocab.pad_id
        )
        self.audio_feature_dim = audio_feature_dim
        self.text_feature_dim = text_feature_dim

    def encode_audio_batch(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Process each unpadded waveform independently for batch invariance."""
        if audio.ndim != 2 or audio_lengths.shape != (audio.shape[0],):
            raise ValueError("audio/audio_lengths must have shapes [B, Amax] and [B]")
        features: list[torch.Tensor] = []
        timestamps: list[torch.Tensor] = []
        for batch_index, length_value in enumerate(audio_lengths):
            length = int(length_value.item())
            if length <= 0 or length > audio.shape[1]:
                raise ValueError(f"invalid audio_lengths[{batch_index}]={length}")
            log_mel, frame_timestamps = self.audio_frontend(audio[batch_index, :length])
            features.append(self.audio_encoder(log_mel))
            timestamps.append(frame_timestamps)
        return features, timestamps

    def encode_word(self, text: str, device: torch.device) -> torch.Tensor:
        token_ids = self.vocab.encode(text)
        if not token_ids:
            token_ids = [self.vocab.unk_id]
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        return self.text_embedding(ids).mean(dim=0)

    def align_text(
        self,
        words: Sequence[Sequence[Mapping[str, Any]]],
        neck_timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use [local_start, local_end); overlapping words are mean pooled."""
        batch_size, max_steps = neck_timestamps.shape
        outputs: list[torch.Tensor] = []
        no_word = self.text_embedding.weight[self.vocab.no_word_id]
        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(sequence_mask[batch_index], as_tuple=False).squeeze(1)
            queries = neck_timestamps[batch_index, valid_positions]
            if not torch.isfinite(queries).all():
                raise ValueError(f"finite neck timestamps required at sequence_mask=True for batch {batch_index}")
            sample_words = words[batch_index]
            if sample_words:
                word_features = torch.stack(
                    [self.encode_word(str(word["text"]), neck_timestamps.device) for word in sample_words]
                )
                starts = torch.tensor(
                    [float(word["local_start"]) for word in sample_words],
                    dtype=queries.dtype,
                    device=queries.device,
                )
                ends = torch.tensor(
                    [float(word["local_end"]) for word in sample_words],
                    dtype=queries.dtype,
                    device=queries.device,
                )
                coverage = (queries.unsqueeze(1) >= starts) & (queries.unsqueeze(1) < ends)
                counts = coverage.sum(dim=1, keepdim=True)
                covered_features = coverage.to(word_features.dtype) @ word_features
                valid_features = torch.where(
                    counts > 0,
                    covered_features / counts.clamp_min(1).to(word_features.dtype),
                    no_word.unsqueeze(0),
                )
            else:
                valid_features = no_word.expand(len(queries), -1)
            sample_output = torch.zeros(
                max_steps,
                self.text_feature_dim,
                dtype=valid_features.dtype,
                device=valid_features.device,
            ).index_copy(0, valid_positions, valid_features)
            outputs.append(sample_output)
        return torch.stack(outputs)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        words: Sequence[Sequence[Mapping[str, Any]]],
        neck_timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if neck_timestamps.ndim != 2 or sequence_mask.shape != neck_timestamps.shape:
            raise ValueError("neck_timestamps and sequence_mask must have matching shape [B, T]")
        if len(words) != audio.shape[0] or neck_timestamps.shape[0] != audio.shape[0]:
            raise ValueError("audio, words, and neck timestamps must have the same batch size")
        if sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must have bool dtype")

        native_features, native_timestamps = self.encode_audio_batch(audio, audio_lengths)
        max_steps = neck_timestamps.shape[1]
        aligned_audio_samples: list[torch.Tensor] = []
        for batch_index, (features, timestamps) in enumerate(zip(native_features, native_timestamps)):
            valid_positions = torch.nonzero(sequence_mask[batch_index], as_tuple=False).squeeze(1)
            queries = neck_timestamps[batch_index, valid_positions]
            aligned = interpolate_at_timestamps(features, timestamps, queries)
            sample_output = torch.zeros(
                max_steps,
                self.audio_feature_dim,
                dtype=aligned.dtype,
                device=aligned.device,
            ).index_copy(0, valid_positions, aligned)
            aligned_audio_samples.append(sample_output)

        return {
            "aligned_audio": torch.stack(aligned_audio_samples),
            "aligned_text": self.align_text(words, neck_timestamps, sequence_mask),
            "audio_native_features": native_features,
            "audio_native_timestamps": native_timestamps,
        }
