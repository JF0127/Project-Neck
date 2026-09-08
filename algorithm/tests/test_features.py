from __future__ import annotations

import unittest

import torch

from algorithm.features import (
    CharacterVocabulary,
    FeatureEncoder,
    NO_WORD_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
    interpolate_at_timestamps,
)


class _Samples(list):
    def __init__(self, values, split="train"):
        super().__init__(values)
        self.split = split


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.vocab = CharacterVocabulary([PAD_TOKEN, UNK_TOKEN, NO_WORD_TOKEN, "A", "B"])
        self.encoder = FeatureEncoder(self.vocab).eval()

    def test_audio_native_and_aligned_shapes(self) -> None:
        audio = torch.randn(1, 16_000)
        result = self.encoder(
            audio=audio,
            audio_lengths=torch.tensor([16_000]),
            words=[[]],
            neck_timestamps=torch.tensor([[0.0, 0.25, 0.75]], dtype=torch.float64),
            sequence_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(result["audio_native_features"][0].shape, (98, 64))
        native_timestamps = result["audio_native_timestamps"][0]
        self.assertEqual(native_timestamps.shape, (98,))
        torch.testing.assert_close(native_timestamps[0], torch.tensor(0.0125, dtype=torch.float64))
        torch.testing.assert_close(native_timestamps[1] - native_timestamps[0], torch.tensor(0.01, dtype=torch.float64))
        self.assertEqual(result["aligned_audio"].shape, (1, 3, 64))

    def test_audio_batch_invariance(self) -> None:
        short = torch.randn(4_000)
        long = torch.randn(8_000)
        queries = torch.tensor([[0.03, 0.10, 0.20]], dtype=torch.float64)
        alone = self.encoder(
            short.unsqueeze(0), torch.tensor([len(short)]), [[]], queries, torch.ones(1, 3, dtype=torch.bool)
        )["aligned_audio"][0]
        padded = torch.zeros(2, len(long))
        padded[0, : len(short)] = short
        padded[1] = long
        together = self.encoder(
            padded,
            torch.tensor([len(short), len(long)]),
            [[], []],
            queries.expand(2, -1).clone(),
            torch.ones(2, 3, dtype=torch.bool),
        )["aligned_audio"][0]
        torch.testing.assert_close(alone, together, rtol=1e-6, atol=1e-7)

    def test_known_physical_timestamp_interpolation_and_edge_hold(self) -> None:
        native_features = torch.tensor([[1.0], [3.0], [7.0]])
        native_times = torch.tensor([0.10, 0.20, 0.40], dtype=torch.float64)
        queries = torch.tensor([0.0, 0.15, 0.30, 0.50], dtype=torch.float64)
        result = interpolate_at_timestamps(native_features, native_times, queries)
        torch.testing.assert_close(result[:, 0], torch.tensor([1.0, 2.0, 5.0, 7.0]))

    def test_nonzero_first_neck_timestamp_is_not_rebased(self) -> None:
        native_features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        native_times = torch.tensor([0.25, 0.35, 0.45, 0.55], dtype=torch.float64)
        queries = torch.tensor([0.35, 0.45, 0.55], dtype=torch.float64)
        result = interpolate_at_timestamps(native_features, native_times, queries)
        torch.testing.assert_close(result[:, 0], torch.tensor([1.0, 2.0, 3.0]))

    def test_text_half_open_intervals(self) -> None:
        words = [[
            {"text": "A", "local_start": 0.0, "local_end": 0.5},
            {"text": "B", "local_start": 0.5, "local_end": 1.0},
        ]]
        timestamps = torch.tensor([[0.25, 0.50, 0.75]], dtype=torch.float64)
        aligned = self.encoder.align_text(words, timestamps, torch.ones(1, 3, dtype=torch.bool))[0]
        expected_a = self.encoder.encode_word("A", timestamps.device)
        expected_b = self.encoder.encode_word("B", timestamps.device)
        torch.testing.assert_close(aligned[0], expected_a)
        torch.testing.assert_close(aligned[1], expected_b)
        torch.testing.assert_close(aligned[2], expected_b)

    def test_no_word_is_finite_learnable_feature(self) -> None:
        timestamps = torch.tensor([[0.25, 1.25]], dtype=torch.float64)
        words = [[{"text": "A", "local_start": 0.0, "local_end": 0.5}]]
        aligned = self.encoder.align_text(words, timestamps, torch.ones(1, 2, dtype=torch.bool))[0]
        expected = self.encoder.text_embedding.weight[self.vocab.no_word_id]
        self.assertTrue(torch.isfinite(aligned[1]).all())
        torch.testing.assert_close(aligned[1], expected)

    def test_vocab_is_train_only_and_unknown_characters_use_unk(self) -> None:
        train = _Samples([{"words": [{"text": "你好"}]}], split="train")
        vocab = build_vocab(train)
        self.assertIn("你", vocab.token_to_id)
        self.assertNotIn("特", vocab.token_to_id)
        self.assertEqual(vocab.encode("特殊词"), [vocab.unk_id, vocab.unk_id, vocab.unk_id])
        with self.assertRaises(ValueError):
            build_vocab(_Samples([{"words": [{"text": "泄漏"}]}], split="val"))

    def test_padding_features_are_zero(self) -> None:
        audio = torch.randn(2, 4_000)
        timestamps = torch.tensor(
            [[0.05, 0.15, float("nan")], [0.05, 0.15, 0.20]], dtype=torch.float64
        )
        sequence_mask = torch.tensor([[True, True, False], [True, True, True]])
        result = self.encoder(
            audio,
            torch.tensor([3_200, 4_000]),
            [[], []],
            timestamps,
            sequence_mask,
        )
        self.assertTrue(torch.equal(result["aligned_audio"][0, 2], torch.zeros(64)))
        self.assertTrue(torch.equal(result["aligned_text"][0, 2], torch.zeros(64)))
        self.assertTrue(torch.isfinite(result["aligned_audio"][sequence_mask]).all())
        self.assertTrue(torch.isfinite(result["aligned_text"][sequence_mask]).all())


if __name__ == "__main__":
    unittest.main()
