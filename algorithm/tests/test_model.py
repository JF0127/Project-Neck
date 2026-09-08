from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from algorithm.data import NeckMotionDataset, neck_motion_collate
from algorithm.features import CharacterVocabulary, FeatureEncoder, NO_WORD_TOKEN, PAD_TOKEN, UNK_TOKEN
from algorithm.models import BaselineModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "datasets" / "zhubo_shuo_lianbo"


def _inputs(lengths: list[int], max_length: int | None = None):
    batch_size = len(lengths)
    steps = max_length or max(lengths)
    audio = torch.randn(batch_size, steps, 64)
    text = torch.randn(batch_size, steps, 64)
    timestamps = torch.full((batch_size, steps), float("nan"), dtype=torch.float64)
    mask = torch.zeros(batch_size, steps, dtype=torch.bool)
    for batch_index, length in enumerate(lengths):
        mask[batch_index, :length] = True
        # Deliberately irregular physical times; no fixed-FPS assumption.
        timestamps[batch_index, :length] = torch.linspace(0.17, 2.83, length)
    return audio, text, timestamps, mask


class BaselineModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = BaselineModel()

    def test_output_shape(self) -> None:
        inputs = _inputs([100, 100])
        prediction = self.model(*inputs)
        self.assertEqual(prediction.shape, (2, 100, 3))

    def test_variable_length_padding(self) -> None:
        audio, text, timestamps, mask = _inputs([100, 60])
        prediction = self.model(audio, text, timestamps, mask)
        self.assertEqual(prediction.shape, (2, 100, 3))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.equal(prediction[1, 60:], torch.zeros(40, 3)))

    def test_first_frame_is_zero(self) -> None:
        prediction = self.model(*_inputs([20, 13]))
        torch.testing.assert_close(prediction[:, 0], torch.zeros(2, 3), rtol=0, atol=0)

    def test_gradients_reach_all_model_stages(self) -> None:
        audio, text, timestamps, mask = _inputs([18, 13])
        prediction = self.model(audio, text, timestamps, mask)
        loss = prediction[mask].square().mean()
        loss.backward()
        for stage_name, stage in (
            ("fusion", self.model.fusion),
            ("transformer", self.model.transformer),
            ("rpy_head", self.model.rpy_head),
        ):
            gradients = [parameter.grad for parameter in stage.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients), stage_name)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), stage_name)
            self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients), stage_name)

    def test_padding_invariance_in_eval_mode(self) -> None:
        self.model.eval()
        short_audio = torch.randn(1, 31, 64)
        short_text = torch.randn(1, 31, 64)
        short_times = torch.linspace(0.35, 1.71, 31, dtype=torch.float64).unsqueeze(0)
        short_mask = torch.ones(1, 31, dtype=torch.bool)
        with torch.no_grad():
            alone = self.model(short_audio, short_text, short_times, short_mask)[0]

            audio = torch.randn(2, 57, 64)
            text = torch.randn(2, 57, 64)
            audio[0, :31] = short_audio[0]
            text[0, :31] = short_text[0]
            timestamps = torch.full((2, 57), float("nan"), dtype=torch.float64)
            timestamps[0, :31] = short_times[0]
            timestamps[1] = torch.linspace(0.11, 3.9, 57)
            mask = torch.zeros(2, 57, dtype=torch.bool)
            mask[0, :31] = True
            mask[1] = True
            together = self.model(audio, text, timestamps, mask)[0, :31]
        torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-6)

    def test_forward_is_independent_of_ground_truth(self) -> None:
        parameters = set(inspect.signature(self.model.forward).parameters)
        self.assertNotIn("target_rpy", parameters)
        self.assertNotIn("valid_mask", parameters)
        self.assertNotIn("target_valid_mask", parameters)
        self.assertNotIn("reference_valid", parameters)


@unittest.skipUnless(DATASET_ROOT.exists(), "canonical Dataset V1 is not present")
class RealFeatureModelTest(unittest.TestCase):
    def test_real_batch_feature_to_prediction(self) -> None:
        dataset = NeckMotionDataset("train", DATASET_ROOT)
        batch = next(
            iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=neck_motion_collate))
        )
        vocab = CharacterVocabulary([PAD_TOKEN, UNK_TOKEN, NO_WORD_TOKEN])
        features = FeatureEncoder(vocab).eval()
        model = BaselineModel().eval()
        with torch.no_grad():
            conditions = features(
                batch["audio"],
                batch["audio_lengths"],
                batch["words"],
                batch["neck_timestamps"],
                batch["sequence_mask"],
            )
            prediction = model(
                conditions["aligned_audio"],
                conditions["aligned_text"],
                batch["neck_timestamps"],
                batch["sequence_mask"],
            )
        self.assertEqual(prediction.shape, (*batch["neck_timestamps"].shape, 3))
        self.assertEqual(prediction.shape[1], batch["target_rpy"].shape[1])
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
