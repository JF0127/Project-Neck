from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from algorithm.data import neck_motion_collate
from algorithm.features import FeatureEncoder, build_vocab
from algorithm.losses import BaselineLoss
from algorithm.models import BaselineModel
from algorithm.train import load_checkpoint, run_training, save_checkpoint


class TinyDataset(Dataset):
    def __init__(self, count: int, split: str) -> None:
        self.split = split
        generator = torch.Generator().manual_seed(100 if split == "train" else 200)
        self.samples = []
        for index in range(count):
            timestamps = torch.tensor([0.0, 0.08, 0.21, 0.37], dtype=torch.float64)
            target = timestamps.float().unsqueeze(1) * torch.tensor([[0.1, -0.2, 0.3]])
            self.samples.append(
                {
                    "fragment_id": f"{split}_{index}",
                    "clip_id": f"clip_{split}_{index}",
                    "source_video_id": f"source_{split}_{index}",
                    "split": split,
                    "audio": torch.randn(3_200 + index * 160, generator=generator),
                    "audio_length": 3_200 + index * 160,
                    "text": "AB",
                    "words": [
                        {"text": "A", "local_start": 0.0, "local_end": 0.05},
                        {"text": "B", "local_start": 0.16, "local_end": 0.25},
                    ],
                    "neck_timestamps": timestamps,
                    "raw_rpy": target.clone(),
                    "target_rpy": target,
                    "valid_mask": torch.ones(4, dtype=torch.bool),
                    "target_valid_mask": torch.ones(4, dtype=torch.bool),
                    "reference_valid": True,
                    "metadata": {},
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def iter_word_entries(self):
        for sample in self.samples:
            yield sample["words"]


def _config(output_root: str) -> dict:
    return {
        "data": {"dataset_root": "unused", "split_version": "split_v1", "num_workers": 0},
        "target": {
            "representation": "rpy_offset",
            "reference": "first_neck_sample",
            "skip_invalid_reference": True,
        },
        "features": {"audio_dim": 64, "text_dim": 64},
        "model": {
            "hidden_dim": 128,
            "layers": 2,
            "heads": 4,
            "ffn_dim": 256,
            "dropout": 0.1,
        },
        "loss": {"position_weight": 1.0, "velocity_weight": 0.1, "position_beta": 0.05},
        "training": {
            "seed": 42,
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "log_every": 1,
        },
        "wandb": {"enabled": False, "project": "test", "name": "test", "tags": []},
        "output": {"root": output_root, "experiment": "test"},
    }


class TrainingTests(unittest.TestCase):
    def test_feature_model_gradients_and_optimizer_step(self) -> None:
        dataset = TinyDataset(2, "train")
        vocab = build_vocab(dataset)
        feature_encoder = FeatureEncoder(vocab)
        model = BaselineModel(dropout=0.0)
        modules = (feature_encoder, model)
        parameters = [parameter for module in modules for parameter in module.parameters()]
        optimizer = AdamW(parameters, lr=1e-3)
        criterion = BaselineLoss()
        batch = neck_motion_collate([dataset[0], dataset[1]])
        before_audio = feature_encoder.audio_encoder.network[0].weight.detach().clone()
        before_text = feature_encoder.text_embedding.weight.detach().clone()
        before_model = model.fusion[0].weight.detach().clone()

        conditions = feature_encoder(
            batch["audio"], batch["audio_lengths"], batch["words"],
            batch["neck_timestamps"], batch["sequence_mask"],
        )
        prediction = model(
            conditions["aligned_audio"], conditions["aligned_text"],
            batch["neck_timestamps"], batch["sequence_mask"],
        )
        losses = criterion(
            prediction, batch["target_rpy"], batch["neck_timestamps"],
            batch["sequence_mask"], batch["target_valid_mask"],
        )
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(feature_encoder.audio_encoder.network[0].weight.grad).all())
        self.assertTrue(torch.isfinite(feature_encoder.text_embedding.weight.grad).all())
        self.assertTrue(torch.isfinite(model.fusion[0].weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before_audio, feature_encoder.audio_encoder.network[0].weight))
        self.assertFalse(torch.equal(before_text, feature_encoder.text_embedding.weight))
        self.assertFalse(torch.equal(before_model, model.fusion[0].weight))

    def test_checkpoint_round_trip_preserves_prediction(self) -> None:
        dataset = TinyDataset(1, "train")
        vocab = build_vocab(dataset)
        feature_encoder = FeatureEncoder(vocab).eval()
        model = BaselineModel(dropout=0.0).eval()
        optimizer = AdamW([*feature_encoder.parameters(), *model.parameters()])
        batch = neck_motion_collate([dataset[0]])
        with torch.no_grad():
            conditions = feature_encoder(
                batch["audio"], batch["audio_lengths"], batch["words"],
                batch["neck_timestamps"], batch["sequence_mask"],
            )
            expected = model(
                conditions["aligned_audio"], conditions["aligned_text"],
                batch["neck_timestamps"], batch["sequence_mask"],
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, feature_encoder, model, optimizer, 1, 3, {}, vocab, 0.2)
            restored_features = FeatureEncoder(vocab).eval()
            restored_model = BaselineModel(dropout=0.0).eval()
            restored_optimizer = AdamW([*restored_features.parameters(), *restored_model.parameters()])
            checkpoint = load_checkpoint(path, restored_features, restored_model, restored_optimizer)
            with torch.no_grad():
                conditions = restored_features(
                    batch["audio"], batch["audio_lengths"], batch["words"],
                    batch["neck_timestamps"], batch["sequence_mask"],
                )
                actual = restored_model(
                    conditions["aligned_audio"], conditions["aligned_text"],
                    batch["neck_timestamps"], batch["sequence_mask"],
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["vocab"]["tokens"], list(vocab.tokens))

    def test_one_epoch_runs_with_wandb_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_training(
                _config(directory),
                train_dataset=TinyDataset(2, "train"),
                val_dataset=TinyDataset(1, "val"),
            )
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "config.yaml").is_file())
            self.assertTrue((run_dir / "vocab.json").is_file())
            self.assertTrue((run_dir / "metrics.jsonl").is_file())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").is_file())
            self.assertTrue((run_dir / "checkpoints" / "last.pt").is_file())
            self.assertTrue(torch.isfinite(torch.tensor(result["metrics"]["val/position_mae"])))


if __name__ == "__main__":
    unittest.main()
