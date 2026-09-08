"""Single Baseline V1 training entry: ``python -m algorithm.train --config ...``."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .data import NeckMotionDataset, neck_motion_collate
from .features import CharacterVocabulary, FeatureEncoder, build_vocab
from .losses import BaselineLoss
from .metrics import MotionMetricAccumulator
from .models import BaselineModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a YAML mapping: {config_path}")
    return config


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _canonical_artifact_path(dataset: NeckMotionDataset, row: Mapping[str, Any], filename: str) -> Path:
    manifest_path = Path(str(row["neck_rpy_path"])).expanduser().parent / filename
    if manifest_path.is_file():
        return manifest_path
    return (
        dataset.dataset_root
        / "fragments"
        / "fragment_v1"
        / "fragments"
        / str(row["fragment_id"])
        / filename
    )


def reference_valid_indices(dataset: Dataset) -> list[int]:
    """Inspect only canonical neck-valid arrays when possible; never alter membership/artifacts."""
    if isinstance(dataset, NeckMotionDataset):
        indices: list[int] = []
        for index, row in enumerate(dataset.manifest_rows):
            path = _canonical_artifact_path(dataset, row, "neck_valid.npy")
            try:
                valid = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"fragment_id={row['fragment_id']} path={path}: {exc}") from exc
            if valid.ndim != 1 or not len(valid) or valid.dtype != np.bool_:
                raise ValueError(
                    f"fragment_id={row['fragment_id']} path={path}: expected non-empty bool [T]"
                )
            if bool(valid[0]):
                indices.append(index)
        return indices
    return [index for index in range(len(dataset)) if bool(dataset[index]["reference_valid"])]


class ReferenceValidSubset(Dataset):
    """Experiment-side view that excludes undefined first-sample references."""

    def __init__(self, dataset: Dataset, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def iter_word_entries(self):
        base_iterator = getattr(self.dataset, "iter_word_entries", None)
        if callable(base_iterator):
            wanted = set(self.indices)
            for index, words in enumerate(base_iterator()):
                if index in wanted:
                    yield words
        else:
            for index in self.indices:
                yield self.dataset[index]["words"]


def filter_invalid_references(dataset: Dataset) -> ReferenceValidSubset:
    return ReferenceValidSubset(dataset, reference_valid_indices(dataset))


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_keys = (
        "audio",
        "audio_lengths",
        "neck_timestamps",
        "target_rpy",
        "sequence_mask",
        "target_valid_mask",
    )
    moved = dict(batch)
    for key in tensor_keys:
        moved[key] = batch[key].to(device)
    return moved


def _forward(
    feature_encoder: FeatureEncoder,
    model: BaselineModel,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    conditions = feature_encoder(
        batch["audio"],
        batch["audio_lengths"],
        batch["words"],
        batch["neck_timestamps"],
        batch["sequence_mask"],
    )
    return model(
        conditions["aligned_audio"],
        conditions["aligned_text"],
        batch["neck_timestamps"],
        batch["sequence_mask"],
    )


def _loss_epoch_summary(
    position_sum: float,
    position_count: int,
    velocity_sum: float,
    velocity_count: int,
    criterion: BaselineLoss,
) -> dict[str, float]:
    position = position_sum / position_count if position_count else 0.0
    velocity = velocity_sum / velocity_count if velocity_count else 0.0
    return {
        "loss": criterion.position_weight * position + criterion.velocity_weight * velocity,
        "position_loss": position,
        "velocity_loss": velocity,
    }


def validate(
    feature_encoder: FeatureEncoder,
    model: BaselineModel,
    loader: DataLoader,
    criterion: BaselineLoss,
    device: torch.device,
) -> dict[str, float | int]:
    feature_encoder.eval()
    model.eval()
    metric_accumulator = MotionMetricAccumulator()
    position_sum = velocity_sum = 0.0
    position_count = velocity_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            prediction = _forward(feature_encoder, model, batch)
            losses = criterion(
                prediction,
                batch["target_rpy"],
                batch["neck_timestamps"],
                batch["sequence_mask"],
                batch["target_valid_mask"],
            )
            n_position = losses["num_position_samples"]
            n_velocity = losses["num_velocity_samples"]
            position_sum += losses["position_loss"].item() * n_position
            velocity_sum += losses["velocity_loss"].item() * n_velocity
            position_count += n_position
            velocity_count += n_velocity
            metric_accumulator.update(
                prediction,
                batch["target_rpy"],
                batch["neck_timestamps"],
                batch["sequence_mask"],
                batch["target_valid_mask"],
            )
    return {
        **_loss_epoch_summary(
            position_sum, position_count, velocity_sum, velocity_count, criterion
        ),
        **metric_accumulator.compute(),
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: str | Path,
    feature_encoder: FeatureEncoder,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    vocab: CharacterVocabulary,
    best_metric: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_encoder_state_dict": feature_encoder.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": copy.deepcopy(dict(config)),
        "vocab": {"tokens": list(vocab.tokens)},
        "best_metric": best_metric,
        "rng_state": _rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    feature_encoder: FeatureEncoder,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    feature_encoder.load_state_dict(checkpoint["feature_encoder_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"algorithm.train.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _create_run_dir(config: Mapping[str, Any]) -> Path:
    root = _resolve_repo_path(config["output"]["root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / config["output"]["experiment"] / f"run_{timestamp}_seed{config['training']['seed']}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def run_training(
    config: Mapping[str, Any],
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(config))
    if config["data"].get("split_version") != "split_v1":
        raise ValueError("Baseline V1 requires canonical split_v1")
    if not config["target"].get("skip_invalid_reference", False):
        raise ValueError("rpy_offset Baseline V1 requires skip_invalid_reference=true")
    seed = int(config["training"]["seed"])
    set_seed(seed)

    if train_dataset is None or val_dataset is None:
        dataset_root = _resolve_repo_path(config["data"]["dataset_root"])
        train_dataset = train_dataset or NeckMotionDataset("train", dataset_root)
        val_dataset = val_dataset or NeckMotionDataset("val", dataset_root)
    original_counts = {"train": len(train_dataset), "val": len(val_dataset)}
    train_data = filter_invalid_references(train_dataset)
    val_data = filter_invalid_references(val_dataset)
    valid_counts = {"train": len(train_data), "val": len(val_data)}
    if not len(train_data) or not len(val_data):
        raise ValueError("train and val must each contain reference-valid fragments")

    vocab = build_vocab(train_data)
    feature_config = config["features"]
    model_config = config["model"]
    feature_encoder = FeatureEncoder(
        vocab,
        audio_feature_dim=int(feature_config["audio_dim"]),
        text_feature_dim=int(feature_config["text_dim"]),
    )
    model = BaselineModel(
        audio_dim=int(feature_config["audio_dim"]),
        text_dim=int(feature_config["text_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["layers"]),
        num_heads=int(model_config["heads"]),
        ffn_dim=int(model_config["ffn_dim"]),
        dropout=float(model_config["dropout"]),
    )
    requested_device = str(config["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    feature_encoder.to(device)
    model.to(device)

    feature_parameters = sum(p.numel() for p in feature_encoder.parameters() if p.requires_grad)
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [
        p for module in (feature_encoder, model) for p in module.parameters() if p.requires_grad
    ]
    optimizer = AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer_parameters = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    if optimizer_parameters != feature_parameters + model_parameters:
        raise RuntimeError("optimizer does not contain every trainable parameter exactly once")
    config["parameters"] = {
        "feature_encoder_trainable": feature_parameters,
        "baseline_model_trainable": model_parameters,
        "total_trainable": feature_parameters + model_parameters,
        "optimizer": optimizer_parameters,
    }
    config["data"]["reference_valid_counts"] = valid_counts
    config["data"]["original_counts"] = original_counts

    criterion = BaselineLoss(
        position_weight=float(config["loss"]["position_weight"]),
        velocity_weight=float(config["loss"]["velocity_weight"]),
        position_beta=float(config["loss"]["position_beta"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "collate_fn": neck_motion_collate,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    run_dir = _create_run_dir(config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    vocab.save(run_dir / "vocab.json")
    logger = _make_logger(run_dir)
    logger.info("device=%s original_counts=%s reference_valid_counts=%s", device, original_counts, valid_counts)
    logger.info("trainable parameters: %s", config["parameters"])

    wandb_run = None
    if config["wandb"].get("enabled", False):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb.enabled=true but wandb is not installed") from exc
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["name"],
            config=config,
            tags=config["wandb"].get("tags", []),
        )

    metrics_path = run_dir / "metrics.jsonl"
    global_step = 0
    best_metric = float("inf")
    last_record: dict[str, Any] = {}
    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            feature_encoder.train()
            model.train()
            position_sum = velocity_sum = 0.0
            position_count = velocity_count = 0
            for raw_batch in train_loader:
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = _forward(feature_encoder, model, batch)
                losses = criterion(
                    prediction,
                    batch["target_rpy"],
                    batch["neck_timestamps"],
                    batch["sequence_mask"],
                    batch["target_valid_mask"],
                )
                losses["loss"].backward()
                grad_norm = clip_grad_norm_(parameters, float(config["training"]["grad_clip"]))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
                optimizer.step()
                global_step += 1

                n_position = losses["num_position_samples"]
                n_velocity = losses["num_velocity_samples"]
                position_sum += losses["position_loss"].item() * n_position
                velocity_sum += losses["velocity_loss"].item() * n_velocity
                position_count += n_position
                velocity_count += n_velocity
                if global_step % int(config["training"]["log_every"]) == 0:
                    step_metrics = {
                        "train/loss": losses["loss"].item(),
                        "train/position_loss": losses["position_loss"].item(),
                        "train/velocity_loss": losses["velocity_loss"].item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }
                    logger.info("step=%d %s", global_step, step_metrics)
                    if wandb_run is not None:
                        wandb_run.log(step_metrics, step=global_step)

            train_metrics = _loss_epoch_summary(
                position_sum, position_count, velocity_sum, velocity_count, criterion
            )
            val_metrics = validate(feature_encoder, model, val_loader, criterion, device)
            last_record = {
                "epoch": epoch,
                "global_step": global_step,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_record, ensure_ascii=False) + "\n")
            logger.info("epoch=%d metrics=%s", epoch, last_record)
            if wandb_run is not None:
                wandb_run.log(last_record, step=global_step)

            position_mae = float(val_metrics["position_mae"])
            if position_mae < best_metric:
                best_metric = position_mae
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    feature_encoder, model, optimizer, epoch, global_step,
                    config, vocab, best_metric,
                )
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                feature_encoder, model, optimizer, epoch, global_step,
                config, vocab, best_metric,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

    return {
        "run_dir": str(run_dir),
        "metrics": last_record,
        "best_metric": best_metric,
        "original_counts": original_counts,
        "reference_valid_counts": valid_counts,
        "parameters": config["parameters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Audio+Text Baseline V1")
    parser.add_argument("--config", required=True, help="Path to baseline YAML config")
    parser.add_argument("--epochs", type=int, help="Override epochs (useful for smoke checks)")
    parser.add_argument(
        "--wandb-disabled", action="store_true", help="Disable W&B without editing the config"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.wandb_disabled:
        config["wandb"]["enabled"] = False
    result = run_training(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
