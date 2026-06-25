"""Train a custom supervised semantic segmentation model on OpenEarthMap tiles."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.tensors import mask_to_label_tensor
from sim_satellite.model import ModelConfig
from sim_satellite.segmentation_compare import (
    SegmentationModelName,
    _batch,
    _build_model,
    _count_parameters,
    _evaluate_model,
    _seed_everything,
    _tiles_for_samples,
    _TileSegmentationDataset,
)


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    manifest_path: str = "data/openearthmap/manifest.jsonl"
    checkpoint_path: str = "data/segmentation/deeplab-lite.pt"
    model: SegmentationModelName = "deeplab-lite"
    tile_grid: int = 3
    image_size: int = 128
    train_images: int = 512
    val_images: int = 128
    max_steps: int = 2000
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    class_weight_power: float = 0.5
    unlabeled_weight: float = 0.25
    eval_every_steps: int = 250
    device: str = "cpu"
    seed: int = 2026


@dataclass(frozen=True)
class SegmentationTrainingResult:
    checkpoint_path: Path
    metrics_path: Path
    model: str
    train_tiles: int
    val_tiles: int
    steps: int
    parameters: int
    final_train_loss: float
    best_mean_iou: float
    best_pixel_accuracy: float
    final_mean_iou: float
    final_pixel_accuracy: float
    class_iou: dict[str, float | None]

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["checkpoint_path"] = str(self.checkpoint_path)
        value["metrics_path"] = str(self.metrics_path)
        return value


def train_segmentation_model(
    config: SegmentationTrainingConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> SegmentationTrainingResult:
    _validate_config(config)
    import torch
    import torch.nn.functional as F

    _seed_everything(config.seed, torch=torch)
    train_tiles, val_tiles = _split_train_val_tiles(config)
    train_dataset = _TileSegmentationDataset(
        train_tiles,
        image_size=config.image_size,
        torch=torch,
    )
    val_dataset = _TileSegmentationDataset(
        val_tiles,
        image_size=config.image_size,
        torch=torch,
    )
    model = _build_model(config.model, num_classes=len(OPENEARTHMAP_CLASSES), torch=torch)
    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_counts = _class_counts(
        train_tiles,
        image_size=config.image_size,
        torch=torch,
    )
    class_weights = _class_weights_from_counts(
        class_counts,
        power=config.class_weight_power,
        unlabeled_weight=config.unlabeled_weight,
        torch=torch,
    ).to(config.device)
    rng = random.Random(config.seed)
    losses: list[float] = []
    best_metrics: dict[str, Any] | None = None
    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = checkpoint_path.with_suffix(".json")

    _report(
        progress,
        f"Training {config.model} on {len(train_tiles)} tiles; "
        f"validating on {len(val_tiles)} tiles",
    )
    for step in range(1, config.max_steps + 1):
        indexes = [rng.randrange(len(train_dataset)) for _ in range(config.batch_size)]
        images, labels = _batch(train_dataset, indexes, device=config.device, torch=torch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
        dice_loss = _multiclass_dice_loss(
            logits,
            labels,
            include_unlabeled=False,
            torch=torch,
        )
        loss = ce_loss + config.dice_weight * dice_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        should_eval = step == config.max_steps or step % config.eval_every_steps == 0
        should_report = step == 1 or should_eval or step % max(1, config.max_steps // 10) == 0
        if should_report:
            _report(progress, f"step={step} loss={losses[-1]:.4f}")
        if should_eval:
            metrics = _evaluate_model(
                model,
                val_dataset,
                include_unlabeled_iou=False,
                device=config.device,
                torch=torch,
            )
            _report(
                progress,
                f"step={step} val_mean_iou={metrics['mean_iou']:.4f} "
                f"val_pixel_accuracy={metrics['pixel_accuracy']:.4f}",
            )
            if best_metrics is None or metrics["mean_iou"] > best_metrics["mean_iou"]:
                best_metrics = metrics
                _save_checkpoint(checkpoint_path, model, config, metrics, step=step, torch=torch)

    final_metrics = _evaluate_model(
        model,
        val_dataset,
        include_unlabeled_iou=False,
        device=config.device,
        torch=torch,
    )
    if best_metrics is None:
        best_metrics = final_metrics
        _save_checkpoint(
            checkpoint_path,
            model,
            config,
            best_metrics,
            step=config.max_steps,
            torch=torch,
        )

    result = SegmentationTrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        model=config.model,
        train_tiles=len(train_tiles),
        val_tiles=len(val_tiles),
        steps=config.max_steps,
        parameters=_count_parameters(model),
        final_train_loss=sum(losses[-min(len(losses), 100) :]) / min(len(losses), 100),
        best_mean_iou=float(best_metrics["mean_iou"]),
        best_pixel_accuracy=float(best_metrics["pixel_accuracy"]),
        final_mean_iou=float(final_metrics["mean_iou"]),
        final_pixel_accuracy=float(final_metrics["pixel_accuracy"]),
        class_iou=final_metrics["class_iou"],
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "result": result.to_json(),
                "class_weights": {
                    class_name: float(class_weights[index].detach().cpu().item())
                    for index, class_name in enumerate(OPENEARTHMAP_CLASSES)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _split_train_val_tiles(
    config: SegmentationTrainingConfig,
) -> tuple[Any, Any]:
    samples = sorted(read_manifest(config.manifest_path), key=lambda sample: str(sample.image_path))
    random.Random(config.seed).shuffle(samples)
    needed = config.train_images + config.val_images
    if len(samples) < needed:
        raise ValueError(
            f"Requested {needed} images but manifest only contains {len(samples)} samples"
        )
    train_samples = samples[: config.train_images]
    val_samples = samples[config.train_images : needed]
    return _tiles_for_samples(train_samples, config.tile_grid), _tiles_for_samples(
        val_samples,
        config.tile_grid,
    )


def _class_counts(
    tiles: Any,
    *,
    image_size: int,
    torch: Any,
) -> Any:
    counts = torch.zeros(len(OPENEARTHMAP_CLASSES), dtype=torch.long)
    model_config = ModelConfig(
        image_size=image_size,
        mask_channels=len(OPENEARTHMAP_CLASSES),
    )
    for tile in tiles:
        with Image.open(tile.sample.mask_path) as mask:
            labels = mask_to_label_tensor(
                mask,
                torch=torch,
                config=model_config,
                crop_box=tile.crop_box,
                size=(image_size, image_size),
            )
        counts += torch.bincount(labels.flatten(), minlength=len(OPENEARTHMAP_CLASSES))
    return counts


def _class_weights_from_counts(
    counts: Any,
    *,
    power: float,
    unlabeled_weight: float,
    torch: Any,
) -> Any:
    if power < 0:
        raise ValueError("power must be non-negative")
    if unlabeled_weight < 0:
        raise ValueError("unlabeled_weight must be non-negative")
    counts = counts.float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    if not bool(present.any().item()):
        return weights
    median = counts[present].median()
    weights[present] = (median / counts[present]).pow(power)
    weights[0] *= unlabeled_weight
    nonzero = weights > 0
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights


def _multiclass_dice_loss(
    logits: Any,
    labels: Any,
    *,
    include_unlabeled: bool,
    torch: Any,
) -> Any:
    import torch.nn.functional as F

    num_classes = logits.shape[1]
    probabilities = logits.softmax(dim=1)
    target = F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    start_class = 0 if include_unlabeled else 1
    probabilities = probabilities[:, start_class:]
    target = target[:, start_class:]
    intersection = (probabilities * target).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _save_checkpoint(
    path: Path,
    model: Any,
    config: SegmentationTrainingConfig,
    metrics: dict[str, Any],
    *,
    step: int,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": config.model,
            "classes": list(OPENEARTHMAP_CLASSES),
            "config": asdict(config),
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def _validate_config(config: SegmentationTrainingConfig) -> None:
    if config.tile_grid < 1:
        raise ValueError("tile_grid must be positive")
    if config.image_size < 16:
        raise ValueError("image_size must be at least 16")
    if config.train_images < 1 or config.val_images < 1:
        raise ValueError("train_images and val_images must be positive")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.dice_weight < 0:
        raise ValueError("dice_weight must be non-negative")
    if config.eval_every_steps < 1:
        raise ValueError("eval_every_steps must be positive")
    if not Path(config.manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest_path}")


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _build_click_command():
    import click

    @click.command(
        "segmentation-train",
        help="Train a custom supervised segmentation checkpoint on OpenEarthMap tiles.",
    )
    @click.option(
        "--manifest",
        "manifest_path",
        type=click.Path(path_type=Path),
        default=Path(SegmentationTrainingConfig.manifest_path),
    )
    @click.option(
        "--checkpoint",
        "checkpoint_path",
        type=click.Path(path_type=Path),
        default=Path(SegmentationTrainingConfig.checkpoint_path),
    )
    @click.option(
        "--model",
        "model_name",
        type=click.Choice(("unet", "deeplab-lite", "segformer-lite")),
        default=SegmentationTrainingConfig.model,
        show_default=True,
    )
    @click.option(
        "--tile-grid",
        type=int,
        default=SegmentationTrainingConfig.tile_grid,
        show_default=True,
    )
    @click.option(
        "--image-size",
        type=int,
        default=SegmentationTrainingConfig.image_size,
        show_default=True,
    )
    @click.option(
        "--train-images",
        type=int,
        default=SegmentationTrainingConfig.train_images,
        show_default=True,
    )
    @click.option(
        "--val-images",
        type=int,
        default=SegmentationTrainingConfig.val_images,
        show_default=True,
    )
    @click.option(
        "--max-steps",
        type=int,
        default=SegmentationTrainingConfig.max_steps,
        show_default=True,
    )
    @click.option(
        "--batch-size",
        type=int,
        default=SegmentationTrainingConfig.batch_size,
        show_default=True,
    )
    @click.option(
        "--learning-rate",
        type=float,
        default=SegmentationTrainingConfig.learning_rate,
        show_default=True,
    )
    @click.option(
        "--weight-decay",
        type=float,
        default=SegmentationTrainingConfig.weight_decay,
        show_default=True,
    )
    @click.option(
        "--dice-weight",
        type=float,
        default=SegmentationTrainingConfig.dice_weight,
        show_default=True,
    )
    @click.option(
        "--class-weight-power",
        type=float,
        default=SegmentationTrainingConfig.class_weight_power,
        show_default=True,
    )
    @click.option(
        "--unlabeled-weight",
        type=float,
        default=SegmentationTrainingConfig.unlabeled_weight,
        show_default=True,
    )
    @click.option(
        "--eval-every-steps",
        type=int,
        default=SegmentationTrainingConfig.eval_every_steps,
        show_default=True,
    )
    @click.option("--device", default=SegmentationTrainingConfig.device, show_default=True)
    @click.option("--seed", type=int, default=SegmentationTrainingConfig.seed, show_default=True)
    def command(
        manifest_path: Path,
        checkpoint_path: Path,
        model_name: SegmentationModelName,
        tile_grid: int,
        image_size: int,
        train_images: int,
        val_images: int,
        max_steps: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        dice_weight: float,
        class_weight_power: float,
        unlabeled_weight: float,
        eval_every_steps: int,
        device: str,
        seed: int,
    ) -> None:
        try:
            result = train_segmentation_model(
                SegmentationTrainingConfig(
                    manifest_path=str(manifest_path),
                    checkpoint_path=str(checkpoint_path),
                    model=model_name,
                    tile_grid=tile_grid,
                    image_size=image_size,
                    train_images=train_images,
                    val_images=val_images,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dice_weight=dice_weight,
                    class_weight_power=class_weight_power,
                    unlabeled_weight=unlabeled_weight,
                    eval_every_steps=eval_every_steps,
                    device=device,
                    seed=seed,
                ),
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


cli = _build_click_command()
