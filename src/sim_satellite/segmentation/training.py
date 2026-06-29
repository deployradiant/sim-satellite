"""Training loop for VAE-fused semantic segmentation classifiers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.segmentation.checkpoint import _save_checkpoint
from sim_satellite.segmentation.config import (
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
    _validate_training_config,
)
from sim_satellite.segmentation.data import (
    _load_segmentation_batch,
    _PreparedSampleCache,
    _random_square_crop_box,
    _split_training_samples,
)
from sim_satellite.segmentation.inference import _evaluate_samples
from sim_satellite.segmentation.metrics import _class_weights, _multiclass_dice_loss
from sim_satellite.segmentation.models import (
    _forward_segmentation_model,
    _segmentation_input_channels,
    build_segmentation_model,
)
from sim_satellite.segmentation.utils import ProgressReporter, _report
from sim_satellite.segmentation.vae import _encode_vae_features, _load_vae_feature_encoder


def train_segmentation_model(
    config: SegmentationTrainingConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationTrainingResult:
    _validate_training_config(config)
    _seed_everything(config.seed)

    device = torch.device(config.device)
    train_samples, validation_samples = _split_training_samples(config)
    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    checkpoint_config = asdict(config)
    vae_encoder = _load_vae_feature_encoder(checkpoint_config, device=device)
    checkpoint_config["vae_latent_channels"] = vae_encoder.latent_channels
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        input_channels=_segmentation_input_channels(config),
        vae_latent_channels=vae_encoder.latent_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    _report(progress, f"Scanning class balance across {len(train_samples)} samples")
    class_weights = _class_weights(
        train_samples,
        cache=cache,
        power=config.class_weight_power,
        unlabeled_weight=config.unlabeled_weight,
        progress=progress,
    ).to(device)

    rng = random.Random(config.seed)
    losses: list[float] = []
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = checkpoint_path.with_suffix(".json")

    _report(
        progress,
        f"Training VAE-fused segmenter on {len(train_samples)} samples; "
        f"validating on {len(validation_samples)} samples",
    )
    model.train()
    for step in range(1, config.steps + 1):
        batch_samples = tuple(
            train_samples[rng.randrange(len(train_samples))] for _ in range(config.batch_size)
        )
        crop_boxes = [
            _random_square_crop_box(cache.get(sample), config.image_size, rng)
            for sample in batch_samples
        ]
        images, labels = _load_segmentation_batch(
            batch_samples,
            crop_boxes,
            cache=cache,
            image_size=config.image_size,
            context_size=config.context_size,
            device=device,
            augmentation_config=config,
            rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_segmentation_model(
            model,
            images,
            vae_latents=_encode_vae_features(vae_encoder, images),
        )
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
        dice_loss = _multiclass_dice_loss(
            logits,
            labels,
            include_unlabeled=False,
        )
        loss = ce_loss + config.dice_weight * dice_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        should_eval = step == config.steps or step % config.eval_every == 0
        should_report = step == 1 or should_eval or step % max(1, config.steps // 10) == 0
        if should_report:
            _report(progress, f"step={step} loss={losses[-1]:.4f}")
        if should_eval:
            metrics = _evaluate_samples(
                model,
                validation_samples,
                image_size=config.image_size,
                context_size=config.context_size,
                device=device,
                cache=cache,
                vae_encoder=vae_encoder,
            )
            _report(
                progress,
                f"step={step} val_mean_iou={metrics['mean_iou']:.4f} "
                f"val_pixel_accuracy={metrics['pixel_accuracy']:.4f}",
            )
            if best_metrics is None or metrics["mean_iou"] > best_metrics["mean_iou"]:
                best_metrics = metrics
                best_step = step
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    config=checkpoint_config,
                    metrics=metrics,
                    step=step,
                )
            model.train()

    final_metrics = _evaluate_samples(
        model,
        validation_samples,
        image_size=config.image_size,
        context_size=config.context_size,
        device=device,
        cache=cache,
        vae_encoder=vae_encoder,
    )
    if best_metrics is None:
        best_metrics = final_metrics
        best_step = config.steps
        _save_checkpoint(
            checkpoint_path,
            model,
            config=checkpoint_config,
            metrics=best_metrics,
            step=config.steps,
        )

    recent_losses = losses[-min(len(losses), 100) :]
    result = SegmentationTrainingResult(
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        steps=config.steps,
        parameters=_count_parameters(model),
        final_train_loss=sum(recent_losses) / len(recent_losses),
        best_step=best_step,
        best_mean_iou=float(best_metrics["mean_iou"]),
        best_pixel_accuracy=float(best_metrics["pixel_accuracy"]),
        best_class_iou=best_metrics["class_iou"],
        final_mean_iou=float(final_metrics["mean_iou"]),
        final_pixel_accuracy=float(final_metrics["pixel_accuracy"]),
        class_iou=final_metrics["class_iou"],
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "config": checkpoint_config,
                "result": asdict(result),
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


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)
