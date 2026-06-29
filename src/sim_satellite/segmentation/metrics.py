"""Segmentation losses and evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.segmentation.data import _PreparedSampleCache
from sim_satellite.segmentation.utils import ProgressReporter, _report, _should_report


def _class_weights(
    samples: Sequence[SatelliteSample],
    *,
    cache: _PreparedSampleCache,
    power: float,
    unlabeled_weight: float,
    progress: ProgressReporter | None,
) -> torch.Tensor:
    counts = torch.zeros(len(OPENEARTHMAP_CLASSES), dtype=torch.long)
    for index, sample in enumerate(samples, start=1):
        labels = cache.get(sample).mask.long()
        counts += torch.bincount(labels.flatten(), minlength=len(OPENEARTHMAP_CLASSES))
        if _should_report(index, len(samples)):
            _report(progress, f"Scanned class balance {index}/{len(samples)} samples")
    return _class_weights_from_counts(
        counts,
        power=power,
        unlabeled_weight=unlabeled_weight,
    )


def _class_weights_from_counts(
    counts: torch.Tensor,
    *,
    power: float,
    unlabeled_weight: float,
) -> torch.Tensor:
    if power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    if unlabeled_weight < 0.0:
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
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_unlabeled: bool,
) -> torch.Tensor:
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


def _confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    valid = (labels >= 0) & (labels < num_classes)
    encoded = num_classes * labels[valid].long() + predictions[valid].long()
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )


def _segmentation_metrics(
    confusion: torch.Tensor,
    *,
    include_unlabeled_iou: bool,
) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    union = predicted + actual - true_positive
    class_iou_values: list[float] = []
    class_iou: dict[str, float | None] = {}
    for class_id, class_name in enumerate(OPENEARTHMAP_CLASSES):
        if union[class_id].item() == 0:
            class_iou[class_name] = None
            continue
        value = float((true_positive[class_id] / union[class_id]).item())
        class_iou[class_name] = value
        if include_unlabeled_iou or class_id != 0:
            class_iou_values.append(value)
    total = int(confusion.sum().item())
    pixel_accuracy = float(true_positive.sum().item() / total) if total else 0.0
    mean_iou = sum(class_iou_values) / len(class_iou_values) if class_iou_values else 0.0
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "class_iou": class_iou,
    }
