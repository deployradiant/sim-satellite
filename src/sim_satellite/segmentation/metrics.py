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


def _focal_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    log_probability = F.log_softmax(logits, dim=1)
    target_log_probability = log_probability.gather(1, labels.long().unsqueeze(1)).squeeze(1)
    probability = target_log_probability.exp()
    return ((1.0 - probability).pow(gamma) * ce_loss).mean()


def _lovasz_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_unlabeled: bool,
    max_pixels: int = 0,
) -> torch.Tensor:
    if max_pixels < 0:
        raise ValueError("lovasz_max_pixels must be non-negative")
    probabilities = logits.softmax(dim=1)
    flat_probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    flat_labels = labels.reshape(-1).long()
    start_class = 0 if include_unlabeled else 1
    if max_pixels > 0 and flat_labels.numel() > max_pixels:
        indices = _lovasz_sample_indices(
            flat_labels,
            num_classes=logits.shape[1],
            start_class=start_class,
            max_pixels=max_pixels,
        )
        flat_probabilities = flat_probabilities[indices]
        flat_labels = flat_labels[indices]

    class_losses: list[torch.Tensor] = []
    for class_id in range(start_class, logits.shape[1]):
        foreground = (flat_labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - flat_probabilities[:, class_id]).abs()
        sorted_errors, order = torch.sort(errors, descending=True)
        sorted_foreground = foreground[order]
        class_losses.append(torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground)))
    if not class_losses:
        return logits.sum() * 0.0
    return torch.stack(class_losses).mean()


def _lovasz_sample_indices(
    labels: torch.Tensor,
    *,
    num_classes: int,
    start_class: int,
    max_pixels: int,
) -> torch.Tensor:
    class_ids = tuple(range(start_class, num_classes))
    per_class_budget = max(1, max_pixels // max(1, len(class_ids)))
    selected: list[torch.Tensor] = []
    for class_id in class_ids:
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_indices.numel() <= per_class_budget:
            selected.append(class_indices)
            continue
        order = torch.randperm(class_indices.numel(), device=labels.device)[:per_class_budget]
        selected.append(class_indices[order])

    selected_indices = (
        torch.cat([indices for indices in selected if indices.numel() > 0])
        if selected
        else torch.empty(0, dtype=torch.long, device=labels.device)
    )
    remaining = max_pixels - int(selected_indices.numel())
    if remaining <= 0:
        return selected_indices[:max_pixels]

    available = torch.ones(labels.numel(), dtype=torch.bool, device=labels.device)
    if selected_indices.numel() > 0:
        available[selected_indices] = False
    candidate_indices = torch.nonzero(available, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return selected_indices
    if candidate_indices.numel() > remaining:
        order = torch.randperm(candidate_indices.numel(), device=labels.device)[:remaining]
        candidate_indices = candidate_indices[order]
    return torch.cat((selected_indices, candidate_indices))


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.cumsum(dim=0)
    union = total + (1.0 - sorted_foreground).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if sorted_foreground.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


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
    labeled_total = int(confusion[1:].sum().item()) if confusion.shape[0] > 1 else 0
    pixel_accuracy = float(true_positive.sum().item() / total) if total else 0.0
    labeled_pixel_accuracy = (
        float(true_positive[1:].sum().item() / labeled_total) if labeled_total else 0.0
    )
    mean_iou = sum(class_iou_values) / len(class_iou_values) if class_iou_values else 0.0
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "labeled_pixel_accuracy": labeled_pixel_accuracy,
        "class_iou": class_iou,
    }
