"""Frozen-segmenter mask-adherence scoring and cache IO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.research.spec import BenchmarkRunOptions, BenchmarkSpec
from sim_satellite.segmentation.checkpoint import (
    SegmentationCheckpoint,
    _load_segmentation_checkpoint,
)
from sim_satellite.segmentation.metrics import _confusion_matrix, _segmentation_metrics
from sim_satellite.segmentation.models import _forward_segmentation_model
from sim_satellite.segmentation.vae import _encode_vae_features

ProgressReporter = Callable[[str], None]
REAL_ADHERENCE_CACHE_VERSION = 1
REAL_ADHERENCE_CACHE_KIND = "real-mask-adherence"


class MaskAdherenceScorer:
    """Score generated and real images against the conditioning masks."""

    def __init__(
        self,
        checkpoint: SegmentationCheckpoint,
        *,
        checkpoint_path: str,
        checkpoint_sha256: str,
        cached_real_confusion: torch.Tensor | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.model = checkpoint.model
        self.vae_encoder = checkpoint.vae_encoder
        self.image_size = int(checkpoint.config["image_size"])
        self.use_context = int(checkpoint.config.get("context_size", 0)) > self.image_size
        class_count = len(OPENEARTHMAP_CLASSES)
        self.generated_confusion = torch.zeros((class_count, class_count), dtype=torch.long)
        self.real_scores_from_cache = cached_real_confusion is not None
        self.real_confusion = (
            cached_real_confusion
            if cached_real_confusion is not None
            else torch.zeros((class_count, class_count), dtype=torch.long)
        )

    def update(self, generated: torch.Tensor, real: torch.Tensor, *, labels: torch.Tensor) -> None:
        labels = labels.detach().cpu()
        self.generated_confusion += _confusion_matrix(
            self._predict(generated),
            labels,
            num_classes=len(OPENEARTHMAP_CLASSES),
        )
        if not self.real_scores_from_cache:
            self.real_confusion += _confusion_matrix(
                self._predict(real),
                labels,
                num_classes=len(OPENEARTHMAP_CLASSES),
            )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        target_size = (int(images.shape[-2]), int(images.shape[-1]))
        batch = images.detach().float().clamp(0.0, 1.0)
        if target_size != (self.image_size, self.image_size):
            batch = F.interpolate(
                batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_context:
            batch = torch.cat((batch, batch), dim=1)
        with torch.no_grad():
            predictions = _forward_segmentation_model(
                self.model,
                batch,
                vae_latents=_encode_vae_features(self.vae_encoder, batch),
            ).argmax(dim=1)
        if tuple(predictions.shape[-2:]) != target_size:
            predictions = (
                F.interpolate(
                    predictions.float().unsqueeze(1),
                    size=target_size,
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        return predictions.detach().cpu()

    def payload(self) -> dict[str, Any]:
        generated = _segmentation_metrics(self.generated_confusion, include_unlabeled_iou=False)
        real = _segmentation_metrics(self.real_confusion, include_unlabeled_iou=False)
        iou_ratio = (
            float(generated["mean_iou"]) / float(real["mean_iou"])
            if float(real["mean_iou"]) > 0.0
            else None
        )
        return {
            "enabled": True,
            "segmentation_checkpoint_path": self.checkpoint_path,
            "generated_mean_iou": float(generated["mean_iou"]),
            "generated_pixel_accuracy": float(generated["pixel_accuracy"]),
            "generated_class_iou": generated["class_iou"],
            "real_mean_iou": float(real["mean_iou"]),
            "real_pixel_accuracy": float(real["pixel_accuracy"]),
            "real_class_iou": real["class_iou"],
            "iou_ratio": iou_ratio,
            "real_scores_from_cache": self.real_scores_from_cache,
        }

    def real_cache_payload(self, *, eval_hash: str, sample_digest: str) -> dict[str, Any]:
        return {
            "cache_version": REAL_ADHERENCE_CACHE_VERSION,
            "kind": REAL_ADHERENCE_CACHE_KIND,
            "eval_hash": eval_hash,
            "sample_digest": sample_digest,
            "segmentation_checkpoint_sha256": self.checkpoint_sha256,
            "confusion": self.real_confusion.tolist(),
        }


def load_mask_adherence_scorer(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    eval_hash: str,
    sample_digest: str,
    real_adherence_cache: Mapping[str, Any] | None,
    progress: ProgressReporter | None,
) -> MaskAdherenceScorer | None:
    segmentation_checkpoint_path = (
        options.segmentation_checkpoint_path or spec.protocol.segmentation_checkpoint_path
    )
    if not options.mask_adherence or not segmentation_checkpoint_path:
        return None
    checkpoint = _load_segmentation_checkpoint(
        Path(segmentation_checkpoint_path),
        device=options.device,
    )
    checkpoint_sha256 = _file_sha256(Path(segmentation_checkpoint_path))
    cached_real_confusion = _cached_real_confusion(
        real_adherence_cache,
        eval_hash=eval_hash,
        sample_digest=sample_digest,
        checkpoint_sha256=checkpoint_sha256,
    )
    if cached_real_confusion is not None:
        _report(progress, "Reusing cached real-image mask-adherence scores")
    return MaskAdherenceScorer(
        checkpoint,
        checkpoint_path=segmentation_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        cached_real_confusion=cached_real_confusion,
    )


def disabled_mask_adherence_payload(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
) -> dict[str, Any]:
    if not spec.protocol.segmentation_checkpoint_path:
        status = "disabled: protocol has no segmentation_checkpoint_path"
    else:
        status = "disabled: run used --no-mask-adherence"
    return {"enabled": False, "status": status}


def real_adherence_cache_path(cache_dir: Path, eval_hash: str) -> Path:
    return cache_dir / f"real-adherence-{eval_hash[:16]}.json"


def load_real_adherence_cache(cache_dir: Path, eval_hash: str) -> dict[str, Any] | None:
    path = real_adherence_cache_path(cache_dir, eval_hash)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def write_real_adherence_cache(cache_dir: Path, payload: Mapping[str, Any]) -> Path:
    eval_hash = payload.get("eval_hash")
    if not isinstance(eval_hash, str) or not eval_hash:
        raise ValueError("Real-adherence cache payload is missing eval_hash")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = real_adherence_cache_path(cache_dir, eval_hash)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _cached_real_confusion(
    cache: Mapping[str, Any] | None,
    *,
    eval_hash: str,
    sample_digest: str,
    checkpoint_sha256: str,
) -> torch.Tensor | None:
    if not isinstance(cache, Mapping):
        return None
    if cache.get("cache_version") != REAL_ADHERENCE_CACHE_VERSION:
        return None
    if cache.get("kind") != REAL_ADHERENCE_CACHE_KIND:
        return None
    if cache.get("eval_hash") != eval_hash:
        return None
    if cache.get("sample_digest") != sample_digest:
        return None
    if cache.get("segmentation_checkpoint_sha256") != checkpoint_sha256:
        return None
    confusion = cache.get("confusion")
    class_count = len(OPENEARTHMAP_CLASSES)
    if not isinstance(confusion, list) or len(confusion) != class_count:
        return None
    try:
        tensor = torch.tensor(confusion, dtype=torch.long)
    except (TypeError, ValueError):
        return None
    if tuple(tensor.shape) != (class_count, class_count):
        return None
    return tensor


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)
