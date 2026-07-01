"""Evaluation and prediction for segmentation classifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.openearthmap import IMAGE_SUFFIXES, OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.tensors import image_to_uint8_tensor
from sim_satellite.segmentation.checkpoint import _load_segmentation_checkpoint
from sim_satellite.segmentation.config import (
    SegmentationEvaluationConfig,
    SegmentationEvaluationResult,
    SegmentationPredictionConfig,
    SegmentationPredictionResult,
    _validate_prediction_config,
)
from sim_satellite.segmentation.data import (
    _context_crop_box,
    _load_segmentation_batch,
    _PreparedSampleCache,
    _samples_for_split,
    _tile_windows,
)
from sim_satellite.segmentation.metrics import _confusion_matrix, _segmentation_metrics
from sim_satellite.segmentation.models import _forward_segmentation_model
from sim_satellite.segmentation.utils import ProgressReporter, _report, _should_report
from sim_satellite.segmentation.vae import VAEFeatureEncoder, _encode_vae_features


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    source: str
    split: str
    metadata: dict[str, Any]


def evaluate_segmentation_model(
    config: SegmentationEvaluationConfig,
) -> SegmentationEvaluationResult:
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device=config.device)
    model = checkpoint.model
    model_config = checkpoint.config
    vae_encoder = checkpoint.vae_encoder
    image_size = int(model_config["image_size"])
    context_size = int(model_config.get("context_size", 0))

    samples = _samples_for_split(config.manifest_path, config.split)
    if config.max_samples > 0:
        samples = samples[: config.max_samples]
    if not samples:
        raise RuntimeError(f"No samples for split `{config.split}` in {config.manifest_path}")

    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    metrics = _evaluate_samples(
        model,
        samples,
        image_size=image_size,
        context_size=context_size,
        device=torch.device(config.device),
        cache=cache,
        vae_encoder=vae_encoder,
    )
    result = SegmentationEvaluationResult(
        checkpoint_path=str(checkpoint_path),
        samples=len(samples),
        tiles=int(metrics["tiles"]),
        mean_iou=float(metrics["mean_iou"]),
        pixel_accuracy=float(metrics["pixel_accuracy"]),
        class_iou=metrics["class_iou"],
    )
    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return result


def predict_segmentation_masks(
    config: SegmentationPredictionConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationPredictionResult:
    _validate_prediction_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device=config.device)
    model = checkpoint.model
    model_config = checkpoint.config
    vae_encoder = checkpoint.vae_encoder
    image_size = int(model_config["image_size"])
    context_size = int(model_config.get("context_size", 0))
    device = torch.device(config.device)

    image_records = _prediction_image_records(config)
    if config.max_images > 0:
        image_records = image_records[: config.max_images]
    if not image_records:
        raise RuntimeError("No prediction images found")

    output_dir = Path(config.output_dir)
    mask_dir = output_dir / "masks"
    color_dir = output_dir / "color-masks"
    confidence_dir = output_dir / "confidence-masks"
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    for index, record in enumerate(image_records, start=1):
        prediction = _predict_image_mask(
            model,
            record.image_path,
            image_size=image_size,
            context_size=context_size,
            device=device,
            vae_encoder=vae_encoder,
            confidence_threshold=config.confidence_threshold,
        )
        mask_path = mask_dir / _prediction_mask_name(index, record.image_path)
        _write_label_mask(mask_path, prediction["labels"])
        color_mask_path = ""
        if config.save_color_masks:
            color_path = color_dir / _prediction_mask_name(index, record.image_path)
            _write_color_mask(color_path, prediction["labels"])
            color_mask_path = str(color_path)
        confidence_mask_path = ""
        if config.save_confidence_masks:
            confidence_path = confidence_dir / _prediction_mask_name(index, record.image_path)
            _write_confidence_mask(confidence_path, prediction["confidence"])
            confidence_mask_path = str(confidence_path)

        payload = {
            **record.metadata,
            "image": str(record.image_path),
            "mask": str(mask_path),
            "color_mask": color_mask_path,
            "confidence_mask": confidence_mask_path,
            "source": record.source,
            "split": record.split,
            "prediction_source": "vae-fused-segmentation-classifier",
            "checkpoint": str(checkpoint_path),
            "confidence_threshold": config.confidence_threshold,
            "mean_confidence": prediction["mean_confidence"],
            "high_confidence_fraction": prediction["high_confidence_fraction"],
            "class_counts": prediction["class_counts"],
        }
        records.append(payload)
        if _should_report(index, len(image_records)):
            _report(progress, f"Predicted {index}/{len(image_records)} masks")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    return SegmentationPredictionResult(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        images=len(records),
    )


def _evaluate_samples(
    model: nn.Module,
    samples: Sequence[SatelliteSample],
    *,
    image_size: int,
    device: torch.device,
    cache: _PreparedSampleCache,
    vae_encoder: VAEFeatureEncoder,
    context_size: int = 0,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), len(OPENEARTHMAP_CLASSES)),
        dtype=torch.long,
    )
    tiles = 0
    with torch.no_grad():
        for sample in samples:
            prepared = cache.get(sample)
            for crop_box in _tile_windows(prepared.size, image_size):
                images, labels = _load_segmentation_batch(
                    (sample,),
                    (crop_box,),
                    cache=cache,
                    image_size=image_size,
                    context_size=context_size,
                    device=device,
                )
                predictions = (
                    _forward_segmentation_model(
                        model,
                        images,
                        vae_latents=_encode_vae_features(vae_encoder, images),
                    )
                    .argmax(dim=1)
                    .cpu()
                )
                confusion += _confusion_matrix(
                    predictions,
                    labels.cpu(),
                    num_classes=len(OPENEARTHMAP_CLASSES),
                )
                tiles += 1
    metrics = _segmentation_metrics(confusion, include_unlabeled_iou=False)
    metrics["tiles"] = tiles
    return metrics


def _predict_image_mask(
    model: nn.Module,
    image_path: Path,
    *,
    image_size: int,
    device: torch.device,
    vae_encoder: VAEFeatureEncoder,
    context_size: int = 0,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        output = torch.zeros((image.height, image.width), dtype=torch.uint8)
        confidence_output = torch.zeros((image.height, image.width), dtype=torch.float32)
        confidence_sum = 0.0
        high_confidence_count = 0
        pixel_count = 0
        for crop_box in _tile_windows(image.size, image_size):
            crop = image.crop(crop_box)
            tensor = image_to_uint8_tensor(
                crop,
                torch=torch,
                size=(image_size, image_size),
            )
            if context_size > image_size:
                context_crop = image.crop(_context_crop_box(image.size, crop_box, context_size))
                context_tensor = image_to_uint8_tensor(
                    context_crop,
                    torch=torch,
                    size=(image_size, image_size),
                )
                tensor = torch.cat((tensor, context_tensor), dim=2)
            batch = tensor.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                probabilities = (
                    _forward_segmentation_model(
                        model,
                        batch,
                        vae_latents=_encode_vae_features(vae_encoder, batch),
                    )
                    .softmax(dim=1)
                    .cpu()
                )
            confidence, labels = probabilities.max(dim=1)
            left, top, right, bottom = crop_box
            target_size = (right - left, bottom - top)
            labels = (
                F.interpolate(
                    labels.float().unsqueeze(1),
                    size=(target_size[1], target_size[0]),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .to(torch.uint8)
            )
            confidence = (
                F.interpolate(
                    confidence.unsqueeze(1),
                    size=(target_size[1], target_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            high_confidence = confidence >= confidence_threshold
            if confidence_threshold > 0.0:
                labels = torch.where(high_confidence, labels, torch.zeros_like(labels))
            output[top:bottom, left:right] = labels
            confidence_output[top:bottom, left:right] = confidence
            confidence_sum += float(confidence.sum().item())
            high_confidence_count += int(high_confidence.sum().item())
            pixel_count += int(confidence.numel())
    counts = torch.bincount(output.long().flatten(), minlength=len(OPENEARTHMAP_CLASSES))
    return {
        "labels": output,
        "confidence": confidence_output.clamp(0.0, 1.0),
        "mean_confidence": confidence_sum / pixel_count if pixel_count else 0.0,
        "high_confidence_fraction": high_confidence_count / pixel_count if pixel_count else 0.0,
        "class_counts": {
            class_name: int(counts[index].item())
            for index, class_name in enumerate(OPENEARTHMAP_CLASSES)
            if counts[index].item()
        },
    }


def _prediction_image_records(config: SegmentationPredictionConfig) -> tuple[ImageRecord, ...]:
    if config.images_manifest_path:
        return _read_image_records(config.images_manifest_path)
    image_dir = Path(config.images_dir)
    return tuple(
        ImageRecord(
            image_path=path,
            source="image-directory",
            split="predict",
            metadata={},
        )
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_image_records(path: str | Path) -> tuple[ImageRecord, ...]:
    manifest_path = Path(path)
    root = manifest_path.parent
    records: list[ImageRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{manifest_path}:{line_number}: image record must be an object")
            image = payload.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(f"{manifest_path}:{line_number}: image is required")
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = root / image_path
            metadata = dict(payload)
            records.append(
                ImageRecord(
                    image_path=image_path,
                    source=str(payload.get("source", "image-manifest")),
                    split=str(payload.get("split", "predict")),
                    metadata=metadata,
                )
            )
    return tuple(records)


def _write_label_mask(path: Path, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = labels.shape
    values = labels.to(device="cpu", dtype=torch.uint8).contiguous()
    Image.frombytes("L", (width, height), values.numpy().tobytes()).save(path)


def _write_color_mask(path: Path, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = labels.shape
    image = Image.frombytes(
        "L",
        (width, height),
        labels.to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes(),
    )
    palette: list[int] = []
    for class_name in OPENEARTHMAP_CLASSES:
        palette.extend(OPENEARTHMAP_RGB[class_name])
    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    image.convert("RGB").save(path)


def _write_confidence_mask(path: Path, confidence: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = confidence.shape
    values = (
        confidence.to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    Image.frombytes("L", (width, height), values.numpy().tobytes()).save(path)


def _prediction_mask_name(index: int, image_path: Path) -> str:
    return f"{index:06d}-{image_path.stem}.png"
