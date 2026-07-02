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
    _validate_evaluation_config,
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
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationEvaluationResult:
    _validate_evaluation_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device=config.device)
    model = checkpoint.model
    model_config = checkpoint.config
    vae_encoder = checkpoint.vae_encoder
    image_size = int(model_config["image_size"])
    context_size = int(model_config.get("context_size", 0))
    tile_stride = config.tile_stride

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
        tile_stride=tile_stride,
        tile_scales=config.tile_scales,
        device=torch.device(config.device),
        cache=cache,
        vae_encoder=vae_encoder,
        output_class_names=checkpoint.classes,
        class_logit_biases=config.class_logit_biases,
        feather_blending=config.feather_blending,
        confidence_threshold=config.confidence_threshold,
        progress=progress,
    )
    result = SegmentationEvaluationResult(
        checkpoint_path=str(checkpoint_path),
        class_logit_biases=dict(config.class_logit_biases),
        confidence_threshold=config.confidence_threshold,
        feather_blending=config.feather_blending,
        samples=len(samples),
        tiles=int(metrics["tiles"]),
        mean_confidence=float(metrics["mean_confidence"]),
        mean_iou=float(metrics["mean_iou"]),
        pixel_accuracy=float(metrics["pixel_accuracy"]),
        labeled_pixel_accuracy=float(metrics["labeled_pixel_accuracy"]),
        selected_mean_confidence=float(metrics["selected_mean_confidence"]),
        selected_pixels=int(metrics["selected_pixels"]),
        selected_labeled_pixels=int(metrics["selected_labeled_pixels"]),
        selected_pixel_coverage=float(metrics["selected_pixel_coverage"]),
        selected_labeled_pixel_coverage=float(metrics["selected_labeled_pixel_coverage"]),
        selected_mean_iou=float(metrics["selected_mean_iou"]),
        selected_pixel_accuracy=float(metrics["selected_pixel_accuracy"]),
        selected_labeled_pixel_accuracy=float(metrics["selected_labeled_pixel_accuracy"]),
        selected_class_iou=metrics["selected_class_iou"],
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
    tile_stride = config.tile_stride
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
            tile_stride=tile_stride,
            tile_scales=config.tile_scales,
            device=device,
            vae_encoder=vae_encoder,
            output_class_names=checkpoint.classes,
            class_logit_biases=config.class_logit_biases,
            feather_blending=config.feather_blending,
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
            "prediction_source": "segmentation-classifier",
            "checkpoint": str(checkpoint_path),
            "class_logit_biases": dict(config.class_logit_biases),
            "confidence_threshold": config.confidence_threshold,
            "feather_blending": config.feather_blending,
            "mean_confidence": prediction["mean_confidence"],
            "high_confidence_pixels": prediction["high_confidence_pixels"],
            "low_confidence_pixels": prediction["low_confidence_pixels"],
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
        class_logit_biases=dict(config.class_logit_biases),
        confidence_threshold=config.confidence_threshold,
        feather_blending=config.feather_blending,
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
    vae_encoder: VAEFeatureEncoder | None,
    context_size: int = 0,
    tile_stride: int = 0,
    tile_scales: Sequence[float] = (),
    output_class_names: Sequence[str] = OPENEARTHMAP_CLASSES,
    class_logit_biases: Mapping[str, float] | None = None,
    feather_blending: bool = False,
    confidence_threshold: float = 0.0,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(
        (len(output_class_names), len(output_class_names)),
        dtype=torch.long,
    )
    selected_confusion = torch.zeros_like(confusion)
    tiles = 0
    total_pixels = 0
    labeled_pixels = 0
    selected_pixels = 0
    selected_labeled_pixels = 0
    confidence_sum = 0.0
    selected_confidence_sum = 0.0
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            prepared = cache.get(sample)
            logits, sample_tiles = _predict_prepared_sample_logits(
                model,
                sample,
                image_size=image_size,
                context_size=context_size,
                tile_stride=tile_stride,
                tile_scales=tile_scales,
                device=device,
                cache=cache,
                vae_encoder=vae_encoder,
                output_class_names=output_class_names,
                feather_blending=feather_blending,
            )
            logits = _apply_class_logit_biases(
                logits,
                class_logit_biases,
                class_names=output_class_names,
            )
            probabilities = logits.softmax(dim=0).cpu()
            confidence, predictions = probabilities.max(dim=0)
            labels = prepared.mask.long().cpu()
            selected_mask = confidence >= confidence_threshold
            confusion += _confusion_matrix(
                predictions,
                labels,
                num_classes=len(output_class_names),
            )
            selected_confusion += _confusion_matrix(
                predictions[selected_mask],
                labels[selected_mask],
                num_classes=len(output_class_names),
            )
            sample_labeled = labels > 0
            total_pixels += int(labels.numel())
            labeled_pixels += int(sample_labeled.sum().item())
            selected_pixels += int(selected_mask.sum().item())
            selected_labeled_pixels += int((selected_mask & sample_labeled).sum().item())
            confidence_sum += float(confidence.sum().item())
            selected_confidence_sum += float(confidence[selected_mask].sum().item())
            tiles += sample_tiles
            if _should_report(index, len(samples)):
                _report(progress, f"Evaluated {index}/{len(samples)} segmentation samples")
    metrics = _segmentation_metrics(confusion, include_unlabeled_iou=False)
    selected_metrics = _segmentation_metrics(selected_confusion, include_unlabeled_iou=False)
    metrics["tiles"] = tiles
    metrics["mean_confidence"] = confidence_sum / total_pixels if total_pixels else 0.0
    metrics["selected_mean_confidence"] = (
        selected_confidence_sum / selected_pixels if selected_pixels else 0.0
    )
    metrics["selected_pixels"] = selected_pixels
    metrics["selected_labeled_pixels"] = selected_labeled_pixels
    metrics["selected_pixel_coverage"] = selected_pixels / total_pixels if total_pixels else 0.0
    metrics["selected_labeled_pixel_coverage"] = (
        selected_labeled_pixels / labeled_pixels if labeled_pixels else 0.0
    )
    metrics["selected_mean_iou"] = selected_metrics["mean_iou"]
    metrics["selected_pixel_accuracy"] = selected_metrics["pixel_accuracy"]
    metrics["selected_labeled_pixel_accuracy"] = selected_metrics["labeled_pixel_accuracy"]
    metrics["selected_class_iou"] = selected_metrics["class_iou"]
    return metrics


def _predict_image_mask(
    model: nn.Module,
    image_path: Path,
    *,
    image_size: int,
    device: torch.device,
    vae_encoder: VAEFeatureEncoder | None,
    context_size: int = 0,
    tile_stride: int = 0,
    tile_scales: Sequence[float] = (),
    output_class_names: Sequence[str] = OPENEARTHMAP_CLASSES,
    class_logit_biases: Mapping[str, float] | None = None,
    feather_blending: bool = False,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        logits, _tiles = _predict_image_logits(
            model,
            image,
            image_size=image_size,
            context_size=context_size,
            tile_stride=tile_stride,
            tile_scales=tile_scales,
            device=device,
            vae_encoder=vae_encoder,
            output_class_names=output_class_names,
            feather_blending=feather_blending,
        )
        logits = _apply_class_logit_biases(
            logits,
            class_logit_biases,
            class_names=output_class_names,
        )
        probabilities = logits.softmax(dim=0).cpu()
        confidence, labels = probabilities.max(dim=0)
        output = labels.to(torch.uint8)
        high_confidence = confidence >= confidence_threshold
        if confidence_threshold > 0.0:
            output = torch.where(high_confidence, output, torch.zeros_like(output))
        confidence_sum = float(confidence.sum().item())
        high_confidence_count = int(high_confidence.sum().item())
        pixel_count = int(confidence.numel())
    counts = torch.bincount(output.long().flatten(), minlength=len(output_class_names))
    return {
        "labels": output,
        "confidence": confidence.clamp(0.0, 1.0),
        "mean_confidence": confidence_sum / pixel_count if pixel_count else 0.0,
        "high_confidence_pixels": high_confidence_count,
        "low_confidence_pixels": pixel_count - high_confidence_count,
        "high_confidence_fraction": high_confidence_count / pixel_count if pixel_count else 0.0,
        "class_counts": {
            class_name: int(counts[index].item())
            for index, class_name in enumerate(output_class_names)
            if counts[index].item()
        },
    }


def _apply_class_logit_biases(
    logits: torch.Tensor,
    class_logit_biases: Mapping[str, float] | None,
    *,
    class_names: Sequence[str] = OPENEARTHMAP_CLASSES,
) -> torch.Tensor:
    if not class_logit_biases:
        return logits
    bias = torch.zeros((len(class_names),), dtype=logits.dtype, device=logits.device)
    class_names = tuple(class_names)
    for class_name, value in class_logit_biases.items():
        bias[class_names.index(class_name)] = float(value)
    return logits + bias.view(-1, 1, 1)


def _predict_prepared_sample_logits(
    model: nn.Module,
    sample: SatelliteSample,
    *,
    image_size: int,
    context_size: int,
    tile_stride: int,
    tile_scales: Sequence[float],
    device: torch.device,
    cache: _PreparedSampleCache,
    vae_encoder: VAEFeatureEncoder | None,
    output_class_names: Sequence[str],
    feather_blending: bool,
) -> tuple[torch.Tensor, int]:
    prepared = cache.get(sample)
    height, width = prepared.mask.shape
    logit_sum = torch.zeros(
        (len(output_class_names), height, width),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    tiles = 0
    for tile_size, stride in _scaled_tile_windows_config(
        image_size=image_size,
        tile_stride=tile_stride,
        tile_scales=tile_scales,
    ):
        for crop_box in _tile_windows(prepared.size, tile_size, stride=stride):
            images, _labels = _load_segmentation_batch(
                (sample,),
                (crop_box,),
                cache=cache,
                image_size=image_size,
                context_size=context_size,
                device=device,
            )
            logits = _forward_segmentation_logits(model, images, vae_encoder=vae_encoder)
            resized = _resized_tile_logits(logits, crop_box)
            _accumulate_resized_tile_logits(
                logit_sum,
                counts,
                resized,
                crop_box,
                tile_size=tile_size,
                tile_stride=stride,
                feather_blending=feather_blending,
            )
            tiles += 1
    if not bool((counts > 0).all().item()):
        raise RuntimeError(f"Segmentation tiling left uncovered pixels for {sample.image_path}")
    return logit_sum / counts, tiles


def _predict_image_logits(
    model: nn.Module,
    image: Image.Image,
    *,
    image_size: int,
    context_size: int,
    tile_stride: int,
    tile_scales: Sequence[float],
    device: torch.device,
    vae_encoder: VAEFeatureEncoder | None,
    output_class_names: Sequence[str],
    feather_blending: bool,
) -> tuple[torch.Tensor, int]:
    logit_sum = torch.zeros(
        (len(output_class_names), image.height, image.width),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((1, image.height, image.width), dtype=torch.float32, device=device)
    tiles = 0
    for tile_size, stride in _scaled_tile_windows_config(
        image_size=image_size,
        tile_stride=tile_stride,
        tile_scales=tile_scales,
    ):
        for crop_box in _tile_windows(image.size, tile_size, stride=stride):
            batch = _image_crop_batch(
                image,
                crop_box,
                image_size=image_size,
                context_size=context_size,
                device=device,
            )
            logits = _forward_segmentation_logits(model, batch, vae_encoder=vae_encoder)
            resized = _resized_tile_logits(logits, crop_box)
            _accumulate_resized_tile_logits(
                logit_sum,
                counts,
                resized,
                crop_box,
                tile_size=tile_size,
                tile_stride=stride,
                feather_blending=feather_blending,
            )
            tiles += 1
    if not bool((counts > 0).all().item()):
        raise RuntimeError("Segmentation tiling left uncovered pixels for prediction image")
    return logit_sum / counts, tiles


def _scaled_tile_windows_config(
    *,
    image_size: int,
    tile_stride: int,
    tile_scales: Sequence[float],
) -> tuple[tuple[int, int], ...]:
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if tile_stride < 0:
        raise ValueError("tile_stride must be non-negative")

    scales = tile_scales or (1.0,)
    configs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for scale in scales:
        if scale <= 0.0:
            raise ValueError("tile scales must be positive")
        tile_size = max(1, int(round(image_size * scale)))
        stride = 0
        if tile_stride > 0:
            stride = max(1, int(round(tile_stride * tile_size / image_size)))
        key = (tile_size, stride)
        if key in seen:
            continue
        seen.add(key)
        configs.append(key)
    return tuple(configs)


def _forward_segmentation_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    vae_encoder: VAEFeatureEncoder | None,
) -> torch.Tensor:
    return _forward_segmentation_model(
        model,
        images,
        vae_latents=_encode_vae_features(vae_encoder, images) if vae_encoder else None,
    )


def _image_crop_batch(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    *,
    image_size: int,
    context_size: int,
    device: torch.device,
) -> torch.Tensor:
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
    return tensor.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def _resized_tile_logits(
    logits: torch.Tensor,
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    left, top, right, bottom = crop_box
    return F.interpolate(
        logits,
        size=(bottom - top, right - left),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _accumulate_resized_tile_logits(
    logit_sum: torch.Tensor,
    counts: torch.Tensor,
    logits: torch.Tensor,
    crop_box: tuple[int, int, int, int],
    *,
    tile_size: int,
    tile_stride: int,
    feather_blending: bool,
) -> None:
    left, top, right, bottom = crop_box
    if feather_blending:
        weight = _tile_blending_weight(
            height=bottom - top,
            width=right - left,
            tile_size=tile_size,
            tile_stride=tile_stride,
            device=logits.device,
            dtype=logits.dtype,
        )
        logit_sum[:, top:bottom, left:right] += logits * weight
        counts[:, top:bottom, left:right] += weight
        return
    logit_sum[:, top:bottom, left:right] += logits
    counts[:, top:bottom, left:right] += 1.0


def _tile_blending_weight(
    *,
    height: int,
    width: int,
    tile_size: int,
    tile_stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("tile dimensions must be positive")
    if tile_stride <= 0 or tile_size <= tile_stride:
        return torch.ones((1, height, width), dtype=dtype, device=device)

    overlap = tile_size - tile_stride
    edge = torch.ones(tile_size, dtype=dtype, device=device)
    edge[tile_stride:] = 1.0 - (
        torch.arange(1, overlap + 1, dtype=dtype, device=device) / (overlap + 1)
    )
    horizontal = edge * edge.flip(0)
    weight = (horizontal[:, None] * horizontal[None, :]).unsqueeze(0)
    if weight.shape[-2:] == (height, width):
        return weight
    return F.interpolate(
        weight.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


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
