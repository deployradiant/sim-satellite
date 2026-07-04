"""Evaluate production-segmentation pixels with legacy VAE fill.

This is analysis tooling, not a production segmentation path. It keeps the old
VAE-fused segmenter isolated inside the script so the package runtime can stay
on the production OpenEarth segmenter.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.modal_app import (
    MODAL_DATA_SETUP_TIMEOUT,
    MODAL_INFERENCE_GPU,
    _ensure_modal_training_data,
    _modal_artifact_path,
    _modal_data_path,
    app,
    artifact_volume,
    image,
    modal_volumes,
)
from sim_satellite.segmentation.checkpoint import _load_segmentation_checkpoint
from sim_satellite.segmentation.data import (
    _load_segmentation_batch,
    _PreparedSampleCache,
    _samples_for_split,
    _tile_windows,
)
from sim_satellite.segmentation.inference import (
    _accumulate_resized_tile_logits,
    _apply_class_logit_biases,
    _predict_prepared_sample_logits,
    _resized_tile_logits,
    _scaled_tile_windows_config,
)
from sim_satellite.segmentation.metrics import _confusion_matrix, _segmentation_metrics
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig

PRODUCTION_CHECKPOINT_PATH = (
    "artifacts/segmentation-openearthmap-smp-unet-efficientnet-b4-scse-official-xbd-full-polish-v1/"
    "smp-unet-efficientnet-b4-scse-official-xbd-full-polish-s2500.pt"
)
LEGACY_VAE_SEGMENTATION_CHECKPOINT_PATH = (
    "artifacts/segmentation-study-2026-06-28-200k/"
    "deeplab_unet_vae200k_fused_noaug_lr3e-4_alldata_s200000_val256.pt"
)
OUTPUT_PATH = (
    "artifacts/modal-metrics/hybrid-production-vae-fill-official-val500-threshold-sweep.json"
)
PRODUCTION_CLASS_LOGIT_BIASES = {
    "agriculture land": -0.23,
    "bareland": -0.9,
    "building": 0.2,
    "developed space": -0.02,
    "rangeland": 0.05,
    "road": -0.2,
    "tree": -0.02,
    "water": -0.8,
}
DEFAULT_THRESHOLDS = (
    0.0,
    0.5,
    0.6,
    0.7,
    0.74,
    0.8,
    0.85,
    0.88,
    0.9,
    0.92,
    0.95,
    0.96,
    0.98,
)


@dataclass(frozen=True)
class LegacyVAEFeatureEncoder:
    model: SatelliteVAE
    latent_channels: int
    mean: torch.Tensor | None
    std: torch.Tensor | None
    normalize: bool


@dataclass(frozen=True)
class LegacyVAESegmentationCheckpoint:
    model: nn.Module
    config: dict[str, Any]
    classes: tuple[str, ...]
    vae_encoder: LegacyVAEFeatureEncoder


@app.function(
    image=image,
    timeout=MODAL_DATA_SETUP_TIMEOUT,
    gpu=MODAL_INFERENCE_GPU,
    volumes=modal_volumes,
)
def hybrid_production_vae_fill_sweep_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_training_data(config["manifest_path"])
    artifact_volume.reload()

    device = torch.device(config["device"])
    production = _load_segmentation_checkpoint(
        Path(config["production_checkpoint_path"]),
        device=config["device"],
    )
    legacy_vae = _load_legacy_vae_segmentation_checkpoint(
        Path(config["vae_checkpoint_path"]),
        device=device,
    )
    if production.classes != legacy_vae.classes:
        raise RuntimeError("Production and VAE checkpoints have different class lists")

    class_count = len(production.classes)
    thresholds = tuple(float(value) for value in config["thresholds"])
    samples = _samples_for_split(config["manifest_path"], config["split"])
    if config["max_samples"] > 0:
        samples = samples[: int(config["max_samples"])]
    if not samples:
        raise RuntimeError(f"No samples for split {config['split']!r}")

    cache = _PreparedSampleCache(max_samples=int(config["cache_samples"]))
    production_confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    vae_confusion = torch.zeros_like(production_confusion)
    hybrid_confusions = {
        threshold: torch.zeros_like(production_confusion) for threshold in thresholds
    }
    production_selected_confusions = {
        threshold: torch.zeros_like(production_confusion) for threshold in thresholds
    }
    vae_fill_confusions = {
        threshold: torch.zeros_like(production_confusion) for threshold in thresholds
    }
    production_selected_pixels = {threshold: 0 for threshold in thresholds}
    production_selected_labeled_pixels = {threshold: 0 for threshold in thresholds}
    vae_fill_pixels = {threshold: 0 for threshold in thresholds}
    vae_fill_labeled_pixels = {threshold: 0 for threshold in thresholds}
    production_selected_confidence_sums = {threshold: 0.0 for threshold in thresholds}

    total_pixels = 0
    labeled_pixels = 0
    production_tiles = 0
    vae_tiles = 0
    production_confidence_sum = 0.0
    vae_confidence_sum = 0.0

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            prepared = cache.get(sample)
            labels = prepared.mask.long().cpu()
            sample_labeled = labels > 0

            production_logits, sample_production_tiles = _predict_prepared_sample_logits(
                production.model,
                sample,
                image_size=int(production.config["image_size"]),
                context_size=int(production.config.get("context_size", 0)),
                tile_stride=int(config["production_tile_stride"]),
                tile_scales=tuple(float(value) for value in config["production_tile_scales"]),
                device=device,
                cache=cache,
                output_class_names=production.classes,
                feather_blending=bool(config["production_feather_blending"]),
            )
            production_logits = _apply_class_logit_biases(
                production_logits,
                config["production_class_logit_biases"],
                class_names=production.classes,
            )
            production_probabilities = production_logits.softmax(dim=0).cpu()
            production_confidence, production_predictions = production_probabilities.max(dim=0)

            vae_logits, sample_vae_tiles = _predict_legacy_vae_prepared_sample_logits(
                legacy_vae,
                sample,
                tile_stride=int(config["vae_tile_stride"]),
                tile_scales=tuple(float(value) for value in config["vae_tile_scales"]),
                device=device,
                cache=cache,
                feather_blending=bool(config["vae_feather_blending"]),
            )
            vae_logits = _apply_class_logit_biases(
                vae_logits,
                config["vae_class_logit_biases"],
                class_names=legacy_vae.classes,
            )
            vae_probabilities = vae_logits.softmax(dim=0).cpu()
            vae_confidence, vae_predictions = vae_probabilities.max(dim=0)

            production_confusion += _confusion_matrix(
                production_predictions,
                labels,
                num_classes=class_count,
            )
            vae_confusion += _confusion_matrix(vae_predictions, labels, num_classes=class_count)
            total_pixels += int(labels.numel())
            labeled_pixels += int(sample_labeled.sum().item())
            production_confidence_sum += float(production_confidence.sum().item())
            vae_confidence_sum += float(vae_confidence.sum().item())
            production_tiles += sample_production_tiles
            vae_tiles += sample_vae_tiles

            for threshold in thresholds:
                production_mask = production_confidence >= threshold
                fill_mask = ~production_mask
                hybrid_predictions = torch.where(
                    production_mask,
                    production_predictions,
                    vae_predictions,
                )
                hybrid_confusions[threshold] += _confusion_matrix(
                    hybrid_predictions,
                    labels,
                    num_classes=class_count,
                )
                production_selected_confusions[threshold] += _confusion_matrix(
                    production_predictions[production_mask],
                    labels[production_mask],
                    num_classes=class_count,
                )
                vae_fill_confusions[threshold] += _confusion_matrix(
                    vae_predictions[fill_mask],
                    labels[fill_mask],
                    num_classes=class_count,
                )
                production_selected_pixels[threshold] += int(production_mask.sum().item())
                production_selected_labeled_pixels[threshold] += int(
                    (production_mask & sample_labeled).sum().item()
                )
                vae_fill_pixels[threshold] += int(fill_mask.sum().item())
                vae_fill_labeled_pixels[threshold] += int((fill_mask & sample_labeled).sum().item())
                production_selected_confidence_sums[threshold] += float(
                    production_confidence[production_mask].sum().item()
                )

            if index == 1 or index % 10 == 0 or index == len(samples):
                print(
                    json.dumps(
                        {
                            "event": "hybrid_production_vae_progress",
                            "sample": index,
                            "total": len(samples),
                        },
                        sort_keys=True,
                    )
                )

    result = {
        "production_checkpoint_path": config["production_checkpoint_path"],
        "vae_checkpoint_path": config["vae_checkpoint_path"],
        "manifest_path": config["manifest_path"],
        "split": config["split"],
        "samples": len(samples),
        "production_tiles": production_tiles,
        "vae_tiles": vae_tiles,
        "production_mean_confidence": (
            production_confidence_sum / total_pixels if total_pixels else 0.0
        ),
        "vae_mean_confidence": vae_confidence_sum / total_pixels if total_pixels else 0.0,
        "production_dense": _segmentation_metrics(
            production_confusion,
            include_unlabeled_iou=False,
        ),
        "vae_dense": _segmentation_metrics(vae_confusion, include_unlabeled_iou=False),
        "hybrid": _hybrid_rows(
            thresholds,
            hybrid_confusions=hybrid_confusions,
            production_selected_confusions=production_selected_confusions,
            vae_fill_confusions=vae_fill_confusions,
            production_selected_pixels=production_selected_pixels,
            production_selected_labeled_pixels=production_selected_labeled_pixels,
            vae_fill_pixels=vae_fill_pixels,
            vae_fill_labeled_pixels=vae_fill_labeled_pixels,
            production_selected_confidence_sums=production_selected_confidence_sums,
            total_pixels=total_pixels,
            labeled_pixels=labeled_pixels,
        ),
        "execution_target": "modal",
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    return result


def _hybrid_rows(
    thresholds: Sequence[float],
    *,
    hybrid_confusions: Mapping[float, torch.Tensor],
    production_selected_confusions: Mapping[float, torch.Tensor],
    vae_fill_confusions: Mapping[float, torch.Tensor],
    production_selected_pixels: Mapping[float, int],
    production_selected_labeled_pixels: Mapping[float, int],
    vae_fill_pixels: Mapping[float, int],
    vae_fill_labeled_pixels: Mapping[float, int],
    production_selected_confidence_sums: Mapping[float, float],
    total_pixels: int,
    labeled_pixels: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hybrid = _segmentation_metrics(
            hybrid_confusions[threshold],
            include_unlabeled_iou=False,
        )
        production_selected = _segmentation_metrics(
            production_selected_confusions[threshold],
            include_unlabeled_iou=False,
        )
        vae_fill = _segmentation_metrics(
            vae_fill_confusions[threshold],
            include_unlabeled_iou=False,
        )
        selected_pixels = production_selected_pixels[threshold]
        fill_pixels = vae_fill_pixels[threshold]
        rows.append(
            {
                "production_confidence_threshold": threshold,
                "mean_iou": hybrid["mean_iou"],
                "pixel_accuracy": hybrid["pixel_accuracy"],
                "labeled_pixel_accuracy": hybrid["labeled_pixel_accuracy"],
                "class_iou": hybrid["class_iou"],
                "production_selected_mean_iou": production_selected["mean_iou"],
                "production_selected_labeled_pixel_accuracy": production_selected[
                    "labeled_pixel_accuracy"
                ],
                "production_selected_pixel_coverage": (
                    selected_pixels / total_pixels if total_pixels else 0.0
                ),
                "production_selected_labeled_pixel_coverage": (
                    production_selected_labeled_pixels[threshold] / labeled_pixels
                    if labeled_pixels
                    else 0.0
                ),
                "production_selected_mean_confidence": (
                    production_selected_confidence_sums[threshold] / selected_pixels
                    if selected_pixels
                    else 0.0
                ),
                "vae_fill_mean_iou": vae_fill["mean_iou"],
                "vae_fill_labeled_pixel_accuracy": vae_fill["labeled_pixel_accuracy"],
                "vae_fill_pixel_coverage": fill_pixels / total_pixels if total_pixels else 0.0,
                "vae_fill_labeled_pixel_coverage": (
                    vae_fill_labeled_pixels[threshold] / labeled_pixels
                    if labeled_pixels
                    else 0.0
                ),
            }
        )
    return rows


def _load_legacy_vae_segmentation_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> LegacyVAESegmentationCheckpoint:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Legacy checkpoint must be a mapping: {checkpoint_path}")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"Legacy checkpoint config must be a mapping: {checkpoint_path}")
    classes = checkpoint.get("classes")
    if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
        raise RuntimeError(f"Legacy checkpoint classes must be a sequence: {checkpoint_path}")
    classes = tuple(str(class_name) for class_name in classes)
    if classes != OPENEARTHMAP_CLASSES:
        raise RuntimeError(
            f"Legacy checkpoint classes do not match OpenEarthMap: {checkpoint_path}"
        )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Legacy checkpoint is missing model_state: {checkpoint_path}")
    model_name = checkpoint.get("model")
    if model_name != "deeplab-unet-v1":
        raise RuntimeError(f"Expected legacy deeplab-unet-v1 checkpoint: {checkpoint_path}")

    vae_encoder = _load_legacy_vae_feature_encoder(
        dict(config),
        device=device,
        segmentation_checkpoint_path=checkpoint_path,
    )
    latent_channels = int(config.get("vae_latent_channels", vae_encoder.latent_channels))
    if latent_channels != vae_encoder.latent_channels:
        raise RuntimeError("Legacy VAE segmentation checkpoint latent channel mismatch")

    model = LegacyVAEASPPUNetSegmentationModel(
        num_classes=len(classes),
        input_channels=3,
        vae_latent_channels=latent_channels,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return LegacyVAESegmentationCheckpoint(
        model=model,
        config=dict(config),
        classes=classes,
        vae_encoder=vae_encoder,
    )


def _load_legacy_vae_feature_encoder(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    segmentation_checkpoint_path: Path,
) -> LegacyVAEFeatureEncoder:
    checkpoint_value = config.get("vae_checkpoint_path")
    if not checkpoint_value:
        raise RuntimeError("Legacy VAE segmentation checkpoint is missing vae_checkpoint_path")
    vae_checkpoint_path = _resolve_legacy_vae_checkpoint_path(
        str(checkpoint_value),
        segmentation_checkpoint_path=segmentation_checkpoint_path,
    )
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"VAE checkpoint must be a mapping: {vae_checkpoint_path}")
    vae_config_values = checkpoint.get("vae_config")
    model_state = checkpoint.get("model_state")
    if not isinstance(vae_config_values, Mapping) or not isinstance(model_state, Mapping):
        raise RuntimeError(f"VAE checkpoint is missing config/state: {vae_checkpoint_path}")

    vae_config = SatelliteVAEConfig(**dict(vae_config_values))
    vae_model = SatelliteVAE(vae_config).to(device)
    vae_model.load_state_dict(model_state)
    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad_(False)

    latent_channels = int(vae_config.latent_channels)
    normalize = bool(config.get("vae_normalize_latents", True))
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    if normalize:
        latent_stats = checkpoint.get("latent_stats")
        if not isinstance(latent_stats, Mapping):
            raise RuntimeError(f"VAE checkpoint is missing latent_stats: {vae_checkpoint_path}")
        mean = _legacy_latent_stat_tensor(
            latent_stats,
            "mean",
            latent_channels=latent_channels,
            device=device,
            checkpoint_path=vae_checkpoint_path,
        )
        std = _legacy_latent_stat_tensor(
            latent_stats,
            "std",
            latent_channels=latent_channels,
            device=device,
            checkpoint_path=vae_checkpoint_path,
        ).clamp_min(1e-6)
    return LegacyVAEFeatureEncoder(
        model=vae_model,
        latent_channels=latent_channels,
        mean=mean,
        std=std,
        normalize=normalize,
    )


def _legacy_latent_stat_tensor(
    latent_stats: Mapping[str, Any],
    name: str,
    *,
    latent_channels: int,
    device: torch.device,
    checkpoint_path: Path,
) -> torch.Tensor:
    values = latent_stats.get(name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(
            f"VAE checkpoint latent_stats.{name} must be a sequence: {checkpoint_path}"
        )
    if len(values) != latent_channels:
        raise RuntimeError(f"VAE checkpoint latent_stats.{name} has wrong size: {checkpoint_path}")
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1, 1)


def _resolve_legacy_vae_checkpoint_path(
    value: str,
    *,
    segmentation_checkpoint_path: Path,
) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    artifact_root_path = _artifact_root_relative_path(
        path,
        checkpoint_path=segmentation_checkpoint_path,
    )
    if artifact_root_path is not None:
        return artifact_root_path
    return segmentation_checkpoint_path.parent / path


def _artifact_root_relative_path(path: Path, *, checkpoint_path: Path) -> Path | None:
    if not path.parts or path.parts[0] != "artifacts":
        return None
    checkpoint_parts = checkpoint_path.resolve().parts
    try:
        artifact_index = checkpoint_parts.index("artifacts")
    except ValueError:
        return None
    artifact_parent = Path(*checkpoint_parts[:artifact_index])
    return artifact_parent / path


def _predict_legacy_vae_prepared_sample_logits(
    checkpoint: LegacyVAESegmentationCheckpoint,
    sample: Any,
    *,
    tile_stride: int,
    tile_scales: Sequence[float],
    device: torch.device,
    cache: _PreparedSampleCache,
    feather_blending: bool,
) -> tuple[torch.Tensor, int]:
    prepared = cache.get(sample)
    height, width = prepared.mask.shape
    logit_sum = torch.zeros(
        (len(checkpoint.classes), height, width),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    tiles = 0
    image_size = int(checkpoint.config["image_size"])
    context_size = int(checkpoint.config.get("context_size", 0))
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
            logits = checkpoint.model(
                images,
                _encode_legacy_vae_features(checkpoint.vae_encoder, images),
            )
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
        raise RuntimeError(f"Legacy VAE tiling left uncovered pixels for {sample.image_path}")
    return logit_sum / counts, tiles


def _encode_legacy_vae_features(
    encoder: LegacyVAEFeatureEncoder,
    images: torch.Tensor,
) -> torch.Tensor:
    vae_images = images[:, :3].mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    mu, _logvar = encoder.model.encode(vae_images)
    if encoder.normalize and encoder.mean is not None and encoder.std is not None:
        mu = (mu - encoder.mean) / encoder.std
    return mu.detach()


class LegacyVAEASPPUNetSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int,
        vae_latent_channels: int,
    ) -> None:
        super().__init__()
        self.stem = _conv_block(input_channels, 48)
        self.down1 = _down_block(48, 96)
        self.down2 = _down_block(96, 192)
        self.vae_projection = nn.Sequential(
            nn.Conv2d(vae_latent_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.down3 = _down_block(192, 256)
        self.context = _AtrousSpatialPyramid(256, 256)
        self.decode2 = _conv_block(448, 192)
        self.decode1 = _conv_block(288, 128)
        self.decode0 = _conv_block(176, 96)
        self.head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, vae_latents: torch.Tensor) -> torch.Tensor:
        level0 = self.stem(image)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level2 = level2 + _resize_like(self.vae_projection(vae_latents), level2)
        latent = self.context(self.down3(level2))
        decoded2 = self.decode2(torch.cat((_resize_like(latent, level2), level2), dim=1))
        decoded1 = self.decode1(torch.cat((_resize_like(decoded2, level1), level1), dim=1))
        decoded0 = self.decode0(torch.cat((_resize_like(decoded1, level0), level0), dim=1))
        return self.head(decoded0)


class _AtrousSpatialPyramid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            (
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6),
            )
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(image) for branch in self.branches], dim=1))


def _resize_like(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        image,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _config() -> dict[str, Any]:
    max_samples = int(os.environ.get("SIM_SATELLITE_HYBRID_MAX_SAMPLES", "0"))
    output_path = OUTPUT_PATH
    if max_samples > 0:
        output_path = (
            f"artifacts/modal-metrics/hybrid-production-vae-fill-official-val{max_samples}"
            "-threshold-sweep.json"
        )
    return {
        "production_checkpoint_path": _modal_artifact_path(PRODUCTION_CHECKPOINT_PATH),
        "vae_checkpoint_path": _modal_artifact_path(LEGACY_VAE_SEGMENTATION_CHECKPOINT_PATH),
        "manifest_path": _modal_data_path(
            "data/openearthmap/manifest.jsonl",
            field_name="manifest_path",
        ),
        "split": "val",
        "max_samples": max_samples,
        "thresholds": DEFAULT_THRESHOLDS,
        "production_tile_stride": 256,
        "production_tile_scales": (0.75, 1.0, 1.25),
        "production_feather_blending": True,
        "production_class_logit_biases": PRODUCTION_CLASS_LOGIT_BIASES,
        "vae_tile_stride": 256,
        "vae_tile_scales": (),
        "vae_feather_blending": False,
        "vae_class_logit_biases": {},
        "output_path": _modal_artifact_path(output_path),
        "device": "cuda",
        "cache_samples": 64,
    }


@app.local_entrypoint()
def modal_main() -> None:
    result = hybrid_production_vae_fill_sweep_remote.remote(_config())
    print(json.dumps(result, indent=2, sort_keys=True))
