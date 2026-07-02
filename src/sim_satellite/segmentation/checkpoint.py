"""Checkpoint IO for segmentation classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.segmentation.config import (
    SUPPORTED_SEGMENTATION_MODEL_NAMES,
    VAE_ASPP_UNET_MODEL_NAME,
    segmentation_model_uses_vae_features,
)
from sim_satellite.segmentation.models import _segmentation_input_channels, build_segmentation_model
from sim_satellite.segmentation.vae import VAEFeatureEncoder, _load_vae_feature_encoder


@dataclass(frozen=True)
class SegmentationCheckpoint:
    model: nn.Module
    config: dict[str, Any]
    classes: tuple[str, ...]
    vae_encoder: VAEFeatureEncoder | None


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = _checkpoint_model_name(config)
    torch.save(
        {
            "model": model_name,
            "model_state": model.state_dict(),
            "classes": list(OPENEARTHMAP_CLASSES),
            "config": config,
            "metrics": metrics,
            "step": step,
        },
        path,
    )


def _load_segmentation_checkpoint(path: Path, *, device: str) -> SegmentationCheckpoint:
    if not path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Segmentation checkpoint must be a mapping: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"Segmentation checkpoint config must be a mapping: {path}")
    model_name = checkpoint.get("model")
    if not isinstance(model_name, str):
        raise RuntimeError(f"Segmentation checkpoint model must be a string: {path}")
    if model_name not in SUPPORTED_SEGMENTATION_MODEL_NAMES:
        raise RuntimeError(
            f"Unsupported segmentation checkpoint model {model_name!r}; "
            f"expected one of {', '.join(SUPPORTED_SEGMENTATION_MODEL_NAMES)}: {path}"
        )
    classes = checkpoint.get("classes")
    if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
        raise RuntimeError(f"Segmentation checkpoint classes must be a sequence: {path}")
    classes = tuple(str(class_name) for class_name in classes)
    if classes != OPENEARTHMAP_CLASSES:
        raise RuntimeError(f"Segmentation checkpoint classes do not match OpenEarthMap: {path}")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Segmentation checkpoint is missing model_state: {path}")
    vae_encoder = None
    vae_latent_channels = 0
    if segmentation_model_uses_vae_features(model_name):
        vae_encoder = _load_vae_feature_encoder(
            dict(config),
            device=torch.device(device),
            checkpoint_path=path,
        )
        vae_latent_channels = _checkpoint_vae_latent_channels(
            config,
            vae_encoder=vae_encoder,
            checkpoint_path=path,
        )
    model = build_segmentation_model(
        num_classes=len(classes),
        input_channels=_segmentation_input_channels(config),
        vae_latent_channels=vae_latent_channels,
        model_name=model_name,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return SegmentationCheckpoint(
        model=model,
        config=dict(config),
        classes=classes,
        vae_encoder=vae_encoder,
    )


def _checkpoint_model_name(config: Mapping[str, Any]) -> str:
    raw_value = config.get("model_name")
    if raw_value is None:
        return VAE_ASPP_UNET_MODEL_NAME
    if not isinstance(raw_value, str):
        raise RuntimeError("Segmentation checkpoint config model_name must be a string")
    return raw_value


def _checkpoint_vae_latent_channels(
    config: Mapping[str, Any],
    *,
    vae_encoder: VAEFeatureEncoder,
    checkpoint_path: Path,
) -> int:
    raw_value = config.get("vae_latent_channels")
    if raw_value is None:
        return vae_encoder.latent_channels
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Segmentation checkpoint vae_latent_channels must be an integer: {checkpoint_path}"
        ) from exc
    if value != vae_encoder.latent_channels:
        raise RuntimeError(
            f"Segmentation checkpoint vae_latent_channels={value} does not match "
            f"VAE latent_channels={vae_encoder.latent_channels}: {checkpoint_path}"
        )
    return value
