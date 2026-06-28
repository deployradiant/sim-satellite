"""VAE latent feature loading for segmentation classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig


@dataclass(frozen=True)
class VAEFeatureEncoder:
    model: SatelliteVAE
    latent_channels: int
    mean: torch.Tensor | None
    std: torch.Tensor | None
    normalize: bool


def _load_vae_feature_encoder(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> VAEFeatureEncoder:
    checkpoint_value = config.get("vae_checkpoint_path", "")
    if not checkpoint_value:
        raise ValueError("vae_checkpoint_path is required")
    vae_checkpoint_path = _resolve_vae_checkpoint_path(
        str(checkpoint_value),
        checkpoint_path=checkpoint_path,
    )
    if not vae_checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")

    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"VAE checkpoint must be a mapping: {vae_checkpoint_path}")
    vae_config_values = checkpoint.get("vae_config")
    if not isinstance(vae_config_values, Mapping):
        raise RuntimeError(f"VAE checkpoint is missing vae_config: {vae_checkpoint_path}")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise RuntimeError(f"VAE checkpoint is missing model_state: {vae_checkpoint_path}")

    vae_config = SatelliteVAEConfig(**dict(vae_config_values))
    vae_model = SatelliteVAE(vae_config).to(device)
    vae_model.load_state_dict(model_state)
    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad_(False)

    latent_channels = int(vae_config.latent_channels)
    if latent_channels < 1:
        raise RuntimeError(f"VAE checkpoint has invalid latent_channels: {vae_checkpoint_path}")
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    normalize = bool(config.get("vae_normalize_latents", True))
    if normalize:
        latent_stats = checkpoint.get("latent_stats")
        if not isinstance(latent_stats, Mapping):
            raise RuntimeError(f"VAE checkpoint is missing latent_stats: {vae_checkpoint_path}")
        mean = _latent_stat_tensor(
            latent_stats,
            "mean",
            latent_channels=latent_channels,
            device=device,
            checkpoint_path=vae_checkpoint_path,
        )
        std = _latent_stat_tensor(
            latent_stats,
            "std",
            latent_channels=latent_channels,
            device=device,
            checkpoint_path=vae_checkpoint_path,
        ).clamp_min(1e-6)

    return VAEFeatureEncoder(
        model=vae_model,
        latent_channels=latent_channels,
        mean=mean,
        std=std,
        normalize=normalize,
    )


def _latent_stat_tensor(
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
        raise RuntimeError(
            f"VAE checkpoint latent_stats.{name} has {len(values)} values; "
            f"expected {latent_channels}: {checkpoint_path}"
        )
    try:
        return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"VAE checkpoint latent_stats.{name} must contain numeric values: {checkpoint_path}"
        ) from exc


def _resolve_vae_checkpoint_path(value: str, *, checkpoint_path: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists() or checkpoint_path is None:
        return path

    artifact_root_path = _artifact_root_relative_path(path, checkpoint_path=checkpoint_path)
    if artifact_root_path is not None:
        return artifact_root_path

    return checkpoint_path.parent / path


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


def _encode_vae_features(
    encoder: VAEFeatureEncoder,
    images: torch.Tensor,
) -> torch.Tensor:
    vae_images = images[:, :3].mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    with torch.no_grad():
        mu, _logvar = encoder.model.encode(vae_images)
    if encoder.normalize and encoder.mean is not None and encoder.std is not None:
        mu = (mu - encoder.mean) / encoder.std
    return mu.detach()
