"""Native VAE latent helpers for latent-space flow matching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig

LATENT_CHECKPOINT_CONFIG_KEY = "training_config"


@dataclass(frozen=True)
class LatentCodecConfig:
    checkpoint_path: str
    downsample_factor: int
    channels: int


class SatelliteVAELatentCodec:
    def __init__(self, config: LatentCodecConfig, *, torch: Any, device: Any) -> None:
        self.config = config
        self.torch = torch
        self.device = device
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"VAE checkpoint must be a mapping: {checkpoint_path}")

        vae_config_payload = checkpoint.get("vae_config")
        model_state = checkpoint.get("model_state")
        if not isinstance(vae_config_payload, Mapping) or not isinstance(model_state, Mapping):
            raise RuntimeError(
                f"VAE checkpoint is missing vae_config or model_state: {checkpoint_path}"
            )

        vae_config = SatelliteVAEConfig(**dict(vae_config_payload))
        if vae_config.downsample_factor != config.downsample_factor:
            raise RuntimeError(
                "VAE checkpoint downsample_factor does not match training config: "
                f"{vae_config.downsample_factor} != {config.downsample_factor}"
            )
        if vae_config.latent_channels != config.channels:
            raise RuntimeError(
                "VAE checkpoint latent_channels does not match training config: "
                f"{vae_config.latent_channels} != {config.channels}"
            )

        self.vae = SatelliteVAE(vae_config).to(device)
        self.vae.load_state_dict(model_state)
        self.vae.requires_grad_(False)
        self.vae.eval()

        mean, std = _latent_normalization_tensors(
            checkpoint.get("latent_stats"),
            channels=config.channels,
            torch=torch,
            device=device,
        )
        self.mean = mean
        self.std = std

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with self.torch.inference_mode():
            mu, _ = self.vae.encode(images.float().clamp(-1.0, 1.0))
            latents = (mu - self.mean) / self.std
        return latents.detach()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        with self.torch.inference_mode():
            unnormalized = latents.float() * self.std + self.mean
            images = self.vae.decode(unnormalized)
        return images.detach().clamp(-1.0, 1.0)


def config_uses_latents(config: Any) -> bool:
    return getattr(config, "training_space", "pixel") == "latent"


def latent_codec_config(config: Any) -> LatentCodecConfig:
    return LatentCodecConfig(
        checkpoint_path=str(config.latent_checkpoint_path),
        downsample_factor=int(config.latent_downsample_factor),
        channels=int(config.latent_channels),
    )


def load_latent_codec(config: Any, *, torch: Any, device: Any) -> SatelliteVAELatentCodec | None:
    if not config_uses_latents(config):
        return None
    return SatelliteVAELatentCodec(latent_codec_config(config), torch=torch, device=device)


def checkpoint_uses_latents(checkpoint: Mapping[str, Any]) -> bool:
    training_config = checkpoint.get(LATENT_CHECKPOINT_CONFIG_KEY)
    return (
        isinstance(training_config, Mapping) and training_config.get("training_space") == "latent"
    )


def load_checkpoint_latent_codec(
    checkpoint: Mapping[str, Any],
    *,
    torch: Any,
    device: Any,
) -> SatelliteVAELatentCodec | None:
    training_config = checkpoint.get(LATENT_CHECKPOINT_CONFIG_KEY)
    if (
        not isinstance(training_config, Mapping)
        or training_config.get("training_space") != "latent"
    ):
        return None
    return SatelliteVAELatentCodec(
        LatentCodecConfig(
            checkpoint_path=str(training_config["latent_checkpoint_path"]),
            downsample_factor=int(training_config["latent_downsample_factor"]),
            channels=int(training_config["latent_channels"]),
        ),
        torch=torch,
        device=device,
    )


def semantic_mask_for_model(
    semantic_mask: torch.Tensor,
    *,
    model_size: tuple[int, int],
) -> torch.Tensor:
    if tuple(semantic_mask.shape[-2:]) == model_size:
        return semantic_mask
    return torch.nn.functional.interpolate(
        semantic_mask,
        size=model_size,
        mode="area",
    )


def data_model_config_for_latents(model_config: Any, latent_codec: Any | None) -> Any:
    if latent_codec is None:
        return model_config
    return type(model_config)(
        image_size=model_config.image_size * int(latent_codec.config.downsample_factor),
        image_channels=3,
        mask_channels=model_config.mask_channels,
        hidden_channels=model_config.hidden_channels,
        seed=model_config.seed,
    )


def _latent_normalization_tensors(
    stats: Any,
    *,
    channels: int,
    torch: Any,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        isinstance(stats, Mapping)
        and isinstance(stats.get("mean"), list)
        and isinstance(stats.get("std"), list)
    ):
        mean_values = [float(value) for value in stats["mean"]]
        std_values = [max(float(value), 1e-6) for value in stats["std"]]
        if len(mean_values) == channels and len(std_values) == channels:
            mean = torch.tensor(mean_values, dtype=torch.float32, device=device).view(
                1, channels, 1, 1
            )
            std = torch.tensor(std_values, dtype=torch.float32, device=device).view(
                1, channels, 1, 1
            )
            return mean, std

    mean = torch.zeros((1, channels, 1, 1), dtype=torch.float32, device=device)
    std = torch.ones((1, channels, 1, 1), dtype=torch.float32, device=device)
    return mean, std
