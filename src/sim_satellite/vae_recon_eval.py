"""Evaluate VAE reconstruction fidelity before latent-flow training."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import torch
from PIL import Image

from sim_satellite import diagnostics, training_data
from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.data.prepared import PreparedSample
from sim_satellite.evaluation_metrics import GenerationMetricAccumulator
from sim_satellite.model import ModelConfig
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

DEFAULT_CODEC = "stabilityai/sd-vae-ft-mse"
DEFAULT_PSNR_GATE = 26.0
DEFAULT_SSIM_GATE = 0.80
DEFAULT_RGB_MEAN_ABS_DELTA_GATE = 0.01
DEFAULT_CHROMA_ABS_DELTA_GATE = 0.01
DEFAULT_SATURATION_ABS_DELTA_GATE = 0.01
DEFAULT_LATENT_MEAN_ABS_GATE = 0.10
DEFAULT_LATENT_STD_MIN = 0.80
DEFAULT_LATENT_STD_MAX = 1.20


@dataclass(frozen=True)
class VaeReconEvalConfig:
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    validation_holdout_patterns: str = ""
    codec: str = DEFAULT_CODEC
    image_size: int = 256
    batch_size: int = 4
    sample_count: int = 0
    device: str = "auto"
    output_path: str = "artifacts/vae-recon-eval.json"
    montage_path: str = ""


def run_vae_reconstruction_eval(config: VaeReconEvalConfig) -> dict[str, Any]:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.sample_count < 0:
        raise ValueError("sample_count must be non-negative")

    samples = select_reconstruction_samples(
        manifest_path=config.manifest_path,
        split=config.split,
        validation_holdout_patterns=config.validation_holdout_patterns,
        sample_count=config.sample_count,
    )
    if not samples:
        raise ValueError("No samples matched the VAE reconstruction selection")

    device = _resolve_device(config.device)
    vae = _load_autoencoder_kl(config.codec, device=device)
    vae.eval()
    model_config = ModelConfig(image_size=config.image_size)
    preloaded_samples = training_data.preload_samples(samples)
    accumulator = GenerationMetricAccumulator()
    latent_batches: list[torch.Tensor] = []
    original_batches: list[torch.Tensor] = []
    reconstruction_batches: list[torch.Tensor] = []
    originals_for_montage: list[torch.Tensor] = []
    reconstructions_for_montage: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in iter_batches(samples, config.batch_size):
            original = _load_center_image_batch(
                batch,
                torch=torch,
                model_config=model_config,
                device=device,
                preloaded_samples=preloaded_samples,
            )
            latents = _vae_encode(vae, original)
            reconstruction = _vae_decode(vae, latents)
            original_01 = (original.float().clamp(-1.0, 1.0) + 1.0) / 2.0
            reconstruction_01 = (reconstruction.float().clamp(-1.0, 1.0) + 1.0) / 2.0
            accumulator.update(reconstruction_01, original_01, torch=torch)
            latent_batches.append(latents.detach().cpu())
            original_batches.append(original_01.detach().cpu())
            reconstruction_batches.append(reconstruction_01.detach().cpu())
            if config.montage_path and len(originals_for_montage) < 8:
                originals_for_montage.extend(original_01.detach().cpu())
                reconstructions_for_montage.extend(reconstruction_01.detach().cpu())

    metrics = accumulator.payload(
        requested_sample_count=len(samples),
        generation_steps=0,
        guidance_scale=0.0,
    )
    color_metrics = reconstruction_color_metrics_from_batches(
        torch.cat(original_batches, dim=0),
        torch.cat(reconstruction_batches, dim=0),
    )
    latent_stats = latent_stats_payload(torch.cat(latent_batches, dim=0))
    gate_payload = reconstruction_gate_payload(metrics, color_metrics, latent_stats)
    payload: dict[str, Any] = {
        "enabled": True,
        "status": "computed",
        "config": asdict(config),
        "codec": config.codec,
        "sample_count": len(samples),
        "metrics": metrics,
        "color_metrics": color_metrics,
        "latent_stats": latent_stats,
        "gates": gate_payload,
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.montage_path:
        write_reconstruction_montage(
            originals_for_montage[:8],
            reconstructions_for_montage[:8],
            Path(config.montage_path),
        )
    return payload


def select_reconstruction_samples(
    *,
    manifest_path: str,
    split: str,
    validation_holdout_patterns: str,
    sample_count: int,
) -> tuple[SatelliteSample, ...]:
    dataset = SatelliteDataset(manifest_path, split=split or None)
    samples = tuple(dataset)
    patterns = tuple(
        pattern.strip() for pattern in validation_holdout_patterns.split(",") if pattern.strip()
    )
    if patterns:
        samples = tuple(sample for sample in samples if _sample_matches_any(sample, patterns))
    if sample_count > 0:
        samples = samples[:sample_count]
    return samples


def reconstruction_color_metrics_from_batches(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, Any]:
    original_stats = diagnostics.image_batch_color_stats(original)
    reconstructed_stats = diagnostics.image_batch_color_stats(reconstructed)
    rgb_delta = tuple(
        reconstructed_value - original_value
        for reconstructed_value, original_value in zip(
            reconstructed_stats.rgb_mean,
            original_stats.rgb_mean,
            strict=True,
        )
    )
    rgb_std_delta = tuple(
        reconstructed_value - original_value
        for reconstructed_value, original_value in zip(
            reconstructed_stats.rgb_std,
            original_stats.rgb_std,
            strict=True,
        )
    )
    return {
        "original": diagnostics.color_stats_payload(original_stats),
        "reconstructed": diagnostics.color_stats_payload(reconstructed_stats),
        "rgb_mean_delta_r": rgb_delta[0],
        "rgb_mean_delta_g": rgb_delta[1],
        "rgb_mean_delta_b": rgb_delta[2],
        "rgb_mean_abs_delta_max": max(abs(value) for value in rgb_delta),
        "rgb_std_delta_r": rgb_std_delta[0],
        "rgb_std_delta_g": rgb_std_delta[1],
        "rgb_std_delta_b": rgb_std_delta[2],
        "chroma_mean_delta": reconstructed_stats.chroma_mean - original_stats.chroma_mean,
        "saturation_mean_delta": (
            reconstructed_stats.saturation_mean - original_stats.saturation_mean
        ),
    }


def latent_stats_payload(latents: torch.Tensor) -> dict[str, Any]:
    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape (N, C, H, W), got {tuple(latents.shape)}")
    flattened = latents.detach().float().movedim(1, -1).reshape(-1, latents.shape[1])
    channel_mean = flattened.mean(dim=0)
    channel_std = flattened.std(dim=0, unbiased=False)
    return {
        "shape": list(latents.shape),
        "channel_mean": [float(value) for value in channel_mean.tolist()],
        "channel_std": [float(value) for value in channel_std.tolist()],
        "mean_abs_max": float(channel_mean.abs().max().item()),
        "std_min": float(channel_std.min().item()),
        "std_max": float(channel_std.max().item()),
    }


def reconstruction_gate_payload(
    metrics: dict[str, Any],
    color_metrics: dict[str, Any],
    latent_stats: dict[str, Any],
) -> dict[str, Any]:
    psnr = metrics.get("psnr")
    ssim = metrics.get("ssim")
    checks = {
        "psnr": psnr is not None and psnr >= DEFAULT_PSNR_GATE,
        "ssim": ssim is not None and ssim >= DEFAULT_SSIM_GATE,
        "rgb_mean_abs_delta": (
            color_metrics["rgb_mean_abs_delta_max"] <= DEFAULT_RGB_MEAN_ABS_DELTA_GATE
        ),
        "chroma_mean_abs_delta": (
            abs(color_metrics["chroma_mean_delta"]) <= DEFAULT_CHROMA_ABS_DELTA_GATE
        ),
        "saturation_mean_abs_delta": (
            abs(color_metrics["saturation_mean_delta"]) <= DEFAULT_SATURATION_ABS_DELTA_GATE
        ),
        "latent_mean_abs": latent_stats["mean_abs_max"] <= DEFAULT_LATENT_MEAN_ABS_GATE,
        "latent_std_range": (
            latent_stats["std_min"] >= DEFAULT_LATENT_STD_MIN
            and latent_stats["std_max"] <= DEFAULT_LATENT_STD_MAX
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "psnr": DEFAULT_PSNR_GATE,
            "ssim": DEFAULT_SSIM_GATE,
            "rgb_mean_abs_delta_max": DEFAULT_RGB_MEAN_ABS_DELTA_GATE,
            "chroma_mean_abs_delta": DEFAULT_CHROMA_ABS_DELTA_GATE,
            "saturation_mean_abs_delta": DEFAULT_SATURATION_ABS_DELTA_GATE,
            "latent_mean_abs_max": DEFAULT_LATENT_MEAN_ABS_GATE,
            "latent_std_min": DEFAULT_LATENT_STD_MIN,
            "latent_std_max": DEFAULT_LATENT_STD_MAX,
        },
    }


def write_reconstruction_montage(
    originals: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    path: Path,
) -> Path:
    if len(originals) != len(reconstructions):
        raise ValueError("Montage originals and reconstructions must have the same length")
    if not originals:
        raise ValueError("Montage requires at least one reconstruction pair")

    tiles: list[Image.Image] = []
    for original, reconstruction in zip(originals, reconstructions, strict=True):
        tiles.append(_tensor_to_pil(original))
        tiles.append(_tensor_to_pil(reconstruction))
    tile_width, tile_height = tiles[0].size
    rows = math.ceil(len(tiles) / 2)
    montage = Image.new("RGB", (tile_width * 2, tile_height * rows))
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % 2) * tile_width, (index // 2) * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)
    return path


def _load_center_image_batch(
    samples: tuple[SatelliteSample, ...],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    preloaded_samples: dict[SatelliteSample, PreparedSample],
) -> torch.Tensor:
    crop_boxes = [
        training_data.center_square_crop_box(
            sample,
            crop_size=model_config.image_size,
            preloaded_samples=preloaded_samples,
        )
        for sample in samples
    ]
    clean, _, _ = training_data.load_batch(
        samples,
        crop_boxes,
        torch=torch,
        model_config=model_config,
        device=device,
        preloaded_samples=preloaded_samples,
    )
    return clean


def _sample_matches_any(sample: SatelliteSample, patterns: tuple[str, ...]) -> bool:
    image_path = sample.image_path.as_posix()
    mask_path = sample.mask_path.as_posix()
    return any(pattern in image_path or pattern in mask_path for pattern in patterns)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return resolved


def _load_autoencoder_kl(codec: str, *, device: torch.device) -> Any:
    try:
        diffusers = importlib.import_module("diffusers")
    except ImportError as exc:
        raise RuntimeError(
            "VAE reconstruction eval requires diffusers. Install with "
            "`uv sync --extra vae` before running `sim-satellite vae-recon-eval`."
        ) from exc

    autoencoder_kl = diffusers.AutoencoderKL.from_pretrained(codec)
    return autoencoder_kl.to(device)


def _vae_encode(vae: Any, images: torch.Tensor) -> torch.Tensor:
    encoded = vae.encode(images)
    latent_dist = encoded.latent_dist
    latents = latent_dist.mean
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return latents * scaling_factor


def _vae_decode(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return vae.decode(latents / scaling_factor).sample


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = diagnostics.rgb_image_tensor(image)
    values = (image * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).numpy()
    return Image.fromarray(values, mode="RGB")


@click.command("vae-recon-eval", help="Evaluate VAE reconstruction quality on prepared samples.")
@click.option("--manifest-path", default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--split", default="train", show_default=True)
@click.option("--validation-holdout-patterns", default="", show_default=True)
@click.option("--codec", default=DEFAULT_CODEC, show_default=True)
@click.option("--image-size", default=256, show_default=True, type=int)
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--sample-count", default=0, show_default=True, type=int)
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--output",
    "output_path",
    default="artifacts/vae-recon-eval.json",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--montage-path",
    default="",
    show_default=True,
    type=click.Path(dir_okay=False),
)
def cli(
    manifest_path: str,
    split: str,
    validation_holdout_patterns: str,
    codec: str,
    image_size: int,
    batch_size: int,
    sample_count: int,
    device: str,
    output_path: Path,
    montage_path: str,
) -> None:
    config = VaeReconEvalConfig(
        manifest_path=manifest_path,
        split=split,
        validation_holdout_patterns=validation_holdout_patterns,
        codec=codec,
        image_size=image_size,
        batch_size=batch_size,
        sample_count=sample_count,
        device=device,
        output_path=str(output_path),
        montage_path=montage_path,
    )
    try:
        payload = run_vae_reconstruction_eval(config)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
