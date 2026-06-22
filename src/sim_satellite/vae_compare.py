"""Compare VAE reconstruction behavior on prepared satellite samples."""

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

DEFAULT_SD_CODEC = "stabilityai/sd-vae-ft-mse"


@dataclass(frozen=True)
class VaeCodecSpec:
    label: str
    model: str


@dataclass(frozen=True)
class VaeCompareConfig:
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    validation_holdout_patterns: str = ""
    image_size: int = 256
    batch_size: int = 4
    sample_count: int = 0
    device: str = "auto"
    codec_specs: tuple[VaeCodecSpec, ...] = (VaeCodecSpec("sd", DEFAULT_SD_CODEC),)
    output_path: str = "artifacts/vae-compare.json"
    montage_path: str = ""
    montage_samples: int = 8


class DiffusersVaeCodec:
    def __init__(self, spec: VaeCodecSpec, *, device: torch.device) -> None:
        try:
            diffusers = importlib.import_module("diffusers")
        except ImportError as exc:
            raise RuntimeError(
                "VAE comparison requires diffusers. Install with "
                "`uv sync --extra vae` before running `sim-satellite vae-compare`."
            ) from exc

        self.spec = spec
        self.device = device
        self.vae = diffusers.AutoencoderKL.from_pretrained(spec.model)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(device)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))

    def reconstruct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            encoded = self.vae.encode(images.float().clamp(-1.0, 1.0))
            latents = encoded.latent_dist.mean * self.scaling_factor
            decoded = self.vae.decode(latents / self.scaling_factor).sample
        return decoded.detach().clamp(-1.0, 1.0), latents.detach()


class VaeReconstructionAccumulator:
    def __init__(self) -> None:
        self.metrics = GenerationMetricAccumulator()
        self.original_color = _ColorAccumulator()
        self.reconstruction_color = _ColorAccumulator()
        self.latent_stats = _LatentStatsAccumulator()

    def update(
        self,
        *,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        latents: torch.Tensor,
    ) -> None:
        original_01 = (original.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        reconstructed_01 = (reconstructed.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        self.metrics.update(reconstructed_01, original_01, torch=torch)
        self.original_color.update(original_01)
        self.reconstruction_color.update(reconstructed_01)
        self.latent_stats.update(latents)

    def payload(self, *, sample_count: int) -> dict[str, Any]:
        original = self.original_color.payload()
        reconstructed = self.reconstruction_color.payload()
        rgb_delta = tuple(
            reconstructed["rgb_mean"][index] - original["rgb_mean"][index] for index in range(3)
        )
        rgb_std_delta = tuple(
            reconstructed["rgb_std"][index] - original["rgb_std"][index] for index in range(3)
        )
        return {
            "metrics": self.metrics.payload(
                requested_sample_count=sample_count,
                generation_steps=0,
                guidance_scale=0.0,
            ),
            "color": {
                "original": original,
                "reconstructed": reconstructed,
                "rgb_mean_delta": list(rgb_delta),
                "rgb_mean_abs_delta_max": max(abs(value) for value in rgb_delta),
                "rgb_std_delta": list(rgb_std_delta),
                "chroma_mean_delta": (reconstructed["chroma_mean"] - original["chroma_mean"]),
                "saturation_mean_delta": (
                    reconstructed["saturation_mean"] - original["saturation_mean"]
                ),
            },
            "latents": self.latent_stats.payload(),
        }


def run_vae_comparison(config: VaeCompareConfig) -> dict[str, Any]:
    _validate_config(config)
    samples = select_reconstruction_samples(
        manifest_path=config.manifest_path,
        split=config.split,
        validation_holdout_patterns=config.validation_holdout_patterns,
        sample_count=config.sample_count,
    )
    if not samples:
        raise ValueError("No samples matched the VAE comparison selection")

    device = _resolve_device(config.device)
    model_config = ModelConfig(image_size=config.image_size)
    preloaded_samples = training_data.preload_samples(samples)
    original_montage: list[torch.Tensor] = []
    montage_by_label: dict[str, list[torch.Tensor]] = {}
    codec_payloads: dict[str, Any] = {}

    for spec in config.codec_specs:
        codec = DiffusersVaeCodec(spec, device=device)
        accumulator = VaeReconstructionAccumulator()
        montage_by_label[spec.label] = []
        with torch.inference_mode():
            for batch in iter_batches(samples, config.batch_size):
                clean = _load_center_image_batch(
                    batch,
                    torch=torch,
                    model_config=model_config,
                    device=device,
                    preloaded_samples=preloaded_samples,
                )
                reconstructed, latents = codec.reconstruct(clean)
                accumulator.update(
                    original=clean,
                    reconstructed=reconstructed,
                    latents=latents,
                )
                _collect_montage_samples(
                    clean,
                    reconstructed,
                    originals=original_montage,
                    reconstructions=montage_by_label[spec.label],
                    limit=config.montage_samples,
                )
        codec_payloads[spec.label] = {
            "model": spec.model,
            **accumulator.payload(sample_count=len(samples)),
        }

    payload = {
        "enabled": True,
        "status": "computed",
        "config": {
            **asdict(config),
            "codec_specs": [asdict(spec) for spec in config.codec_specs],
        },
        "sample_count": len(samples),
        "codecs": codec_payloads,
        "comparison": comparison_payload(codec_payloads),
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.montage_path:
        write_comparison_montage(
            codec_labels=tuple(spec.label for spec in config.codec_specs),
            originals=original_montage,
            reconstructions=montage_by_label,
            path=Path(config.montage_path),
        )
    return payload


def parse_codec_specs(values: tuple[str, ...]) -> tuple[VaeCodecSpec, ...]:
    specs = tuple(_parse_codec_spec(value) for value in values)
    labels = [spec.label for spec in specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Codec labels must be unique: {', '.join(duplicates)}")
    return specs


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


def comparison_payload(codec_payloads: dict[str, Any]) -> dict[str, Any]:
    labels = list(codec_payloads)
    if len(labels) < 2:
        return {"baseline": labels[0] if labels else None, "deltas": {}}

    baseline_label = labels[0]
    baseline = codec_payloads[baseline_label]
    deltas: dict[str, Any] = {}
    for label in labels[1:]:
        candidate = codec_payloads[label]
        deltas[label] = {
            "mse_delta": _metric_delta(candidate, baseline, "mse"),
            "psnr_delta": _metric_delta(candidate, baseline, "psnr"),
            "ssim_delta": _metric_delta(candidate, baseline, "ssim"),
            "rgb_mean_abs_delta_max_delta": (
                candidate["color"]["rgb_mean_abs_delta_max"]
                - baseline["color"]["rgb_mean_abs_delta_max"]
            ),
            "chroma_mean_delta_abs_delta": (
                abs(candidate["color"]["chroma_mean_delta"])
                - abs(baseline["color"]["chroma_mean_delta"])
            ),
            "saturation_mean_delta_abs_delta": (
                abs(candidate["color"]["saturation_mean_delta"])
                - abs(baseline["color"]["saturation_mean_delta"])
            ),
        }
    return {"baseline": baseline_label, "deltas": deltas}


def write_comparison_montage(
    *,
    codec_labels: tuple[str, ...],
    originals: list[torch.Tensor],
    reconstructions: dict[str, list[torch.Tensor]],
    path: Path,
) -> Path:
    if not originals:
        raise ValueError("Montage requires at least one sample")
    tile_width, tile_height = _tensor_to_pil(originals[0]).size
    rows = len(originals)
    columns = 1 + len(codec_labels)
    montage = Image.new("RGB", (tile_width * columns, tile_height * rows))
    for row, original in enumerate(originals):
        montage.paste(_tensor_to_pil(original), (0, row * tile_height))
        for column, label in enumerate(codec_labels, start=1):
            montage.paste(
                _tensor_to_pil(reconstructions[label][row]),
                (column * tile_width, row * tile_height),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)
    return path


def _validate_config(config: VaeCompareConfig) -> None:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if config.montage_samples < 0:
        raise ValueError("montage_samples must be non-negative")
    if not config.codec_specs:
        raise ValueError("At least one codec is required")


def _parse_codec_spec(value: str) -> VaeCodecSpec:
    label, separator, model = value.partition("=")
    if not separator or not label.strip() or not model.strip():
        raise ValueError(
            "Codec specs must use LABEL=MODEL_OR_PATH, for example "
            "`sd=stabilityai/sd-vae-ft-mse` or `custom=artifacts/custom-vae`"
        )
    return VaeCodecSpec(label=label.strip(), model=model.strip())


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
    clean, _ = training_data.load_batch(
        samples,
        crop_boxes,
        torch=torch,
        model_config=model_config,
        device=device,
        preloaded_samples=preloaded_samples,
    )
    return clean


def _collect_montage_samples(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    originals: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    limit: int,
) -> None:
    if limit <= 0:
        return
    remaining = limit - len(reconstructions)
    if remaining <= 0:
        return
    reconstructed_01 = (reconstructed.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) / 2.0
    original_01 = (original.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) / 2.0
    if len(originals) < limit:
        originals.extend(original_01[:remaining])
    reconstructions.extend(reconstructed_01[:remaining])


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> float | None:
    candidate_value = candidate["metrics"].get(metric)
    baseline_value = baseline["metrics"].get(metric)
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value - baseline_value)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = diagnostics.rgb_image_tensor(image)
    values = (image * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).numpy()
    return Image.fromarray(values, mode="RGB")


class _ColorAccumulator:
    def __init__(self) -> None:
        self.image_count = 0
        self.pixel_count = 0
        self.width = 0
        self.height = 0
        self.rgb_sum = torch.zeros(3, dtype=torch.float64)
        self.rgb_square_sum = torch.zeros(3, dtype=torch.float64)
        self.chroma_sum = 0.0
        self.chroma_square_sum = 0.0
        self.saturation_sum = 0.0
        self.saturation_square_sum = 0.0

    def update(self, images: torch.Tensor) -> None:
        images = diagnostics.rgb_image_batch_tensor(images).double()
        image_count, height, width, _ = images.shape
        flat = images.reshape(-1, 3)
        chroma = diagnostics.image_chroma(images).double().reshape(-1)
        saturation = diagnostics.image_saturation(images).double().reshape(-1)
        self.image_count += int(image_count)
        self.pixel_count += int(flat.shape[0])
        self.width = int(width)
        self.height = int(height)
        self.rgb_sum += flat.sum(dim=0)
        self.rgb_square_sum += (flat * flat).sum(dim=0)
        self.chroma_sum += float(chroma.sum().item())
        self.chroma_square_sum += float((chroma * chroma).sum().item())
        self.saturation_sum += float(saturation.sum().item())
        self.saturation_square_sum += float((saturation * saturation).sum().item())

    def payload(self) -> dict[str, Any]:
        if self.pixel_count <= 0:
            raise RuntimeError("Color accumulator has no samples")
        rgb_mean = self.rgb_sum / self.pixel_count
        rgb_variance = (self.rgb_square_sum / self.pixel_count - rgb_mean * rgb_mean).clamp_min(0.0)
        chroma_mean = self.chroma_sum / self.pixel_count
        saturation_mean = self.saturation_sum / self.pixel_count
        return {
            "image_count": self.image_count,
            "width": self.width,
            "height": self.height,
            "rgb_mean": [float(value) for value in rgb_mean.tolist()],
            "rgb_std": [float(value) for value in torch.sqrt(rgb_variance).tolist()],
            "chroma_mean": chroma_mean,
            "chroma_std": _standard_deviation(
                self.chroma_square_sum,
                chroma_mean,
                self.pixel_count,
            ),
            "saturation_mean": saturation_mean,
            "saturation_std": _standard_deviation(
                self.saturation_square_sum,
                saturation_mean,
                self.pixel_count,
            ),
        }


class _LatentStatsAccumulator:
    def __init__(self) -> None:
        self.value_count = 0
        self.channel_count = 0
        self.shape: list[int] | None = None
        self.channel_sum: torch.Tensor | None = None
        self.channel_square_sum: torch.Tensor | None = None

    def update(self, latents: torch.Tensor) -> None:
        if latents.ndim != 4:
            raise ValueError(
                f"Expected latents with shape (N, C, H, W), got {tuple(latents.shape)}"
            )
        channel_count = int(latents.shape[1])
        flattened = (
            latents.detach().cpu().float().movedim(1, -1).reshape(-1, channel_count).double()
        )
        if self.channel_sum is None:
            self.channel_count = channel_count
            self.channel_sum = torch.zeros(channel_count, dtype=torch.float64)
            self.channel_square_sum = torch.zeros(channel_count, dtype=torch.float64)
            self.shape = [0, channel_count, int(latents.shape[2]), int(latents.shape[3])]
        if channel_count != self.channel_count:
            raise ValueError(
                f"All latent batches must have the same channel count: {channel_count} "
                f"!= {self.channel_count}"
            )
        assert self.shape is not None
        assert self.channel_sum is not None
        assert self.channel_square_sum is not None
        self.shape[0] += int(latents.shape[0])
        self.value_count += int(flattened.shape[0])
        self.channel_sum += flattened.sum(dim=0)
        self.channel_square_sum += (flattened * flattened).sum(dim=0)

    def payload(self) -> dict[str, Any]:
        if self.channel_sum is None or self.channel_square_sum is None or self.shape is None:
            raise RuntimeError("Latent accumulator has no samples")
        channel_mean = self.channel_sum / self.value_count
        channel_variance = (
            self.channel_square_sum / self.value_count - channel_mean * channel_mean
        ).clamp_min(0.0)
        channel_std = torch.sqrt(channel_variance)
        return {
            "shape": self.shape,
            "channel_mean": [float(value) for value in channel_mean.tolist()],
            "channel_std": [float(value) for value in channel_std.tolist()],
            "mean_abs_max": float(channel_mean.abs().max().item()),
            "std_min": float(channel_std.min().item()),
            "std_max": float(channel_std.max().item()),
        }


def _standard_deviation(square_sum: float, mean: float, count: int) -> float:
    variance = max(square_sum / count - mean * mean, 0.0)
    return math.sqrt(variance)


@click.command("vae-compare", help="Compare VAE reconstructions on prepared samples.")
@click.option("--manifest-path", default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--split", default="train", show_default=True)
@click.option("--validation-holdout-patterns", default="", show_default=True)
@click.option("--image-size", default=256, show_default=True, type=int)
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--sample-count", default=0, show_default=True, type=int)
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--codec",
    "codec_values",
    multiple=True,
    default=(f"sd={DEFAULT_SD_CODEC}",),
    show_default=True,
    help=(
        "VAE codec as LABEL=MODEL_OR_PATH. Pass twice to compare SD and custom, "
        "for example --codec sd=stabilityai/sd-vae-ft-mse --codec custom=artifacts/custom-vae"
    ),
)
@click.option(
    "--output",
    "output_path",
    default="artifacts/vae-compare.json",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--montage-path", default="", show_default=True, type=click.Path(dir_okay=False))
@click.option("--montage-samples", default=8, show_default=True, type=int)
def cli(
    manifest_path: str,
    split: str,
    validation_holdout_patterns: str,
    image_size: int,
    batch_size: int,
    sample_count: int,
    device: str,
    codec_values: tuple[str, ...],
    output_path: Path,
    montage_path: str,
    montage_samples: int,
) -> None:
    try:
        config = VaeCompareConfig(
            manifest_path=manifest_path,
            split=split,
            validation_holdout_patterns=validation_holdout_patterns,
            image_size=image_size,
            batch_size=batch_size,
            sample_count=sample_count,
            device=device,
            codec_specs=parse_codec_specs(codec_values),
            output_path=str(output_path),
            montage_path=montage_path,
            montage_samples=montage_samples,
        )
        payload = run_vae_comparison(config)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
