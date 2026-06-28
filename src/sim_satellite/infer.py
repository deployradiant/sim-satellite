"""Inference orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch

from sim_satellite.data.loader import SatelliteDataset
from sim_satellite.data.tensors import load_mask_tensor
from sim_satellite.generation import generate_conditioned_images
from sim_satellite.latent import (
    checkpoint_uses_latents,
    data_model_config_for_latents,
    load_checkpoint_latent_codec,
    semantic_mask_for_model,
)
from sim_satellite.model import ModelConfig, build_model
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

DEFAULT_TARGET = "modal"
DEFAULT_INFERENCE_STEPS = 16


@dataclass(frozen=True)
class InferenceConfig:
    seed: int = 2026
    checkpoint_path: str = "artifacts/tiny-sim-satellite.pt"
    output_path: str = "artifacts/sample.pt"
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    mask_path: str | None = None
    device: str = "cpu"
    steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not self.output_path:
            raise ValueError("output_path is required")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> InferenceConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown inference config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 2026
    mask_path: str = ""
    steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if not self.mask_path:
            raise ValueError("mask_path is required")


@dataclass(frozen=True)
class InferenceResult:
    output_path: str
    checkpoint_path: str
    mask_path: str
    device: str


@dataclass(frozen=True)
class GenerationResult:
    image: Any
    semantic_mask: Any
    model_config: ModelConfig
    checkpoint_path: Path
    mask_path: Path


@dataclass(frozen=True)
class LoadedGenerator:
    torch: Any
    model: Any
    model_config: ModelConfig
    latent_codec: Any | None
    checkpoint_path: Path
    device: Any


def run_inference(
    config: InferenceConfig | None = None,
) -> InferenceResult:
    config = config or InferenceConfig()
    mask_path = _required_mask_path(config.mask_path)
    loaded_generator = load_generator(config.checkpoint_path, device=config.device)
    generation = generate_image_with_model(
        loaded_generator,
        GenerationConfig(
            seed=config.seed,
            mask_path=mask_path,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
        ),
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image": generation.image.detach().cpu(),
            "semantic_mask": generation.semantic_mask.detach().cpu(),
            "model_config": asdict(generation.model_config),
            "inference_config": asdict(config),
        },
        output_path,
    )

    return InferenceResult(
        output_path=str(output_path),
        checkpoint_path=str(generation.checkpoint_path),
        mask_path=str(generation.mask_path),
        device=str(generation.image.device),
    )


def load_generator(checkpoint_path: str | Path, *, device: str) -> LoadedGenerator:
    torch_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path, torch=torch)
    model_state = checkpoint["model_state"]
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    _validate_latent_state_matches_checkpoint(model_config, checkpoint, checkpoint_path)
    model = build_model(model_config)
    model = model.to(torch_device)
    _load_model_state(model, model_state, checkpoint_path=checkpoint_path)
    latent_codec = load_checkpoint_latent_codec(checkpoint, torch=torch, device=torch_device)
    model.eval()
    return LoadedGenerator(
        torch=torch,
        model=model,
        model_config=model_config,
        latent_codec=latent_codec,
        checkpoint_path=checkpoint_path,
        device=torch_device,
    )


def generate_image_with_model(
    loaded_generator: LoadedGenerator,
    config: GenerationConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> GenerationResult:
    torch = loaded_generator.torch
    model = loaded_generator.model
    model_config = loaded_generator.model_config
    latent_codec = loaded_generator.latent_codec
    data_model_config = data_model_config_for_latents(model_config, latent_codec)
    device = loaded_generator.device

    mask_path = _resolve_mask_path(config.mask_path)
    semantic_mask = (
        load_mask_tensor(mask_path, torch=torch, config=data_model_config).unsqueeze(0).to(device)
    )
    generation_mask = semantic_mask_for_model(
        semantic_mask,
        model_size=(model_config.image_size, model_config.image_size),
    )

    image = generate_conditioned_images(
        model,
        generation_mask,
        torch=torch,
        model_config=model_config,
        device=device,
        seed=config.seed,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        latent_codec=latent_codec,
        progress=progress,
    )

    return GenerationResult(
        image=image,
        semantic_mask=semantic_mask,
        model_config=model_config,
        checkpoint_path=loaded_generator.checkpoint_path,
        mask_path=mask_path,
    )


def _load_checkpoint(checkpoint_path: Path, *, torch: Any) -> Mapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Inference checkpoint must be a mapping: {checkpoint_path}")

    missing = [key for key in ("model_config", "model_state") if key not in checkpoint]
    if missing:
        raise RuntimeError(
            f"Inference checkpoint is missing {', '.join(missing)}: {checkpoint_path}"
        )
    if not isinstance(checkpoint["model_config"], Mapping):
        raise RuntimeError(
            f"Inference checkpoint model_config must be a mapping: {checkpoint_path}"
        )
    if not isinstance(checkpoint["model_state"], Mapping):
        raise RuntimeError(f"Inference checkpoint model_state must be a mapping: {checkpoint_path}")
    return checkpoint


def _load_model_state(
    model: Any,
    model_state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    try:
        model.load_state_dict(model_state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Inference checkpoint is incompatible with flow U-Net: {checkpoint_path}. {exc}"
        ) from exc


def _validate_latent_state_matches_checkpoint(
    model_config: ModelConfig,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    if model_config.image_channels == 3:
        return
    if checkpoint_uses_latents(checkpoint):
        return
    raise RuntimeError(
        "Inference checkpoint has non-RGB model channels but is missing latent "
        f"training_config metadata: {checkpoint_path}"
    )


def _required_mask_path(mask_path: str | None) -> str:
    if not mask_path:
        raise ValueError("mask_path is required for inference")
    return mask_path


def _resolve_mask_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Inference mask not found: {path}")
    return path


def _default_mask_path(config: InferenceConfig) -> Path:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Inference manifest not found: {manifest_path}. "
            "Pass --mask-path or run `sim-satellite openearthmap` first."
        )
    dataset = SatelliteDataset(manifest_path, split=config.split)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")
    return dataset[0].mask_path


def _with_cli_mask_path(config: InferenceConfig) -> InferenceConfig:
    if config.mask_path is not None:
        return config
    return InferenceConfig(
        **{
            **asdict(config),
            "mask_path": str(_default_mask_path(config)),
        }
    )


def _run_with_target(config: InferenceConfig, target: str) -> dict[str, Any]:
    config = _with_cli_mask_path(config)
    if target == "local":
        result = asdict(run_inference(config))
        result["execution_target"] = "local"
        return result

    from sim_satellite.modal_app import run_inference_target

    if target != "modal":
        raise ValueError(f"Unsupported inference target: {target}")
    return run_inference_target(config)


def _build_click_command() -> click.Command:
    @click.command("infer", help="Run image generation or steering inference.")
    @click.option(
        "--target",
        type=click.Choice(["modal", "local"]),
        default=DEFAULT_TARGET,
        show_default=True,
    )
    @click.option("--checkpoint-path", default=InferenceConfig.checkpoint_path, show_default=True)
    @click.option("--output-path", default=InferenceConfig.output_path, show_default=True)
    @click.option("--manifest-path", default=InferenceConfig.manifest_path, show_default=True)
    @click.option("--split", default=InferenceConfig.split, show_default=True)
    @click.option(
        "--mask-path",
        default=None,
        help="Mask to condition on. Defaults to first manifest mask.",
    )
    @click.option("--device", default=InferenceConfig.device, show_default=True)
    @click.option("--steps", default=InferenceConfig.steps, show_default=True)
    @click.option("--seed", default=InferenceConfig.seed, show_default=True)
    @click.option("--guidance-scale", default=InferenceConfig.guidance_scale, show_default=True)
    def command(
        target: str,
        checkpoint_path: str,
        output_path: str,
        manifest_path: str,
        split: str,
        mask_path: str | None,
        device: str,
        steps: int,
        seed: int,
        guidance_scale: float,
    ) -> None:
        config = InferenceConfig(
            seed=seed,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            manifest_path=manifest_path,
            split=split,
            mask_path=mask_path,
            device=device,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        click.echo(json.dumps(_run_with_target(config, target), indent=2))

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
