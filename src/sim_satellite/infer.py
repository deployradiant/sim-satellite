"""Inference orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from sim_satellite.data.loader import SatelliteDataset
from sim_satellite.data.tensors import load_mask_tensor
from sim_satellite.model import ModelConfig, build_legacy_tiny_denoiser, build_model, require_torch
from sim_satellite.train import DEFAULT_MANIFEST_PATH

DEFAULT_TARGET = "modal"
LEGACY_TINY_STATE_KEYS = frozenset(
    {
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    seed: int = 1337
    checkpoint_path: str = "artifacts/tiny-sim-satellite.pt"
    output_path: str = "artifacts/sample.pt"
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    mask_path: str | None = None
    device: str = "cpu"
    steps: int = 8
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> InferenceConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown inference config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 1337
    checkpoint_path: str = "artifacts/tiny-sim-satellite.pt"
    mask_path: str = ""
    device: str = "cpu"
    steps: int = 8
    guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not self.mask_path:
            raise ValueError("mask_path is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GenerationConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown generation config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True)
class InferenceResult:
    output_path: str
    checkpoint_path: str
    mask_path: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    image: Any
    semantic_mask: Any
    model_config: ModelConfig
    checkpoint_path: Path
    mask_path: Path
    config: GenerationConfig


@dataclass(frozen=True)
class LoadedGenerator:
    torch: Any
    model: Any
    model_config: ModelConfig
    checkpoint_path: Path
    device: Any
    architecture: str
    legacy_timesteps: int | None = None


def coerce_inference_config(
    config: InferenceConfig | Mapping[str, Any] | None,
) -> InferenceConfig:
    if config is None:
        return InferenceConfig()
    if isinstance(config, InferenceConfig):
        return config
    return InferenceConfig.from_dict(config)


def coerce_generation_config(
    config: GenerationConfig | Mapping[str, Any],
) -> GenerationConfig:
    if isinstance(config, GenerationConfig):
        return config
    return GenerationConfig.from_dict(config)


def run_inference(
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferenceResult:
    config = coerce_inference_config(config)
    mask_path = _required_mask_path(config.mask_path)
    loaded_generator = load_generator(config.checkpoint_path, device=config.device)
    generation = generate_image_with_model(
        loaded_generator,
        GenerationConfig(
            seed=config.seed,
            checkpoint_path=config.checkpoint_path,
            mask_path=mask_path,
            device=config.device,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
        )
    )
    torch, _ = require_torch()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image": generation.image.detach().cpu(),
            "semantic_mask": generation.semantic_mask.detach().cpu(),
            "model_config": generation.model_config.to_dict(),
            "inference_config": config.to_dict(),
        },
        output_path,
    )

    return InferenceResult(
        output_path=str(output_path),
        checkpoint_path=str(generation.checkpoint_path),
        mask_path=str(generation.mask_path),
        device=str(generation.image.device),
    )


def generate_image(config: GenerationConfig | Mapping[str, Any]) -> GenerationResult:
    config = coerce_generation_config(config)
    loaded_generator = load_generator(config.checkpoint_path, device=config.device)
    return generate_image_with_model(loaded_generator, config)


def load_generator(checkpoint_path: str | Path, *, device: str) -> LoadedGenerator:
    torch, _ = require_torch()

    torch_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path, torch=torch)
    model_state = checkpoint["model_state"]
    if _is_legacy_tiny_state(model_state):
        model_config, timesteps = _legacy_tiny_checkpoint_config(checkpoint, checkpoint_path)
        model = build_legacy_tiny_denoiser(model_config, timesteps=timesteps).to(torch_device)
        architecture = "legacy tiny denoiser"
        legacy_timesteps = timesteps
    else:
        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        model = build_model(model_config).to(torch_device)
        architecture = "flow U-Net"
        legacy_timesteps = None

    _load_model_state(
        model,
        model_state,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
    )
    model.eval()
    return LoadedGenerator(
        torch=torch,
        model=model,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        architecture=architecture,
        legacy_timesteps=legacy_timesteps,
    )


def generate_image_with_model(
    loaded_generator: LoadedGenerator,
    config: GenerationConfig | Mapping[str, Any],
) -> GenerationResult:
    config = coerce_generation_config(config)
    torch = loaded_generator.torch
    model = loaded_generator.model
    model_config = loaded_generator.model_config
    device = loaded_generator.device

    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    image_shape = (
        1,
        model_config.image_channels,
        model_config.image_size,
        model_config.image_size,
    )
    if loaded_generator.legacy_timesteps is None:
        image = torch.randn(image_shape, generator=generator, device="cpu").to(device)
    else:
        image = torch.rand(image_shape, generator=generator, device="cpu").to(device)
    mask_path = _resolve_mask_path(config.mask_path)
    semantic_mask = (
        load_mask_tensor(mask_path, torch=torch, config=model_config).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        if loaded_generator.legacy_timesteps is None:
            for step in reversed(range(1, config.steps + 1)):
                timestep = torch.full(
                    (1,),
                    step / float(config.steps),
                    dtype=torch.float32,
                    device=device,
                )
                velocity = _guided_prediction(
                    model,
                    image,
                    semantic_mask,
                    timestep,
                    guidance_scale=config.guidance_scale,
                    torch=torch,
                )
                image = image - velocity / float(config.steps)
            image = image.clamp(0.0, 1.0)
        else:
            for step in reversed(range(config.steps)):
                timestep = torch.full(
                    (1,),
                    step % loaded_generator.legacy_timesteps,
                    dtype=torch.long,
                    device=device,
                )
                predicted_noise = _guided_prediction(
                    model,
                    image,
                    semantic_mask,
                    timestep,
                    guidance_scale=config.guidance_scale,
                    torch=torch,
                )
                image = (image - predicted_noise / float(config.steps)).clamp(0.0, 1.0)

    return GenerationResult(
        image=image,
        semantic_mask=semantic_mask,
        model_config=model_config,
        checkpoint_path=loaded_generator.checkpoint_path,
        mask_path=mask_path,
        config=config,
    )


def _guided_prediction(
    model: Any,
    image: Any,
    semantic_mask: Any,
    timestep: Any,
    *,
    guidance_scale: float,
    torch: Any,
) -> Any:
    conditional = model(image, semantic_mask, timestep)
    if guidance_scale == 1.0:
        return conditional

    empty_mask = torch.zeros_like(semantic_mask)
    unconditional = model(image, empty_mask, timestep)
    return unconditional + guidance_scale * (conditional - unconditional)


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


def _is_legacy_tiny_state(model_state: Mapping[str, Any]) -> bool:
    return set(model_state) == LEGACY_TINY_STATE_KEYS


def _legacy_tiny_checkpoint_config(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> tuple[ModelConfig, int]:
    raw_config = checkpoint["model_config"]
    if not isinstance(raw_config, Mapping):
        raise RuntimeError(
            f"Inference checkpoint model_config must be a mapping: {checkpoint_path}"
        )
    if "timesteps" not in raw_config:
        raise RuntimeError(
            f"Legacy tiny checkpoint is missing timesteps: {checkpoint_path}"
        )

    timesteps = int(raw_config["timesteps"])
    if timesteps < 1:
        raise RuntimeError(
            f"Legacy tiny checkpoint timesteps must be positive: {checkpoint_path}"
        )

    values = dict(raw_config)
    values.pop("timesteps")
    return ModelConfig.from_dict(values), timesteps


def _load_model_state(
    model: Any,
    model_state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    architecture: str,
) -> None:
    try:
        model.load_state_dict(model_state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Inference checkpoint is incompatible with {architecture}: {checkpoint_path}. "
            f"{exc}"
        ) from exc


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
            **config.to_dict(),
            "mask_path": str(_default_mask_path(config)),
        }
    )


def _run_with_target(config: InferenceConfig, target: str) -> dict[str, Any]:
    config = _with_cli_mask_path(config)
    if target == "local":
        result = run_inference(config).to_dict()
        result["execution_target"] = "local"
        return result

    from sim_satellite.modal_app import run_inference_target

    return run_inference_target(config, target=target)


def _build_click_command() -> Any:
    try:
        import click
    except ModuleNotFoundError:
        return None

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
    if cli is None:
        raise SystemExit("Click is required for the inference CLI.")
    cli()
