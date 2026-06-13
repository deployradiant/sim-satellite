"""Inference orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from sim_satellite.data.loader import SatelliteDataset
from sim_satellite.data.tensors import load_mask_tensor
from sim_satellite.model import ModelConfig, build_model, require_torch
from sim_satellite.train import DEFAULT_MANIFEST_PATH

DEFAULT_TARGET = "modal"


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

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
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
class InferenceResult:
    output_path: str
    checkpoint_path: str
    mask_path: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_inference_config(
    config: InferenceConfig | Mapping[str, Any] | None,
) -> InferenceConfig:
    if config is None:
        return InferenceConfig()
    if isinstance(config, InferenceConfig):
        return config
    return InferenceConfig.from_dict(config)


def run_inference(
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferenceResult:
    config = coerce_inference_config(config)
    torch, _ = require_torch()

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint:
        raise RuntimeError(f"Inference checkpoint is missing model_config: {checkpoint_path}")
    model_config = ModelConfig.from_dict(checkpoint["model_config"])

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    image = torch.rand(
        (
            1,
            model_config.image_channels,
            model_config.image_size,
            model_config.image_size,
        ),
        generator=generator,
        device="cpu",
    ).to(device)
    mask_path = _resolve_mask_path(config)
    semantic_mask = (
        load_mask_tensor(mask_path, torch=torch, config=model_config).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        for step in reversed(range(config.steps)):
            timestep = torch.full(
                (1,),
                step % model_config.timesteps,
                dtype=torch.long,
                device=device,
            )
            predicted_noise = model(image, semantic_mask, timestep)
            image = (image - predicted_noise / float(config.steps)).clamp(0.0, 1.0)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image": image.detach().cpu(),
            "semantic_mask": semantic_mask.detach().cpu(),
            "model_config": model_config.to_dict(),
            "inference_config": config.to_dict(),
        },
        output_path,
    )

    return InferenceResult(
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        mask_path=str(mask_path),
        device=str(device),
    )


def _resolve_mask_path(config: InferenceConfig) -> Path:
    if config.mask_path is not None:
        path = Path(config.mask_path)
        if not path.exists():
            raise FileNotFoundError(f"Inference mask not found: {path}")
        return path

    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Inference manifest not found: {manifest_path}. "
            "Pass --mask-path or run `sim-satellite data openearthmap` first."
        )
    dataset = SatelliteDataset(manifest_path, split=config.split)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")
    return dataset[0].mask_path


def _run_with_target(config: InferenceConfig, target: str) -> dict[str, Any]:
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
        )
        click.echo(json.dumps(_run_with_target(config, target), indent=2))

    return command


cli = _build_click_command()


if __name__ == "__main__":
    if cli is None:
        raise SystemExit("Click is required for the inference CLI.")
    cli()
