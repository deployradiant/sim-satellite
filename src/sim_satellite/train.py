"""Training orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS
from sim_satellite.data.tensors import load_image_tensor, load_mask_tensor
from sim_satellite.model import (
    ModelConfig,
    add_noise,
    build_model,
    require_torch,
)

DEFAULT_TARGET = "modal"
DEFAULT_MANIFEST_PATH = "data/openearthmap/manifest.jsonl"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    steps: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-3
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    checkpoint_path: str = "artifacts/tiny-sim-satellite.pt"
    device: str = "cpu"
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = len(OPENEARTHMAP_CLASS_IDS)
    hidden_channels: int = 32
    timesteps: int = 16

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.manifest_path:
            raise ValueError("manifest_path is required")
        if not self.split:
            raise ValueError("split is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingConfig:
        names = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            image_size=self.image_size,
            image_channels=self.image_channels,
            mask_channels=self.mask_channels,
            hidden_channels=self.hidden_channels,
            timesteps=self.timesteps,
            seed=self.seed,
        )


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    final_loss: float
    steps: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_training_config(
    config: TrainingConfig | Mapping[str, Any] | None,
) -> TrainingConfig:
    if config is None:
        return TrainingConfig()
    if isinstance(config, TrainingConfig):
        return config
    return TrainingConfig.from_dict(config)


def run_training(
    config: TrainingConfig | Mapping[str, Any] | None = None,
) -> TrainingResult:
    config = coerce_training_config(config)
    torch, _ = require_torch()

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model_config = config.model_config()
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batches = _load_manifest_batches(config)

    final_loss = 0.0
    model.train()
    for step in range(config.steps):
        clean, semantic_mask = _load_training_batch(
            batches[step % len(batches)],
            torch=torch,
            model_config=model_config,
            device=device,
        )
        timestep = torch.randint(
            low=0,
            high=model_config.timesteps,
            size=(clean.shape[0],),
            generator=generator,
            device="cpu",
        ).to(device)
        noise = torch.randn(clean.shape, generator=generator, device="cpu").to(device)
        noisy = add_noise(clean, noise, timestep, model_config.timesteps)

        optimizer.zero_grad(set_to_none=True)
        predicted_noise = model(noisy, semantic_mask, timestep)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": config.to_dict(),
            "final_loss": final_loss,
        },
        checkpoint_path,
    )

    return TrainingResult(
        checkpoint_path=str(checkpoint_path),
        final_loss=final_loss,
        steps=config.steps,
        device=str(device),
    )


def _load_manifest_batches(config: TrainingConfig) -> tuple[tuple[SatelliteSample, ...], ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. "
            "Run `sim-satellite data openearthmap` first."
        )

    dataset = SatelliteDataset(manifest_path, split=config.split, shuffle_seed=config.seed)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")

    return tuple(iter_batches(dataset.samples, config.batch_size))


def _load_training_batch(
    samples: tuple[SatelliteSample, ...],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
) -> tuple[Any, Any]:
    images = [
        load_image_tensor(sample.image_path, torch=torch, config=model_config) for sample in samples
    ]
    masks = [
        load_mask_tensor(sample.mask_path, torch=torch, config=model_config) for sample in samples
    ]
    return torch.stack(images).to(device), torch.stack(masks).to(device)


def _run_with_target(config: TrainingConfig, target: str) -> dict[str, Any]:
    if target == "local":
        result = run_training(config).to_dict()
        result["execution_target"] = "local"
        return result

    from sim_satellite.modal_app import run_training_target

    return run_training_target(config, target=target)


def _build_click_command() -> Any:
    try:
        import click
    except ModuleNotFoundError:
        return None

    @click.command("train", help="Run model training.")
    @click.option(
        "--target",
        type=click.Choice(["modal", "local"]),
        default=DEFAULT_TARGET,
        show_default=True,
    )
    @click.option("--steps", default=TrainingConfig.steps, show_default=True)
    @click.option("--batch-size", default=TrainingConfig.batch_size, show_default=True)
    @click.option("--learning-rate", default=TrainingConfig.learning_rate, show_default=True)
    @click.option("--manifest-path", default=TrainingConfig.manifest_path, show_default=True)
    @click.option("--split", default=TrainingConfig.split, show_default=True)
    @click.option("--checkpoint-path", default=TrainingConfig.checkpoint_path, show_default=True)
    @click.option("--device", default=TrainingConfig.device, show_default=True)
    @click.option("--seed", default=TrainingConfig.seed, show_default=True)
    def command(
        target: str,
        steps: int,
        batch_size: int,
        learning_rate: float,
        manifest_path: str,
        split: str,
        checkpoint_path: str,
        device: str,
        seed: int,
    ) -> None:
        config = TrainingConfig(
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            manifest_path=manifest_path,
            split=split,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        click.echo(json.dumps(_run_with_target(config, target), indent=2))

    return command


cli = _build_click_command()


if __name__ == "__main__":
    if cli is None:
        raise SystemExit("Click is required for the training CLI.")
    cli()
