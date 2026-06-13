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
    build_model,
    flow_velocity,
    interpolate_flow_sample,
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
    metrics_path: str = ""
    loss_plot_path: str = ""
    device: str = "cpu"
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = len(OPENEARTHMAP_CLASS_IDS)
    hidden_channels: int = 32
    mask_drop_probability: float = 0.0
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.mask_drop_probability < 0.0 or self.mask_drop_probability > 1.0:
            raise ValueError("mask_drop_probability must be between 0 and 1")
        if not self.manifest_path:
            raise ValueError("manifest_path is required")
        if not self.split:
            raise ValueError("split is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown training config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            image_size=self.image_size,
            image_channels=self.image_channels,
            mask_channels=self.mask_channels,
            hidden_channels=self.hidden_channels,
            seed=self.seed,
        )


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    final_loss: float
    steps: int
    device: str
    model_parameters: int
    metrics_path: str
    loss_plot_path: str

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
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batches = _load_manifest_batches(config)

    final_loss = 0.0
    losses: list[dict[str, float | int]] = []
    model.train()
    for step in range(config.steps):
        clean, semantic_mask = _load_training_batch(
            batches[step % len(batches)],
            torch=torch,
            model_config=model_config,
            device=device,
            generator=generator,
        )
        noise = torch.randn(clean.shape, generator=generator, device="cpu").to(device)
        timestep = torch.rand(clean.shape[0], generator=generator, device="cpu").to(device)
        noisy = interpolate_flow_sample(clean, noise, timestep)
        target = flow_velocity(clean, noise)

        semantic_mask = _drop_conditioning_mask(
            semantic_mask,
            drop_probability=config.mask_drop_probability,
            torch=torch,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, semantic_mask, timestep)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        step_number = step + 1
        losses.append({"step": step_number, "loss": final_loss})
        if step_number == 1 or step_number == config.steps or step_number % config.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "step": step_number,
                        "steps": config.steps,
                        "loss": final_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = _metrics_path(config, checkpoint_path)
    loss_plot_path = _loss_plot_path(config, checkpoint_path)
    _write_training_metrics(
        metrics_path,
        model_config=model_config,
        config=config,
        model_parameters=model_parameters,
        losses=losses,
        final_loss=final_loss,
    )
    _write_loss_plot(loss_plot_path, losses)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": config.to_dict(),
            "final_loss": final_loss,
            "metrics_path": str(metrics_path),
            "loss_plot_path": str(loss_plot_path),
        },
        checkpoint_path,
    )

    return TrainingResult(
        checkpoint_path=str(checkpoint_path),
        final_loss=final_loss,
        steps=config.steps,
        device=str(device),
        model_parameters=model_parameters,
        metrics_path=str(metrics_path),
        loss_plot_path=str(loss_plot_path),
    )


def _metrics_path(config: TrainingConfig, checkpoint_path: Path) -> Path:
    if config.metrics_path:
        return Path(config.metrics_path)
    return checkpoint_path.with_suffix(".metrics.json")


def _loss_plot_path(config: TrainingConfig, checkpoint_path: Path) -> Path:
    if config.loss_plot_path:
        return Path(config.loss_plot_path)
    return checkpoint_path.with_suffix(".loss.png")


def _write_training_metrics(
    path: Path,
    *,
    model_config: ModelConfig,
    config: TrainingConfig,
    model_parameters: int,
    losses: list[dict[str, float | int]],
    final_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_config": model_config.to_dict(),
                "training_config": config.to_dict(),
                "model_parameters": model_parameters,
                "final_loss": final_loss,
                "losses": losses,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_loss_plot(path: Path, losses: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    steps = [int(item["step"]) for item in losses]
    values = [float(item["loss"]) for item in losses]
    smoothed = _exponential_moving_average(values, decay=0.95)

    figure, axis = pyplot.subplots(figsize=(8, 4.5), dpi=140)
    axis.plot(steps, values, color="#94a3b8", linewidth=1.0, alpha=0.55, label="step loss")
    axis.plot(steps, smoothed, color="#0f766e", linewidth=2.0, label="EMA 0.95")
    axis.set_title("Training Loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("MSE")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    pyplot.close(figure)


def _exponential_moving_average(values: list[float], *, decay: float) -> list[float]:
    if not values:
        return []
    average = values[0]
    smoothed = []
    for value in values:
        average = decay * average + (1.0 - decay) * value
        smoothed.append(average)
    return smoothed


def _drop_conditioning_mask(
    semantic_mask: Any,
    *,
    drop_probability: float,
    torch: Any,
    generator: Any,
) -> Any:
    if drop_probability <= 0.0:
        return semantic_mask

    keep_mask = torch.rand(
        semantic_mask.shape[0],
        generator=generator,
        device="cpu",
    ).to(semantic_mask.device)
    keep_mask = (keep_mask >= drop_probability).float().view(-1, 1, 1, 1)
    return semantic_mask * keep_mask


def _load_manifest_batches(config: TrainingConfig) -> tuple[tuple[SatelliteSample, ...], ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. "
            "Run `sim-satellite openearthmap` first."
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
    generator: Any,
) -> tuple[Any, Any]:
    crop_boxes = [
        _random_square_crop_box(
            sample,
            crop_size=model_config.image_size,
            torch=torch,
            generator=generator,
        )
        for sample in samples
    ]
    images = [
        load_image_tensor(sample.image_path, torch=torch, config=model_config, crop_box=crop_box)
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    masks = [
        load_mask_tensor(sample.mask_path, torch=torch, config=model_config, crop_box=crop_box)
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    return torch.stack(images).to(device), torch.stack(masks).to(device)


def _random_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    torch: Any,
    generator: Any,
) -> tuple[int, int, int, int]:
    from PIL import Image

    with Image.open(sample.image_path) as image:
        image_size = image.size
    with Image.open(sample.mask_path) as mask:
        mask_size = mask.size
    if image_size != mask_size:
        raise ValueError(
            f"Image and mask sizes must match for aligned crops: "
            f"{sample.image_path} is {image_size}, {sample.mask_path} is {mask_size}"
        )

    width, height = image_size
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {image_size}")

    max_left = width - size
    max_top = height - size
    left = _random_offset(max_left, torch=torch, generator=generator)
    top = _random_offset(max_top, torch=torch, generator=generator)
    return left, top, left + size, top + size


def _random_offset(max_offset: int, *, torch: Any, generator: Any) -> int:
    if max_offset <= 0:
        return 0
    return int(torch.randint(0, max_offset + 1, (1,), generator=generator, device="cpu").item())


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
    @click.option("--metrics-path", default=TrainingConfig.metrics_path, show_default=True)
    @click.option("--loss-plot-path", default=TrainingConfig.loss_plot_path, show_default=True)
    @click.option("--device", default=TrainingConfig.device, show_default=True)
    @click.option("--seed", default=TrainingConfig.seed, show_default=True)
    @click.option("--image-size", default=TrainingConfig.image_size, show_default=True)
    @click.option("--hidden-channels", default=TrainingConfig.hidden_channels, show_default=True)
    @click.option(
        "--mask-drop-probability",
        default=TrainingConfig.mask_drop_probability,
        show_default=True,
    )
    @click.option("--log-every", default=TrainingConfig.log_every, show_default=True)
    def command(
        target: str,
        steps: int,
        batch_size: int,
        learning_rate: float,
        manifest_path: str,
        split: str,
        checkpoint_path: str,
        metrics_path: str,
        loss_plot_path: str,
        device: str,
        seed: int,
        image_size: int,
        hidden_channels: int,
        mask_drop_probability: float,
        log_every: int,
    ) -> None:
        config = TrainingConfig(
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            manifest_path=manifest_path,
            split=split,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            loss_plot_path=loss_plot_path,
            device=device,
            image_size=image_size,
            hidden_channels=hidden_channels,
            mask_drop_probability=mask_drop_probability,
            log_every=log_every,
        )
        click.echo(json.dumps(_run_with_target(config, target), indent=2))

    return command


cli = _build_click_command()


if __name__ == "__main__":
    if cli is None:
        raise SystemExit("Click is required for the training CLI.")
    cli()
