"""Evaluate generation metrics for one or more existing checkpoints."""

from __future__ import annotations

import glob
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import click

from sim_satellite import evaluation_metrics, training_data, training_log
from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.generation import generate_conditioned_images
from sim_satellite.model import ModelConfig, build_model
from sim_satellite.training_config import TrainingConfig, load_training_config_file

DEFAULT_GENERATION_METRICS_SAMPLES = 60
DEFAULT_GENERATION_METRICS_STEPS = 60
DEFAULT_METRICS_JSON_PATH = Path("artifacts/generation-metrics-comparison.json")
DEFAULT_METRICS_MARKDOWN_PATH = Path("artifacts/generation-metrics-comparison.md")


def resolve_checkpoint_paths(
    checkpoint_paths: tuple[Path, ...],
    checkpoint_globs: tuple[str, ...],
    *,
    default_checkpoint_path: Path,
) -> tuple[Path, ...]:
    paths = list(checkpoint_paths)
    for pattern in checkpoint_globs:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            raise ValueError(f"Checkpoint glob did not match any files: {pattern}")
        paths.extend(matches)
    if not paths:
        paths.append(default_checkpoint_path)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return tuple(resolved)


def generation_metrics_output_paths(
    json_path: Path | None,
    markdown_path: Path | None,
) -> tuple[Path, Path]:
    return (
        json_path or DEFAULT_METRICS_JSON_PATH,
        markdown_path or DEFAULT_METRICS_MARKDOWN_PATH,
    )


def evaluate_generation_metrics_for_checkpoints(
    config: TrainingConfig,
    *,
    checkpoint_paths: tuple[Path, ...],
    output_json_path: Path,
    output_markdown_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_checkpoint_paths(checkpoint_paths)
    validation_samples = _load_validation_samples(config)
    selected_samples = validation_samples[: config.generation_metrics_samples]
    if len(selected_samples) < config.generation_metrics_samples:
        raise RuntimeError(
            "Not enough validation samples for generation metrics: "
            f"requested {config.generation_metrics_samples}, found {len(selected_samples)}"
        )

    payload: dict[str, Any] = {
        "dry_run": dry_run,
        "started_at_epoch_utc": time.time(),
        "config": _config_payload(config),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "validation_samples": [str(sample.prepared_path) for sample in selected_samples],
        "output_json_path": str(output_json_path),
        "output_markdown_path": str(output_markdown_path),
    }
    if dry_run:
        payload["results"] = []
        payload["markdown_table"] = render_generation_metrics_markdown(payload)
        return payload

    import torch

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "generation metrics evaluation requires CUDA; use --dry-run to preflight"
        )

    checkpoint_model_config = _shared_checkpoint_model_config(checkpoint_paths, torch=torch)
    batches = tuple(iter_batches(selected_samples, config.batch_size))
    preloaded = _preload_validation_samples(selected_samples)
    validation_batches = tuple(tuple(preloaded[sample] for sample in batch) for batch in batches)
    results = [
        _evaluate_checkpoint(
            checkpoint_path,
            config=config,
            model_config=checkpoint_model_config,
            validation_batches=validation_batches,
            torch=torch,
            device=device,
        )
        for checkpoint_path in checkpoint_paths
    ]
    payload["results"] = results
    payload["finished_at_epoch_utc"] = time.time()
    payload["markdown_table"] = render_generation_metrics_markdown(payload)
    _write_generation_metrics_outputs(payload, output_json_path, output_markdown_path)
    return payload


def render_generation_metrics_markdown(payload: dict[str, Any]) -> str:
    rows = [
        (
            "checkpoint",
            "status",
            "samples",
            "mse",
            "psnr",
            "ssim",
            "mask_rgb_mae",
            "mask_grad_mae",
            "boundary_delta",
        ),
        ("---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"),
    ]
    for result in payload.get("results", []):
        metrics = result["generation_metrics"]
        rows.append(
            (
                _markdown_escape(Path(result["checkpoint_path"]).name),
                str(metrics.get("status", "")),
                str(metrics.get("sample_count", "")),
                _format_metric(metrics.get("mse")),
                _format_metric(metrics.get("psnr")),
                _format_metric(metrics.get("ssim")),
                _format_metric(metrics.get("mask_rgb_mean_mae_macro")),
                _format_metric(metrics.get("mask_gradient_mae_macro")),
                _format_metric(metrics.get("mask_boundary_gradient_delta")),
            )
        )
    if len(rows) == 2:
        rows.append(("(dry run)", "not computed", "0", "", "", "", "", "", ""))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    config: TrainingConfig,
    model_config: ModelConfig,
    validation_batches: tuple[tuple[Any, ...], ...],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started_at = time.monotonic()
    checkpoint, _ = training_log.load_training_checkpoint(
        checkpoint_path,
        torch=torch,
        map_location="cpu",
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    accumulator = evaluation_metrics.GenerationMetricAccumulator(
        class_count=model_config.mask_channels,
        image_channels=model_config.image_channels,
        torch=torch,
    )
    with torch.no_grad():
        for batch_index, samples in enumerate(validation_batches):
            remaining = config.generation_metrics_samples - accumulator.sample_count
            if remaining <= 0:
                break
            samples = samples[:remaining]
            crop_boxes = [
                training_data.center_square_crop_box(
                    sample,
                    crop_size=model_config.image_size,
                )
                for sample in samples
            ]
            clean, semantic_mask = training_data.load_batch(
                samples,
                crop_boxes,
                torch=torch,
                model_config=model_config,
                device=device,
            )
            generated = generate_conditioned_images(
                model,
                semantic_mask,
                torch=torch,
                model_config=model_config,
                device=device,
                seed=config.seed + 20_000 + batch_index,
                steps=config.generation_metrics_steps,
                guidance_scale=config.generation_metrics_guidance_scale,
                precision=config.precision,
            )
            real = (clean.float().clamp(-1.0, 1.0) + 1.0) / 2.0
            accumulator.update(generated, real, semantic_mask=semantic_mask, torch=torch)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_final_loss": float(checkpoint["final_loss"]),
        "model_config": asdict(model_config),
        "elapsed_seconds": max(0.0, time.monotonic() - started_at),
        "generation_metrics": accumulator.payload(
            requested_sample_count=config.generation_metrics_samples,
            generation_steps=config.generation_metrics_steps,
            guidance_scale=config.generation_metrics_guidance_scale,
        ),
    }


def _load_validation_samples(config: TrainingConfig) -> tuple[SatelliteSample, ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. Run data preparation first."
        )

    if config.validation_split:
        dataset = SatelliteDataset(
            manifest_path,
            split=config.validation_split,
            shuffle_seed=config.seed + 1,
        )
        samples = dataset.samples
    else:
        patterns = _validation_holdout_patterns(config)
        if not patterns:
            raise ValueError(
                "generation metrics evaluation requires validation_split "
                "or validation_holdout_patterns"
            )
        dataset = SatelliteDataset(manifest_path, split=config.split, shuffle_seed=config.seed + 1)
        samples = tuple(
            sample for sample in dataset.samples if _sample_matches_any(sample, patterns)
        )

    if not samples:
        raise RuntimeError(f"No validation samples found in {manifest_path}")
    return samples


def _shared_checkpoint_model_config(
    checkpoint_paths: tuple[Path, ...],
    *,
    torch: Any,
) -> ModelConfig:
    first_model_config: ModelConfig | None = None
    for checkpoint_path in checkpoint_paths:
        _, model_config = training_log.load_training_checkpoint(
            checkpoint_path,
            torch=torch,
            map_location="cpu",
        )
        if first_model_config is None:
            first_model_config = model_config
            continue
        if model_config != first_model_config:
            raise ValueError(
                "All checkpoints must use the same model_config for fixed-mask comparison"
            )
    if first_model_config is None:
        raise ValueError("At least one checkpoint is required")
    return first_model_config


def _preload_validation_samples(
    samples: tuple[SatelliteSample, ...],
) -> dict[SatelliteSample, Any]:
    preload_started_at = time.monotonic()
    click.echo(
        json.dumps(
            {
                "event": "generation_metrics_preload_start",
                "sample_count": len(samples),
                "logged_at_epoch_utc": time.time(),
            }
        )
    )

    def report_progress(completed: int, total: int) -> None:
        click.echo(
            json.dumps(
                {
                    "event": "generation_metrics_preload_progress",
                    "completed_samples": completed,
                    "total_samples": total,
                    "elapsed_seconds": max(0.0, time.monotonic() - preload_started_at),
                    "logged_at_epoch_utc": time.time(),
                }
            )
        )

    return training_data.preload_samples(samples, progress=report_progress)


def _validate_checkpoint_paths(checkpoint_paths: tuple[Path, ...]) -> None:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required")
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint not found: {', '.join(missing)}")


def _write_generation_metrics_outputs(
    payload: dict[str, Any],
    output_json_path: Path,
    output_markdown_path: Path,
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown_path.write_text(payload["markdown_table"], encoding="utf-8")


def _config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "manifest_path": config.manifest_path,
        "split": config.split,
        "validation_split": config.validation_split,
        "validation_holdout_patterns": config.validation_holdout_patterns,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "generation_metrics_samples": config.generation_metrics_samples,
        "generation_metrics_steps": config.generation_metrics_steps,
        "generation_metrics_guidance_scale": config.generation_metrics_guidance_scale,
        "device": config.device,
        "precision": config.precision,
    }


def _validation_holdout_patterns(config: TrainingConfig) -> tuple[str, ...]:
    return tuple(
        pattern.strip()
        for pattern in config.validation_holdout_patterns.split(",")
        if pattern.strip()
    )


def _sample_matches_any(sample: SatelliteSample, patterns: tuple[str, ...]) -> bool:
    image_path = sample.image_path.as_posix()
    mask_path = sample.mask_path.as_posix()
    return any(pattern in image_path or pattern in mask_path for pattern in patterns)


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


@click.command(
    "generation-metrics",
    help="Evaluate generation metrics for one or more existing checkpoints.",
)
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--checkpoint",
    "checkpoint_paths",
    multiple=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Checkpoint to evaluate. May be passed more than once.",
)
@click.option(
    "--checkpoint-glob",
    "checkpoint_globs",
    multiple=True,
    help="Glob of checkpoints to evaluate. May be passed more than once.",
)
@click.option("--samples", default=DEFAULT_GENERATION_METRICS_SAMPLES, show_default=True)
@click.option(
    "--steps",
    "generation_steps",
    default=DEFAULT_GENERATION_METRICS_STEPS,
    show_default=True,
)
@click.option("--guidance-scale", type=float, default=None)
@click.option(
    "--manifest-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override the config manifest path.",
)
@click.option(
    "--validation-split",
    default=None,
    help="Override the config validation split, e.g. val.",
)
@click.option(
    "--validation-holdout-patterns",
    default=None,
    help="Comma-separated image/mask path substrings to hold out for validation.",
)
@click.option(
    "--device",
    default=None,
    help="Override the config device, e.g. cuda, cuda:0, or cpu.",
)
@click.option("--precision", type=click.Choice(["float32", "bf16"]), default=None)
@click.option(
    "--output-json",
    "output_json_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--output-markdown",
    "output_markdown_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--target", type=click.Choice(["local", "modal"]), default="local", show_default=True)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate paths and sample selection without generation.",
)
def cli(
    config_path: Path,
    checkpoint_paths: tuple[Path, ...],
    checkpoint_globs: tuple[str, ...],
    samples: int,
    generation_steps: int,
    guidance_scale: float | None,
    manifest_path: Path | None,
    validation_split: str | None,
    validation_holdout_patterns: str | None,
    device: str | None,
    precision: str | None,
    output_json_path: Path | None,
    output_markdown_path: Path | None,
    target: str,
    dry_run: bool,
) -> None:
    try:
        base_config = load_training_config_file(config_path)
        checkpoints = resolve_checkpoint_paths(
            checkpoint_paths,
            checkpoint_globs,
            default_checkpoint_path=Path(base_config.checkpoint_path),
        )
        json_path, markdown_path = generation_metrics_output_paths(
            output_json_path,
            output_markdown_path,
        )
        config = replace(
            base_config,
            generation_metrics_samples=samples,
            generation_metrics_steps=generation_steps,
            generation_metrics_guidance_scale=(
                base_config.generation_metrics_guidance_scale
                if guidance_scale is None
                else guidance_scale
            ),
            manifest_path=(
                base_config.manifest_path if manifest_path is None else str(manifest_path)
            ),
            validation_split=(
                base_config.validation_split if validation_split is None else validation_split
            ),
            validation_holdout_patterns=(
                base_config.validation_holdout_patterns
                if validation_holdout_patterns is None
                else validation_holdout_patterns
            ),
            device=base_config.device if device is None else device,
            precision=base_config.precision if precision is None else precision,
        )
        _validate_evaluation_config(config)
        result = _run_with_target(
            config,
            checkpoint_paths=checkpoints,
            output_json_path=json_path,
            output_markdown_path=markdown_path,
            target=target,
            dry_run=dry_run,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(json.dumps(result, indent=2, sort_keys=True))


def _run_with_target(
    config: TrainingConfig,
    *,
    checkpoint_paths: tuple[Path, ...],
    output_json_path: Path,
    output_markdown_path: Path,
    target: str,
    dry_run: bool,
) -> dict[str, Any]:
    if target == "local":
        return evaluate_generation_metrics_for_checkpoints(
            config,
            checkpoint_paths=checkpoint_paths,
            output_json_path=output_json_path,
            output_markdown_path=output_markdown_path,
            dry_run=dry_run,
        )
    if target == "modal":
        from sim_satellite.modal_app import run_generation_metrics_target

        return run_generation_metrics_target(
            config,
            checkpoint_paths=checkpoint_paths,
            output_json_path=output_json_path,
            output_markdown_path=output_markdown_path,
            dry_run=dry_run,
        )
    raise ValueError(f"Unsupported generation metrics target: {target}")


def _validate_evaluation_config(config: TrainingConfig) -> None:
    if config.generation_metrics_samples <= 0:
        raise ValueError("samples must be positive")
    if config.generation_metrics_steps <= 0:
        raise ValueError("steps must be positive")
    if config.generation_metrics_guidance_scale < 0.0:
        raise ValueError("guidance-scale must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.precision not in {"float32", "bf16"}:
        raise ValueError("precision must be one of: bf16, float32")
