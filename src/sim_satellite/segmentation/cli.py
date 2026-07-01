"""Click commands for segmentation workflows."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from sim_satellite.segmentation.config import (
    DEFAULT_SEGMENTATION_CHECKPOINT_PATH,
    DEFAULT_SEGMENTATION_PREDICTION_DIR,
    SegmentationEvaluationConfig,
    SegmentationPredictionConfig,
    load_segmentation_training_config_file,
)
from sim_satellite.segmentation.inference import (
    evaluate_segmentation_model,
    predict_segmentation_masks,
)
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH


def _run_evaluation_with_target(
    config: SegmentationEvaluationConfig,
    target: str,
) -> dict[str, Any]:
    if target == "local":
        result = asdict(evaluate_segmentation_model(config))
        result["execution_target"] = "local"
        return result

    if target != "modal":
        raise ValueError(f"Unsupported segmentation evaluation target: {target}")

    from sim_satellite.modal_app import run_segmentation_evaluation_target

    return run_segmentation_evaluation_target(config)


@click.command("segmentation-train", help="Train the VAE-fused segmentation classifier.")
@click.argument("config_path", type=click.Path(path_type=Path))
def train_cli(config_path: Path) -> None:
    try:
        from sim_satellite.modal_app import run_segmentation_training_target

        result = run_segmentation_training_target(
            load_segmentation_training_config_file(config_path)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@click.command("segmentation-evaluate", help="Evaluate a VAE-fused segmentation checkpoint.")
@click.option(
    "--target",
    type=click.Choice(["modal", "local"]),
    default="local",
    show_default=True,
)
@click.option("--checkpoint", "checkpoint_path", default=DEFAULT_SEGMENTATION_CHECKPOINT_PATH)
@click.option("--manifest", "manifest_path", default=DEFAULT_MANIFEST_PATH)
@click.option("--split", default="train")
@click.option("--max-samples", type=int, default=128, show_default=True)
@click.option("--output", "output_path", default="")
@click.option("--device", default="cpu", show_default=True)
def evaluate_cli(
    checkpoint_path: str,
    manifest_path: str,
    split: str,
    max_samples: int,
    output_path: str,
    device: str,
    target: str,
) -> None:
    try:
        result = _run_evaluation_with_target(
            SegmentationEvaluationConfig(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                split=split,
                max_samples=max_samples,
                output_path=output_path,
                device=device,
            ),
            target,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@click.command("segmentation-predict", help="Predict semantic masks for unlabeled images.")
@click.option("--checkpoint", "checkpoint_path", default=DEFAULT_SEGMENTATION_CHECKPOINT_PATH)
@click.option("--output-dir", default=DEFAULT_SEGMENTATION_PREDICTION_DIR)
@click.option("--images-manifest", "images_manifest_path", default="")
@click.option("--images-dir", default="")
@click.option("--max-images", type=int, default=0, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.0, show_default=True)
@click.option("--confidence-masks/--no-confidence-masks", default=True, show_default=True)
@click.option("--color-masks/--no-color-masks", default=True, show_default=True)
def predict_cli(
    checkpoint_path: str,
    output_dir: str,
    images_manifest_path: str,
    images_dir: str,
    max_images: int,
    device: str,
    confidence_threshold: float,
    confidence_masks: bool,
    color_masks: bool,
) -> None:
    try:
        result = predict_segmentation_masks(
            SegmentationPredictionConfig(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                images_manifest_path=images_manifest_path,
                images_dir=images_dir,
                max_images=max_images,
                device=device,
                confidence_threshold=confidence_threshold,
                save_confidence_masks=confidence_masks,
                save_color_masks=color_masks,
            ),
            progress=click.echo,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
