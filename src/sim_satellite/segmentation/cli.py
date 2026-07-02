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
    DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE,
    SegmentationEvaluationConfig,
    SegmentationPredictionConfig,
    load_segmentation_training_config_file,
)
from sim_satellite.segmentation.inference import (
    evaluate_segmentation_model,
    predict_segmentation_masks,
)
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

PRODUCTION_OPENEARTH_PRESET_NAME = "production-openearth-v1"
OPENEARTH_INFERENCE_PRESET_NAMES = (PRODUCTION_OPENEARTH_PRESET_NAME,)
PRODUCTION_OPENEARTH_CHECKPOINT_PATH = (
    "artifacts/segmentation-openearthmap-smp-unet-efficientnet-b4-scse-official-xbd-full-polish-v1/"
    "smp-unet-efficientnet-b4-scse-official-xbd-full-polish-s2500.pt"
)
PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES = {
    "agriculture land": -0.23,
    "bareland": -0.9,
    "building": 0.2,
    "developed space": -0.02,
    "rangeland": 0.05,
    "road": -0.2,
    "tree": -0.02,
    "water": -0.8,
}
PRODUCTION_OPENEARTH_CONFIDENCE_THRESHOLD = 0.8
PRODUCTION_OPENEARTH_TILE_SCALES = (0.75, 1.0, 1.25)
PRODUCTION_OPENEARTH_TILE_STRIDE = 256


def _class_float_map_from_json(value: str, *, option_name: str) -> dict[str, float]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{option_name} JSON must be an object")
    return {str(class_name): float(weight) for class_name, weight in payload.items()}


def _validate_segmentation_inference_preset(value: str) -> str:
    if not value:
        return ""
    if value not in OPENEARTH_INFERENCE_PRESET_NAMES:
        raise ValueError(
            "segmentation inference preset must be one of: "
            + ", ".join(OPENEARTH_INFERENCE_PRESET_NAMES)
        )
    return value


def _segmentation_preset_checkpoint_path(preset: str, checkpoint_path: str) -> str:
    return PRODUCTION_OPENEARTH_CHECKPOINT_PATH if preset else checkpoint_path


def _segmentation_preset_class_logit_biases(
    preset: str,
    class_logit_biases: dict[str, float],
) -> dict[str, float]:
    return dict(PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES) if preset else class_logit_biases


def _segmentation_preset_confidence_threshold(preset: str, confidence_threshold: float) -> float:
    return PRODUCTION_OPENEARTH_CONFIDENCE_THRESHOLD if preset else confidence_threshold


def _segmentation_preset_tile_stride(preset: str, tile_stride: int) -> int:
    return PRODUCTION_OPENEARTH_TILE_STRIDE if preset else tile_stride


def _segmentation_preset_tile_scales(
    preset: str,
    tile_scales: tuple[float, ...],
) -> tuple[float, ...]:
    return PRODUCTION_OPENEARTH_TILE_SCALES if preset else tile_scales


def _segmentation_preset_feather_blending(preset: str, feather_blending: bool) -> bool:
    return True if preset else feather_blending


def _segmentation_preset_output_dir(preset: str, output_dir: str) -> str:
    if preset and output_dir == DEFAULT_SEGMENTATION_PREDICTION_DIR:
        return str(Path(DEFAULT_SEGMENTATION_PREDICTION_DIR) / preset)
    return output_dir


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


def _run_prediction_with_target(
    config: SegmentationPredictionConfig,
    target: str,
) -> dict[str, Any]:
    if target == "local":
        result = asdict(predict_segmentation_masks(config, progress=click.echo))
        result["execution_target"] = "local"
        return result

    if target != "modal":
        raise ValueError(f"Unsupported segmentation prediction target: {target}")

    from sim_satellite.modal_app import run_segmentation_prediction_target

    return run_segmentation_prediction_target(config)


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
@click.option(
    "--class-logit-bias-json",
    default="",
    help="Optional JSON object mapping OpenEarthMap class names to post-logit biases.",
)
@click.option("--manifest", "manifest_path", default=DEFAULT_MANIFEST_PATH)
@click.option("--split", default="train")
@click.option("--max-samples", type=int, default=128, show_default=True)
@click.option(
    "--tile-stride",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--tile-scale",
    "tile_scales",
    type=float,
    multiple=True,
    help="Native tile scale to average at inference; repeat for multi-scale inference.",
)
@click.option(
    "--feather-blending/--uniform-blending",
    default=False,
    show_default=True,
    help="Taper overlapping tile logits before averaging.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Report selected-pixel metrics for predictions at or above this confidence.",
)
@click.option(
    "--inference-preset",
    default="",
    help=(
        "Apply a named segmentation inference preset. Choices: "
        + ", ".join(OPENEARTH_INFERENCE_PRESET_NAMES)
    ),
)
@click.option("--output", "output_path", default="")
@click.option("--device", default="cpu", show_default=True)
def evaluate_cli(
    checkpoint_path: str,
    class_logit_bias_json: str,
    manifest_path: str,
    split: str,
    max_samples: int,
    tile_stride: int,
    tile_scales: tuple[float, ...],
    feather_blending: bool,
    confidence_threshold: float,
    inference_preset: str,
    output_path: str,
    device: str,
    target: str,
) -> None:
    try:
        inference_preset = _validate_segmentation_inference_preset(inference_preset)
        class_logit_biases = _class_float_map_from_json(
            class_logit_bias_json,
            option_name="class logit bias",
        )
        checkpoint_path = _segmentation_preset_checkpoint_path(
            inference_preset,
            checkpoint_path,
        )
        class_logit_biases = _segmentation_preset_class_logit_biases(
            inference_preset,
            class_logit_biases,
        )
        confidence_threshold = _segmentation_preset_confidence_threshold(
            inference_preset,
            confidence_threshold,
        )
        tile_stride = _segmentation_preset_tile_stride(inference_preset, tile_stride)
        tile_scales = _segmentation_preset_tile_scales(inference_preset, tile_scales)
        feather_blending = _segmentation_preset_feather_blending(
            inference_preset,
            feather_blending,
        )
        result = _run_evaluation_with_target(
            SegmentationEvaluationConfig(
                checkpoint_path=checkpoint_path,
                class_logit_biases=class_logit_biases,
                manifest_path=manifest_path,
                split=split,
                max_samples=max_samples,
                tile_stride=tile_stride,
                tile_scales=tile_scales,
                feather_blending=feather_blending,
                confidence_threshold=confidence_threshold,
                output_path=output_path,
                device=device,
            ),
            target,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@click.command("segmentation-predict", help="Predict semantic masks for unlabeled images.")
@click.option(
    "--target",
    type=click.Choice(["modal", "local"]),
    default="local",
    show_default=True,
)
@click.option("--checkpoint", "checkpoint_path", default=DEFAULT_SEGMENTATION_CHECKPOINT_PATH)
@click.option(
    "--class-logit-bias-json",
    default="",
    help="Optional JSON object mapping OpenEarthMap class names to post-logit biases.",
)
@click.option("--output-dir", default=DEFAULT_SEGMENTATION_PREDICTION_DIR)
@click.option("--images-manifest", "images_manifest_path", default="")
@click.option("--images-dir", default="")
@click.option("--max-images", type=int, default=0, show_default=True)
@click.option(
    "--tile-stride",
    type=int,
    default=DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE,
    show_default=True,
)
@click.option(
    "--tile-scale",
    "tile_scales",
    type=float,
    multiple=True,
    help="Native tile scale to average at inference; repeat for multi-scale inference.",
)
@click.option(
    "--feather-blending/--uniform-blending",
    default=False,
    show_default=True,
    help="Taper overlapping tile logits before averaging.",
)
@click.option(
    "--inference-preset",
    default="",
    help=(
        "Apply a named segmentation inference preset. Choices: "
        + ", ".join(OPENEARTH_INFERENCE_PRESET_NAMES)
    ),
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.0, show_default=True)
@click.option("--confidence-masks/--no-confidence-masks", default=True, show_default=True)
@click.option("--color-masks/--no-color-masks", default=True, show_default=True)
def predict_cli(
    checkpoint_path: str,
    class_logit_bias_json: str,
    output_dir: str,
    images_manifest_path: str,
    images_dir: str,
    max_images: int,
    tile_stride: int,
    tile_scales: tuple[float, ...],
    feather_blending: bool,
    inference_preset: str,
    device: str,
    confidence_threshold: float,
    confidence_masks: bool,
    color_masks: bool,
    target: str,
) -> None:
    try:
        inference_preset = _validate_segmentation_inference_preset(inference_preset)
        class_logit_biases = _class_float_map_from_json(
            class_logit_bias_json,
            option_name="class logit bias",
        )
        checkpoint_path = _segmentation_preset_checkpoint_path(
            inference_preset,
            checkpoint_path,
        )
        class_logit_biases = _segmentation_preset_class_logit_biases(
            inference_preset,
            class_logit_biases,
        )
        confidence_threshold = _segmentation_preset_confidence_threshold(
            inference_preset,
            confidence_threshold,
        )
        tile_stride = _segmentation_preset_tile_stride(inference_preset, tile_stride)
        tile_scales = _segmentation_preset_tile_scales(inference_preset, tile_scales)
        feather_blending = _segmentation_preset_feather_blending(
            inference_preset,
            feather_blending,
        )
        output_dir = _segmentation_preset_output_dir(inference_preset, output_dir)
        result = _run_prediction_with_target(
            SegmentationPredictionConfig(
                checkpoint_path=checkpoint_path,
                class_logit_biases=class_logit_biases,
                output_dir=output_dir,
                images_manifest_path=images_manifest_path,
                images_dir=images_dir,
                max_images=max_images,
                tile_stride=tile_stride,
                tile_scales=tile_scales,
                feather_blending=feather_blending,
                device=device,
                confidence_threshold=confidence_threshold,
                save_confidence_masks=confidence_masks,
                save_color_masks=color_masks,
            ),
            target,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, sort_keys=True))
