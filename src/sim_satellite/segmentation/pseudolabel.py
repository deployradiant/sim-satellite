"""Pseudo-label OpenEarthMap images and compare segmentation training runs."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import (
    OpenEarthMapImage,
    download_and_prepare_openearthmap,
    scan_openearthmap,
)
from sim_satellite.data.segmentation import (
    SegmentationPreparationConfig,
    prepare_segmentation_manifest,
)
from sim_satellite.segmentation.config import (
    DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH,
    SegmentationPredictionConfig,
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
)
from sim_satellite.segmentation.inference import predict_segmentation_masks
from sim_satellite.segmentation.training import train_segmentation_model
from sim_satellite.segmentation.utils import ProgressReporter, _report

DEFAULT_OPENEARTHMAP_PSEUDO_LABELER_CHECKPOINT_PATH = (
    "artifacts/segmentation-study-2026-06-28-200k/"
    "deeplab_unet_vae200k_fused_noaug_lr3e-4_alldata_s200000_val256.pt"
)
DEFAULT_OPENEARTHMAP_PSEUDO_STUDY_DATA_DIR = "data/openearthmap/pseudo-label-study"
DEFAULT_OPENEARTHMAP_PSEUDO_STUDY_ARTIFACT_DIR = "artifacts/openearthmap-pseudo-label-study"
DEFAULT_OPENEARTHMAP_PSEUDO_BASELINE_CHECKPOINT_PATH = (
    "artifacts/openearthmap-pseudo-label-study/labelled-only-vae-aspp-unet-200k.pt"
)
DEFAULT_OPENEARTHMAP_PSEUDO_FULL_CHECKPOINT_PATH = (
    "artifacts/openearthmap-pseudo-label-study/full-pseudo-labelled-vae-aspp-unet-200k.pt"
)


@dataclass(frozen=True)
class OpenEarthMapPseudoLabelStudyConfig:
    openearthmap_data_dir: str = "data/openearthmap"
    output_data_dir: str = DEFAULT_OPENEARTHMAP_PSEUDO_STUDY_DATA_DIR
    artifact_dir: str = DEFAULT_OPENEARTHMAP_PSEUDO_STUDY_ARTIFACT_DIR
    classifier_checkpoint_path: str = DEFAULT_OPENEARTHMAP_PSEUDO_LABELER_CHECKPOINT_PATH
    baseline_checkpoint_path: str = DEFAULT_OPENEARTHMAP_PSEUDO_BASELINE_CHECKPOINT_PATH
    full_checkpoint_path: str = DEFAULT_OPENEARTHMAP_PSEUDO_FULL_CHECKPOINT_PATH
    validation_samples: int = 256
    max_pseudo_images: int = 0
    seed: int = 2026
    force_prepared: bool = False
    save_color_masks: bool = False
    training: SegmentationTrainingConfig = field(default_factory=lambda: _default_training_config())

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> OpenEarthMapPseudoLabelStudyConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(
                f"Unknown OpenEarthMap pseudo-label study config fields: {', '.join(unknown)}"
            )

        raw_training = values.get("training")
        if raw_training is not None and not isinstance(raw_training, Mapping):
            raise ValueError("training must be an object when provided")

        config_values = {
            name: values[name] for name in names if name in values and name != "training"
        }
        config = cls(**config_values)
        if raw_training is None:
            return config
        training_values = asdict(config.training)
        training_values.update(raw_training)
        return replace(
            config,
            training=SegmentationTrainingConfig.from_dict(training_values),
        )


@dataclass(frozen=True)
class OpenEarthMapPseudoComparisonManifests:
    baseline_manifest_path: str
    full_manifest_path: str
    labelled_source_samples: int
    labelled_train_samples: int
    labelled_validation_samples: int
    pseudo_train_samples: int
    baseline_samples: int
    full_samples: int


@dataclass(frozen=True)
class OpenEarthMapPseudoLabelStudyResult:
    classifier_checkpoint_path: str
    output_data_dir: str
    artifact_dir: str
    unlabeled_images_manifest_path: str
    pseudo_raw_manifest_path: str
    pseudo_prepared_manifest_path: str
    comparison_path: str
    comparison_plot_path: str
    openearthmap_labelled_samples: int
    openearthmap_unlabeled_images: int
    pseudo_labelled_images: int
    manifests: OpenEarthMapPseudoComparisonManifests
    baseline: SegmentationTrainingResult
    full: SegmentationTrainingResult
    delta_best_mean_iou: float
    delta_final_mean_iou: float
    delta_best_pixel_accuracy: float
    delta_final_pixel_accuracy: float


def load_openearthmap_pseudolabel_study_config_file(
    config_path: Path,
) -> OpenEarthMapPseudoLabelStudyConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("OpenEarthMap pseudo-label study config JSON must be an object")
    return OpenEarthMapPseudoLabelStudyConfig.from_dict(values)


def run_openearthmap_pseudolabel_study(
    config: OpenEarthMapPseudoLabelStudyConfig,
    *,
    progress: ProgressReporter | None = None,
    after_train: Callable[[str, SegmentationTrainingResult], None] | None = None,
) -> OpenEarthMapPseudoLabelStudyResult:
    _validate_study_config(config)
    data_dir = Path(config.openearthmap_data_dir)
    output_data_dir = Path(config.output_data_dir)
    artifact_dir = Path(config.artifact_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _report(progress, f"Preparing labelled OpenEarthMap data under {data_dir}")
    openearthmap = download_and_prepare_openearthmap(
        data_dir,
        force_prepared=config.force_prepared,
        progress=progress,
    )

    _report(progress, "Scanning OpenEarthMap images without masks")
    scan = scan_openearthmap(openearthmap.extracted_dir, progress=progress)
    pseudo_images = tuple(sorted(scan.unlabeled_images, key=lambda item: str(item.image_path)))
    if config.max_pseudo_images > 0:
        pseudo_images = pseudo_images[: config.max_pseudo_images]
    if not pseudo_images:
        raise RuntimeError(
            "OpenEarthMap scan found no unlabeled images to pseudo-label. "
            "The full-data study needs images without masks."
        )

    unlabeled_images_manifest_path = output_data_dir / "unlabeled-images.jsonl"
    _write_unlabeled_image_manifest(pseudo_images, unlabeled_images_manifest_path)
    _report(
        progress,
        f"Wrote {len(pseudo_images)} unlabeled image records to {unlabeled_images_manifest_path}",
    )

    prediction_result = predict_segmentation_masks(
        SegmentationPredictionConfig(
            checkpoint_path=config.classifier_checkpoint_path,
            output_dir=str(output_data_dir / "pseudo-labels"),
            images_manifest_path=str(unlabeled_images_manifest_path),
            max_images=0,
            device=config.training.device,
            save_color_masks=config.save_color_masks,
        ),
        progress=progress,
    )
    pseudo_prepared_manifest_path = output_data_dir / "pseudo-prepared-manifest.jsonl"
    preparation_result = prepare_segmentation_manifest(
        SegmentationPreparationConfig(
            raw_manifest_path=prediction_result.manifest_path,
            output_manifest_path=str(pseudo_prepared_manifest_path),
            prepared_dir=str(output_data_dir / "prepared"),
            source="openearthmap-pseudo",
            force_prepared=config.force_prepared,
        ),
        progress=progress,
    )

    manifests = write_openearthmap_pseudolabel_comparison_manifests(
        labelled_manifest_path=openearthmap.manifest_path,
        pseudo_manifest_path=Path(preparation_result.output_manifest_path),
        baseline_manifest_path=output_data_dir / "labelled-only-manifest.jsonl",
        full_manifest_path=output_data_dir / "full-pseudo-labelled-manifest.jsonl",
        validation_samples=config.validation_samples,
        seed=config.seed,
    )

    baseline_config = _training_config_for_manifest(
        config,
        manifest_path=manifests.baseline_manifest_path,
        checkpoint_path=config.baseline_checkpoint_path,
    )
    full_config = _training_config_for_manifest(
        config,
        manifest_path=manifests.full_manifest_path,
        checkpoint_path=config.full_checkpoint_path,
    )

    _report(progress, "Training labelled-only OpenEarthMap baseline")
    baseline_result = train_segmentation_model(baseline_config, progress=progress)
    if after_train is not None:
        after_train("baseline", baseline_result)

    _report(progress, "Training OpenEarthMap labelled plus pseudo-labelled full-data arm")
    full_result = train_segmentation_model(full_config, progress=progress)
    if after_train is not None:
        after_train("full", full_result)

    comparison_path = artifact_dir / "comparison.json"
    comparison_plot_path = artifact_dir / "comparison.png"
    result = OpenEarthMapPseudoLabelStudyResult(
        classifier_checkpoint_path=config.classifier_checkpoint_path,
        output_data_dir=str(output_data_dir),
        artifact_dir=str(artifact_dir),
        unlabeled_images_manifest_path=str(unlabeled_images_manifest_path),
        pseudo_raw_manifest_path=prediction_result.manifest_path,
        pseudo_prepared_manifest_path=preparation_result.output_manifest_path,
        comparison_path=str(comparison_path),
        comparison_plot_path=str(comparison_plot_path),
        openearthmap_labelled_samples=openearthmap.samples,
        openearthmap_unlabeled_images=len(scan.unlabeled_images),
        pseudo_labelled_images=preparation_result.samples,
        manifests=manifests,
        baseline=baseline_result,
        full=full_result,
        delta_best_mean_iou=full_result.best_mean_iou - baseline_result.best_mean_iou,
        delta_final_mean_iou=full_result.final_mean_iou - baseline_result.final_mean_iou,
        delta_best_pixel_accuracy=(
            full_result.best_pixel_accuracy - baseline_result.best_pixel_accuracy
        ),
        delta_final_pixel_accuracy=(
            full_result.final_pixel_accuracy - baseline_result.final_pixel_accuracy
        ),
    )
    comparison_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_comparison_plot(result, comparison_plot_path)
    return result


def write_openearthmap_pseudolabel_comparison_manifests(
    *,
    labelled_manifest_path: str | Path,
    pseudo_manifest_path: str | Path,
    baseline_manifest_path: str | Path,
    full_manifest_path: str | Path,
    validation_samples: int,
    seed: int,
) -> OpenEarthMapPseudoComparisonManifests:
    labelled_source_samples = read_manifest(labelled_manifest_path)
    labelled_samples = [sample for sample in labelled_source_samples if sample.split == "train"]
    if not labelled_samples:
        raise RuntimeError(f"Labelled manifest has no train samples: {labelled_manifest_path}")
    if validation_samples <= 0 or validation_samples >= len(labelled_samples):
        raise RuntimeError(f"validation_samples must be between 1 and {len(labelled_samples) - 1}")

    pseudo_source_samples = read_manifest(pseudo_manifest_path)
    pseudo_train_samples = [
        _with_split(sample, "train") for sample in pseudo_source_samples if sample.split == "train"
    ]
    if not pseudo_train_samples:
        raise RuntimeError(f"Pseudo-label manifest has no train samples: {pseudo_manifest_path}")

    shuffled_labelled = list(labelled_samples)
    random.Random(seed).shuffle(shuffled_labelled)
    labelled_validation = tuple(shuffled_labelled[:validation_samples])
    labelled_train = tuple(shuffled_labelled[validation_samples:])

    baseline_samples = [
        _with_split(sample, "train")
        for sample in sorted(labelled_train, key=lambda item: str(item.image_path))
    ]
    baseline_samples.extend(
        _with_split(sample, "val")
        for sample in sorted(labelled_validation, key=lambda item: str(item.image_path))
    )

    full_samples = list(baseline_samples[: len(labelled_train)])
    full_samples.extend(sorted(pseudo_train_samples, key=lambda item: str(item.image_path)))
    full_samples.extend(baseline_samples[len(labelled_train) :])

    write_manifest(baseline_samples, baseline_manifest_path)
    write_manifest(full_samples, full_manifest_path)

    return OpenEarthMapPseudoComparisonManifests(
        baseline_manifest_path=str(baseline_manifest_path),
        full_manifest_path=str(full_manifest_path),
        labelled_source_samples=len(labelled_source_samples),
        labelled_train_samples=len(labelled_train),
        labelled_validation_samples=len(labelled_validation),
        pseudo_train_samples=len(pseudo_train_samples),
        baseline_samples=len(baseline_samples),
        full_samples=len(full_samples),
    )


def _default_training_config() -> SegmentationTrainingConfig:
    return SegmentationTrainingConfig(
        split="train",
        validation_split="val",
        validation_fraction=0.2,
        max_validation_samples=256,
        steps=200000,
        batch_size=8,
        image_size=256,
        vae_checkpoint_path=DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH,
        learning_rate=3e-4,
        eval_every=10000,
        cache_samples=256,
        augment=False,
    )


def _training_config_for_manifest(
    config: OpenEarthMapPseudoLabelStudyConfig,
    *,
    manifest_path: str,
    checkpoint_path: str,
) -> SegmentationTrainingConfig:
    return replace(
        config.training,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        split="train",
        validation_split="val",
        max_validation_samples=config.validation_samples,
        seed=config.seed,
    )


def _write_unlabeled_image_manifest(
    images: Sequence[OpenEarthMapImage],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for image in images:
            json.dump(
                {
                    "image": str(image.image_path),
                    "source": "openearthmap-pseudo",
                    "split": "train",
                    "openearthmap_split": image.split,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")


def _with_split(sample: SatelliteSample, split: str) -> SatelliteSample:
    return SatelliteSample(
        image_path=sample.image_path,
        mask_path=sample.mask_path,
        prepared_path=sample.prepared_path,
        split=split,
        source=sample.source,
    )


def _write_comparison_plot(
    result: OpenEarthMapPseudoLabelStudyResult,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = ("Best mIoU", "Final mIoU", "Best pixel acc", "Final pixel acc")
    baseline_values = (
        result.baseline.best_mean_iou,
        result.baseline.final_mean_iou,
        result.baseline.best_pixel_accuracy,
        result.baseline.final_pixel_accuracy,
    )
    full_values = (
        result.full.best_mean_iou,
        result.full.final_mean_iou,
        result.full.best_pixel_accuracy,
        result.full.final_pixel_accuracy,
    )
    positions = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [position - width / 2 for position in positions],
        baseline_values,
        width,
        label="Labelled only",
        color="#4C78A8",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        full_values,
        width,
        label="Labelled + pseudo",
        color="#F58518",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("OpenEarthMap segmentation comparison")
    ax.set_xticks(list(positions), labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validate_study_config(config: OpenEarthMapPseudoLabelStudyConfig) -> None:
    required_paths = (
        "openearthmap_data_dir",
        "output_data_dir",
        "artifact_dir",
        "classifier_checkpoint_path",
        "baseline_checkpoint_path",
        "full_checkpoint_path",
    )
    for name in required_paths:
        if not getattr(config, name):
            raise ValueError(f"{name} is required")
    if config.validation_samples < 1:
        raise ValueError("validation_samples must be positive")
    if config.max_pseudo_images < 0:
        raise ValueError("max_pseudo_images must be non-negative")
