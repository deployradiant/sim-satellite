from __future__ import annotations

from pathlib import Path

from PIL import Image

from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import scan_openearthmap
from sim_satellite.segmentation.pseudolabel import (
    OpenEarthMapPseudoLabelStudyConfig,
    write_openearthmap_pseudolabel_comparison_manifests,
)


def test_openearthmap_scan_retains_images_without_masks(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    labelled_image_dir = extracted_dir / "train" / "labelled-city" / "images"
    labelled_mask_dir = extracted_dir / "train" / "labelled-city" / "labels"
    _write_rgb_fixture(labelled_image_dir / "with-mask.png")
    _write_rgb_fixture(labelled_image_dir / "without-mask.png")
    _write_mask_fixture(labelled_mask_dir / "with-mask.png")

    unlabeled_image_dir = extracted_dir / "train" / "unlabelled-city" / "images"
    _write_rgb_fixture(unlabeled_image_dir / "no-label-dir.png")

    scan = scan_openearthmap(extracted_dir)

    assert len(scan.samples) == 1
    assert scan.scanned_images == 3
    assert scan.missing_masks == 1
    assert scan.image_dirs_without_labels == 1
    assert [image.image_path.name for image in scan.unlabeled_images] == [
        "without-mask.png",
        "no-label-dir.png",
    ]
    assert {image.split for image in scan.unlabeled_images} == {"train"}


def test_pseudolabel_comparison_manifests_hold_out_same_labelled_validation(
    tmp_path: Path,
) -> None:
    labelled_manifest = tmp_path / "labelled.jsonl"
    pseudo_manifest = tmp_path / "pseudo.jsonl"
    write_manifest(
        [
            _sample(tmp_path, "labelled", index, source="openearthmap", split="train")
            for index in range(5)
        ],
        labelled_manifest,
    )
    write_manifest(
        [
            _sample(tmp_path, "pseudo", index, source="openearthmap-pseudo", split="train")
            for index in range(2)
        ],
        pseudo_manifest,
    )

    result = write_openearthmap_pseudolabel_comparison_manifests(
        labelled_manifest_path=labelled_manifest,
        pseudo_manifest_path=pseudo_manifest,
        baseline_manifest_path=tmp_path / "baseline.jsonl",
        full_manifest_path=tmp_path / "full.jsonl",
        validation_samples=2,
        seed=123,
    )

    baseline_samples = read_manifest(result.baseline_manifest_path)
    full_samples = read_manifest(result.full_manifest_path)
    baseline_validation = {
        sample.image_path.name for sample in baseline_samples if sample.split == "val"
    }
    full_validation = {sample.image_path.name for sample in full_samples if sample.split == "val"}

    assert result.labelled_train_samples == 3
    assert result.labelled_validation_samples == 2
    assert result.pseudo_train_samples == 2
    assert result.baseline_samples == 5
    assert result.full_samples == 7
    assert baseline_validation == full_validation
    assert len([sample for sample in full_samples if sample.source == "openearthmap-pseudo"]) == 2


def test_pseudolabel_study_training_overrides_keep_study_defaults() -> None:
    config = OpenEarthMapPseudoLabelStudyConfig.from_dict(
        {
            "validation_samples": 32,
            "training": {
                "steps": 10,
            },
        }
    )

    assert config.validation_samples == 32
    assert config.training.steps == 10
    assert config.training.image_size == 256
    assert config.training.augment is False
    assert config.training.validation_split == "val"


def _sample(
    tmp_path: Path,
    prefix: str,
    index: int,
    *,
    source: str,
    split: str,
) -> SatelliteSample:
    return SatelliteSample(
        image_path=tmp_path / prefix / f"image-{index}.png",
        mask_path=tmp_path / prefix / f"mask-{index}.png",
        prepared_path=tmp_path / prefix / f"prepared-{index}.pt",
        split=split,
        source=source,
    )


def _write_rgb_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (32, 64, 96)).save(path)


def _write_mask_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), 4).save(path)
