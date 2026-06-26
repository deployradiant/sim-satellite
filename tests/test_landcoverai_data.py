from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from sim_satellite.data.landcoverai import (
    _crop_box_for_tile,
    download_and_prepare_landcoverai,
    remap_landcoverai_mask,
)
from sim_satellite.data.loader import read_manifest
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.model import ModelConfig


def test_prepare_landcoverai_tiles_remaps_and_writes_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    scene = "N-00-00-A-a-0-0"
    _write_scene(
        data_dir / "extracted",
        scene=scene,
        mask_values=[1, 2, 3, 4],
    )
    (data_dir / "extracted" / "train.txt").write_text(f"{scene}_0\n", encoding="utf-8")

    result = download_and_prepare_landcoverai(
        data_dir,
        download=False,
        extract=False,
        validate_official_counts=False,
    )

    assert result.samples == 1
    assert result.prepared_samples == 1
    assert result.total_tiles == 1
    assert result.supervised_tiles == 1
    assert result.skipped_tiles == 0
    assert result.split_counts == {"train": 1}

    samples = read_manifest(result.manifest_path)
    assert len(samples) == 1
    assert samples[0].source == "landcoverai"
    assert samples[0].split == "train"
    assert samples[0].image_path == data_dir / "tiles" / "train" / "images" / f"{scene}_0.png"
    assert samples[0].mask_path == data_dir / "tiles" / "train" / "masks" / f"{scene}_0.png"
    assert samples[0].prepared_path.exists()

    with Image.open(samples[0].mask_path) as remapped:
        values = mask_to_class_ids(
            remapped.resize((2, 2), Image.Resampling.NEAREST),
            ModelConfig(mask_channels=9),
        )
    assert values == [8, 5, 6, 4]

    prepared = load_prepared_sample(samples[0].prepared_path, torch=torch)
    assert prepared.image.shape == (512, 512, 3)
    assert prepared.mask.shape == (512, 512)


def test_prepare_landcoverai_can_filter_sparse_tiles(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    kept_scene = "N-00-00-A-a-0-1"
    sparse_scene = "N-00-00-A-a-0-2"
    _write_scene(data_dir / "extracted", scene=kept_scene, mask_values=[2, 2, 2, 2])
    _write_scene(data_dir / "extracted", scene=sparse_scene, mask_values=[0, 0, 0, 4])
    (data_dir / "extracted" / "train.txt").write_text(
        f"{kept_scene}_0\n{sparse_scene}_0\n",
        encoding="utf-8",
    )

    result = download_and_prepare_landcoverai(
        data_dir,
        download=False,
        extract=False,
        min_labeled_fraction=0.5,
        validate_official_counts=False,
    )

    samples = read_manifest(result.manifest_path)
    assert result.total_tiles == 2
    assert result.supervised_tiles == 2
    assert result.skipped_tiles == 1
    assert [sample.image_path.name for sample in samples] == [f"{kept_scene}_0.png"]


def test_prepare_landcoverai_reports_missing_scene_tiles(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    (data_dir / "extracted").mkdir(parents=True)
    (data_dir / "extracted" / "train.txt").write_text("missing_0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing source files"):
        download_and_prepare_landcoverai(
            data_dir,
            download=False,
            extract=False,
            validate_official_counts=False,
        )


def test_prepare_landcoverai_rejects_non_official_split_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    scene = "N-00-00-A-a-0-0"
    _write_scene(data_dir / "extracted", scene=scene, mask_values=[1, 2, 3, 4])
    (data_dir / "extracted" / "train.txt").write_text(f"{scene}_0\n", encoding="utf-8")
    (data_dir / "extracted" / "val.txt").write_text("", encoding="utf-8")
    (data_dir / "extracted" / "test.txt").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unexpected LandCover.ai split counts"):
        download_and_prepare_landcoverai(data_dir, download=False, extract=False)


def test_prepare_landcoverai_rejects_out_of_bounds_split_tiles(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    scene = "N-00-00-A-a-0-0"
    _write_scene(data_dir / "extracted", scene=scene, mask_values=[1, 2, 3, 4])
    (data_dir / "extracted" / "train.txt").write_text(f"{scene}_1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside source image bounds"):
        download_and_prepare_landcoverai(
            data_dir,
            download=False,
            extract=False,
            validate_official_counts=False,
        )


def test_landcoverai_crop_box_uses_width_based_columns() -> None:
    assert _crop_box_for_tile((1024, 1024), 0) == (0, 0, 512, 512)
    assert _crop_box_for_tile((1024, 1024), 1) == (512, 0, 1024, 512)
    assert _crop_box_for_tile((1024, 1024), 2) == (0, 512, 512, 1024)
    assert _crop_box_for_tile((1024, 1024), 3) == (512, 512, 1024, 1024)


def test_remap_landcoverai_background_stays_unlabeled() -> None:
    mask = Image.new("L", (5, 1))
    mask.putdata([0, 1, 2, 3, 4])

    remapped = remap_landcoverai_mask(mask)

    assert mask_to_class_ids(remapped, ModelConfig(mask_channels=9)) == [0, 8, 5, 6, 4]


def test_prepare_landcoverai_rejects_unsafe_zip_members(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    archive_path = data_dir / "raw" / "landcover.ai.v1.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe zip member"):
        download_and_prepare_landcoverai(data_dir, download=False, extract=True)

    assert not (data_dir.parent / "escape.txt").exists()


def test_prepare_landcoverai_rejects_unmarked_cached_extraction(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    archive_path = data_dir / "raw" / "landcover.ai.v1.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train.txt", "")
    extracted_dir = data_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "train.txt").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without a valid completion marker"):
        download_and_prepare_landcoverai(data_dir, download=False, extract=True)


def _write_scene(root: Path, *, scene: str, mask_values: list[int]) -> None:
    image_path = root / "images" / f"{scene}.tif"
    mask_path = root / "masks" / f"{scene}.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (1, 2, 3)).save(image_path)
    mask = Image.new("L", (512, 512), mask_values[0])
    pixels = mask.load()
    quadrant_values = {
        (0, 0): mask_values[0],
        (1, 0): mask_values[1],
        (0, 1): mask_values[2],
        (1, 1): mask_values[3],
    }
    for y in range(512):
        for x in range(512):
            pixels[x, y] = quadrant_values[(x // 256, y // 256)]
    mask.save(mask_path)
