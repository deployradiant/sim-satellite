from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image

from sim_satellite.data.landcoverai import (
    download_and_prepare_landcoverai,
    remap_landcoverai_mask,
    scan_landcoverai,
)
from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_RGB, download_and_prepare_openearthmap
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.model import ModelConfig
from sim_satellite.training_data import center_square_crop_box, load_batch, preload_samples


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


def test_prepare_landcoverai_matches_openearth_training_batch_contract(tmp_path: Path) -> None:
    openearthmap_dir = tmp_path / "openearthmap"
    landcoverai_dir = tmp_path / "landcoverai"
    _write_openearth_scene(
        openearthmap_dir,
        mask_classes=["building", "tree", "water", "road"],
    )
    scene = "N-00-00-A-a-0-0"
    _write_scene(
        landcoverai_dir / "extracted",
        scene=scene,
        mask_values=[1, 2, 3, 4],
    )
    (landcoverai_dir / "extracted" / "train.txt").write_text(
        f"{scene}_0\n",
        encoding="utf-8",
    )

    openearthmap_result = download_and_prepare_openearthmap(
        openearthmap_dir,
        download=False,
        extract=False,
    )
    landcoverai_result = download_and_prepare_landcoverai(
        landcoverai_dir,
        download=False,
        extract=False,
        validate_official_counts=False,
    )

    openearthmap_sample = read_manifest(openearthmap_result.manifest_path)[0]
    landcoverai_sample = read_manifest(landcoverai_result.manifest_path)[0]
    openearthmap_prepared = load_prepared_sample(openearthmap_sample.prepared_path, torch=torch)
    landcoverai_prepared = load_prepared_sample(landcoverai_sample.prepared_path, torch=torch)

    assert openearthmap_prepared.image.shape == landcoverai_prepared.image.shape
    assert openearthmap_prepared.mask.shape == landcoverai_prepared.mask.shape
    assert openearthmap_prepared.image.dtype == landcoverai_prepared.image.dtype == torch.uint8
    assert openearthmap_prepared.mask.dtype == landcoverai_prepared.mask.dtype == torch.uint8
    assert torch.equal(openearthmap_prepared.image, landcoverai_prepared.image)
    assert torch.equal(openearthmap_prepared.mask, landcoverai_prepared.mask)

    config = ModelConfig(image_size=256, mask_channels=9)
    samples = (openearthmap_sample, landcoverai_sample)
    preloaded = preload_samples(samples)
    preloaded_samples = tuple(preloaded[sample] for sample in samples)
    crop_boxes = [
        center_square_crop_box(sample, crop_size=config.image_size) for sample in preloaded_samples
    ]
    images, masks = load_batch(
        preloaded_samples,
        crop_boxes,
        torch=torch,
        model_config=config,
        device=torch.device("cpu"),
    )

    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 9, 256, 256)
    assert images.dtype == masks.dtype == torch.float32
    assert torch.equal(images[0], images[1])
    assert torch.equal(masks[0], masks[1])
    assert torch.allclose(masks.sum(dim=1), torch.ones((2, 256, 256)))


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


def test_prepare_landcoverai_rejects_duplicate_split_tiles(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    scene = "N-00-00-A-a-0-0"
    _write_scene(data_dir / "extracted", scene=scene, mask_values=[1, 2, 3, 4])
    (data_dir / "extracted" / "train.txt").write_text(
        f"{scene}_0\n{scene}_0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Duplicate LandCover.ai split tile ids"):
        download_and_prepare_landcoverai(
            data_dir,
            download=False,
            extract=False,
            validate_official_counts=False,
        )


def test_prepare_landcoverai_rejects_invalid_split_tile_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "landcoverai"
    (data_dir / "extracted").mkdir(parents=True)
    (data_dir / "extracted" / "train.txt").write_text("not-a-tile-id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid LandCover.ai split tile id"):
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


def test_scan_landcoverai_opens_each_scene_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted_dir = tmp_path / "extracted"
    scene = "N-00-00-A-a-0-0"
    _write_scene(
        extracted_dir,
        scene=scene,
        mask_values=[1, 2, 3, 4],
        size=(1024, 512),
    )
    (extracted_dir / "train.txt").write_text(f"{scene}_0\n{scene}_1\n", encoding="utf-8")

    opened_tifs: Counter[tuple[str, str]] = Counter()
    original_open = Image.open

    def counted_open(path: str | Path, *args: object, **kwargs: object) -> Image.Image:
        path = Path(path)
        if path.suffix == ".tif":
            opened_tifs[(path.parent.name, path.name)] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("sim_satellite.data.landcoverai.Image.open", counted_open)

    scan = scan_landcoverai(
        extracted_dir,
        tile_dir=tmp_path / "tiles",
        validate_official_counts=False,
    )

    assert scan.total_tiles == 2
    assert opened_tifs == Counter({("images", f"{scene}.tif"): 1, ("masks", f"{scene}.tif"): 1})


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


def _write_scene(
    root: Path,
    *,
    scene: str,
    mask_values: list[int],
    size: tuple[int, int] = (512, 512),
) -> None:
    image_path = root / "images" / f"{scene}.tif"
    mask_path = root / "masks" / f"{scene}.tif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (1, 2, 3)).save(image_path)
    mask = Image.new("L", size, mask_values[0])
    pixels = mask.load()
    quadrant_values = {
        (0, 0): mask_values[0],
        (1, 0): mask_values[1],
        (0, 1): mask_values[2],
        (1, 1): mask_values[3],
    }
    half_width = size[0] // 2
    half_height = size[1] // 2
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = quadrant_values[(x // half_width, y // half_height)]
    mask.save(mask_path)


def _write_openearth_scene(
    root: Path,
    *,
    mask_classes: list[str],
    size: tuple[int, int] = (512, 512),
) -> None:
    image_path = root / "extracted" / "train" / "images" / "tile.png"
    mask_path = root / "extracted" / "train" / "labels" / "tile.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (1, 2, 3)).save(image_path)
    mask = Image.new("RGB", size, OPENEARTHMAP_RGB[mask_classes[0]])
    pixels = mask.load()
    quadrant_values = {
        (0, 0): OPENEARTHMAP_RGB[mask_classes[0]],
        (1, 0): OPENEARTHMAP_RGB[mask_classes[1]],
        (0, 1): OPENEARTHMAP_RGB[mask_classes[2]],
        (1, 1): OPENEARTHMAP_RGB[mask_classes[3]],
    }
    half_width = size[0] // 2
    half_height = size[1] // 2
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = quadrant_values[(x // half_width, y // half_height)]
    mask.save(mask_path)
