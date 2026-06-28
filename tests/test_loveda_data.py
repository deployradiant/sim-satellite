from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.loveda import download_and_prepare_loveda
from sim_satellite.data.openearthmap import OPENEARTHMAP_RGB, download_and_prepare_openearthmap
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.model import ModelConfig
from sim_satellite.training_data import center_square_crop_box, load_batch, preload_samples


def test_prepare_loveda_remaps_masks_and_writes_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    image_path = data_dir / "extracted" / "Train" / "Urban" / "images_png" / "tile.png"
    mask_path = data_dir / "extracted" / "Train" / "Urban" / "masks_png" / "tile.png"
    _write_rgb_image(image_path)
    _write_loveda_mask(mask_path, [2, 3, 6, 7])

    result = download_and_prepare_loveda(data_dir, download=False, extract=False)

    assert result.samples == 1
    assert result.prepared_samples == 1
    assert result.scanned_images == 1
    assert result.missing_masks == 0
    assert result.split_counts == {"train": 1}

    samples = read_manifest(result.manifest_path)
    assert len(samples) == 1
    assert samples[0].image_path == image_path
    assert samples[0].source == "loveda"
    assert samples[0].split == "train"
    assert samples[0].prepared_path.exists()
    assert (
        samples[0].mask_path == data_dir / "remapped" / "Train" / "Urban" / "masks_png" / "tile.png"
    )

    with Image.open(samples[0].mask_path) as remapped:
        assert mask_to_class_ids(remapped, ModelConfig(mask_channels=9)) == [8, 4, 5, 7]

    prepared = load_prepared_sample(samples[0].prepared_path, torch=torch)
    assert prepared.image.shape == (2, 2, 3)
    assert prepared.mask.tolist() == [[8, 4], [5, 7]]


def test_prepare_loveda_matches_openearth_training_batch_contract(tmp_path: Path) -> None:
    openearthmap_dir = tmp_path / "openearthmap"
    loveda_dir = tmp_path / "loveda"
    _write_openearth_scene(
        openearthmap_dir,
        mask_classes=["building", "road", "water", "agriculture land"],
    )
    image_path = loveda_dir / "extracted" / "Train" / "Urban" / "images_png" / "tile.png"
    mask_path = loveda_dir / "extracted" / "Train" / "Urban" / "masks_png" / "tile.png"
    _write_rgb_image(image_path, size=(512, 512))
    _write_loveda_mask(mask_path, [2, 3, 4, 7], size=(512, 512))

    openearthmap_result = download_and_prepare_openearthmap(
        openearthmap_dir,
        download=False,
        extract=False,
    )
    loveda_result = download_and_prepare_loveda(
        loveda_dir,
        download=False,
        extract=False,
    )

    openearthmap_sample = read_manifest(openearthmap_result.manifest_path)[0]
    loveda_sample = read_manifest(loveda_result.manifest_path)[0]
    openearthmap_prepared = load_prepared_sample(openearthmap_sample.prepared_path, torch=torch)
    loveda_prepared = load_prepared_sample(loveda_sample.prepared_path, torch=torch)

    assert openearthmap_prepared.image.shape == loveda_prepared.image.shape
    assert openearthmap_prepared.mask.shape == loveda_prepared.mask.shape
    assert openearthmap_prepared.image.dtype == loveda_prepared.image.dtype == torch.uint8
    assert openearthmap_prepared.mask.dtype == loveda_prepared.mask.dtype == torch.uint8
    assert torch.equal(openearthmap_prepared.image, loveda_prepared.image)
    assert torch.equal(openearthmap_prepared.mask, loveda_prepared.mask)

    config = ModelConfig(image_size=256, mask_channels=9)
    samples = (openearthmap_sample, loveda_sample)
    preloaded = preload_samples(samples)
    crop_boxes = [
        center_square_crop_box(sample, crop_size=config.image_size, preloaded_samples=preloaded)
        for sample in samples
    ]
    images, masks = load_batch(
        samples,
        crop_boxes,
        torch=torch,
        model_config=config,
        device=torch.device("cpu"),
        preloaded_samples=preloaded,
    )

    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 9, 256, 256)
    assert images.dtype == masks.dtype == torch.float32
    assert torch.equal(images[0], images[1])
    assert torch.equal(masks[0], masks[1])
    assert torch.allclose(masks.sum(dim=1), torch.ones((2, 256, 256)))


def test_prepare_loveda_tracks_validation_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    image_path = data_dir / "extracted" / "Val" / "Rural" / "images_png" / "tile.png"
    mask_path = data_dir / "extracted" / "Val" / "Rural" / "masks_png" / "tile.png"
    _write_rgb_image(image_path)
    _write_loveda_mask(mask_path, [4, 5, 0, 1])

    result = download_and_prepare_loveda(data_dir, download=False, extract=False)
    samples = read_manifest(result.manifest_path)

    assert result.split_counts == {"val": 1}
    assert samples[0].split == "val"
    with Image.open(samples[0].mask_path) as remapped:
        assert mask_to_class_ids(remapped, ModelConfig(mask_channels=9)) == [6, 1, 0, 0]


def test_prepare_loveda_reports_missing_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    image_dir = data_dir / "extracted" / "Train" / "Urban" / "images_png"
    mask_dir = data_dir / "extracted" / "Train" / "Urban" / "masks_png"
    _write_rgb_image(image_dir / "paired.png")
    _write_loveda_mask(mask_dir / "paired.png", [2, 2, 2, 2])
    _write_rgb_image(image_dir / "missing.png")

    result = download_and_prepare_loveda(data_dir, download=False, extract=False)

    assert result.scanned_images == 2
    assert result.missing_masks == 1
    assert result.samples == 1


def test_prepare_loveda_maps_unknown_mask_values_to_unlabeled(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    image_path = data_dir / "extracted" / "Train" / "Urban" / "images_png" / "tile.png"
    mask_path = data_dir / "extracted" / "Train" / "Urban" / "masks_png" / "tile.png"
    _write_rgb_image(image_path)
    _write_loveda_mask(mask_path, [2, 3, 7, 8])

    result = download_and_prepare_loveda(data_dir, download=False, extract=False)
    samples = read_manifest(result.manifest_path)

    assert result.unknown_mask_pixels == 1
    assert result.unknown_mask_values == {8: 1}
    with Image.open(samples[0].mask_path) as remapped:
        assert mask_to_class_ids(remapped, ModelConfig(mask_channels=9)) == [8, 4, 7, 0]


def test_prepare_loveda_rejects_image_dirs_without_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    _write_rgb_image(data_dir / "extracted" / "Train" / "Urban" / "images_png" / "tile.png")

    with pytest.raises(RuntimeError, match="image directories without matching masks"):
        download_and_prepare_loveda(data_dir, download=False, extract=False)


def test_prepare_loveda_can_require_official_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    image_path = data_dir / "extracted" / "Train" / "Urban" / "images_png" / "tile.png"
    mask_path = data_dir / "extracted" / "Train" / "Urban" / "masks_png" / "tile.png"
    _write_rgb_image(image_path)
    _write_loveda_mask(mask_path, [2, 3, 6, 7])

    with pytest.raises(RuntimeError, match="official train\\+val layout"):
        download_and_prepare_loveda(
            data_dir,
            download=False,
            extract=False,
            validate_official_counts=True,
        )


def test_prepare_loveda_rejects_when_no_pairs_are_found(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    (data_dir / "extracted" / "Train" / "Urban" / "images_png").mkdir(parents=True)
    (data_dir / "extracted" / "Train" / "Urban" / "masks_png").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="No LoveDA image/mask pairs"):
        download_and_prepare_loveda(data_dir, download=False, extract=False)


def test_prepare_loveda_rejects_unsafe_zip_members(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    archive_path = data_dir / "raw" / "Train.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(data_dir / "raw" / "Val.zip", "w") as archive:
        archive.writestr("Val/Rural/images_png/tile.png", _png_bytes("RGB", (1, 2, 3)))

    with pytest.raises(RuntimeError, match="unsafe zip member"):
        download_and_prepare_loveda(data_dir, download=False, extract=True)

    assert not (data_dir.parent / "escape.txt").exists()


def test_prepare_loveda_replaces_partial_extraction_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "loveda"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    with zipfile.ZipFile(raw_dir / "Train.zip", "w") as archive:
        archive.writestr("Train/Urban/images_png/tile.png", _png_bytes("RGB", (1, 2, 3)))
        archive.writestr("Train/Urban/masks_png/tile.png", _png_bytes("L", 2))
    with zipfile.ZipFile(raw_dir / "Val.zip", "w") as archive:
        archive.writestr("Val/Rural/images_png/tile.png", _png_bytes("RGB", (4, 5, 6)))
        archive.writestr("Val/Rural/masks_png/tile.png", _png_bytes("L", 6))

    stale_path = data_dir / "extracted" / "stale.txt"
    _touch(stale_path)

    result = download_and_prepare_loveda(data_dir, download=False, extract=True)
    samples = read_manifest(result.manifest_path)

    assert not stale_path.exists()
    assert result.samples == 2
    assert {sample.split for sample in samples} == {"train", "val"}
    assert (data_dir / "extracted" / ".loveda-extract-complete.json").exists()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _write_rgb_image(path: Path, *, size: tuple[int, int] = (2, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (1, 2, 3)).save(path)


def _write_loveda_mask(
    path: Path,
    values: list[int],
    *,
    size: tuple[int, int] = (2, 2),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", size, values[0])
    pixels = mask.load()
    quadrant_values = {
        (0, 0): values[0],
        (1, 0): values[1],
        (0, 1): values[2],
        (1, 1): values[3],
    }
    half_width = size[0] // 2
    half_height = size[1] // 2
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = quadrant_values[(x // half_width, y // half_height)]
    mask.save(path)


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


def _png_bytes(mode: str, color: float | tuple[float, ...] | str | None) -> bytes:
    buffer = BytesIO()
    Image.new(mode, (1, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()
