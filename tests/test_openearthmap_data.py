from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import download_and_prepare_openearthmap


def test_manifest_requires_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"image": "image.png", "mask": "mask.png", '
        '"prepared": "prepared.pt", "split": "train"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required source"):
        read_manifest(manifest_path)


def test_prepare_openearthmap_writes_manifest_without_classes_sidecar(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "openearthmap"
    image_path = data_dir / "extracted" / "train" / "images" / "tile.png"
    mask_path = data_dir / "extracted" / "train" / "labels" / "tile.png"
    _write_rgb_image(image_path)
    _write_label_mask(mask_path)

    result = download_and_prepare_openearthmap(data_dir, download=False, extract=False)

    assert result.samples == 1
    assert result.scanned_images == 1
    assert result.missing_masks == 0
    assert result.image_dirs_without_labels == 0

    samples = read_manifest(result.manifest_path)
    assert len(samples) == 1
    assert samples[0].image_path == image_path
    assert samples[0].mask_path == mask_path
    assert samples[0].prepared_path == data_dir / "prepared" / "train" / "images" / "tile.pt"
    assert samples[0].split == "train"
    assert samples[0].prepared_path.exists()
    assert result.prepared_dir == data_dir / "prepared"
    assert result.prepared_samples == 1
    assert not (data_dir / "classes.json").exists()


def test_prepare_openearthmap_reports_and_skips_missing_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    image_dir = data_dir / "extracted" / "train" / "images"
    label_dir = data_dir / "extracted" / "train" / "labels"
    _write_rgb_image(image_dir / "paired.png")
    _write_label_mask(label_dir / "paired.png")
    _write_rgb_image(image_dir / "unlabeled.png")

    result = download_and_prepare_openearthmap(data_dir, download=False, extract=False)
    samples = read_manifest(result.manifest_path)

    assert result.scanned_images == 2
    assert result.missing_masks == 1
    assert result.samples == 1
    assert [sample.image_path.name for sample in samples] == ["paired.png"]


def test_prepare_openearthmap_rejects_when_no_pairs_are_found(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    (data_dir / "extracted" / "train" / "images").mkdir(parents=True)
    (data_dir / "extracted" / "train" / "labels").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="No OpenEarthMap image/mask pairs"):
        download_and_prepare_openearthmap(data_dir, download=False, extract=False)


def test_prepare_openearthmap_rejects_image_dirs_without_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    _write_rgb_image(data_dir / "extracted" / "train" / "images" / "tile.png")

    with pytest.raises(RuntimeError, match="image directories without matching labels"):
        download_and_prepare_openearthmap(data_dir, download=False, extract=False)


def test_prepare_openearthmap_rejects_unsafe_zip_members(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    archive_path = data_dir / "raw" / "OpenEarthMap.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe zip member"):
        download_and_prepare_openearthmap(data_dir, download=False, extract=True)

    assert not (data_dir.parent / "escape.txt").exists()


def test_prepare_openearthmap_replaces_partial_extraction_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    archive_path = data_dir / "raw" / "OpenEarthMap.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("OpenEarthMap/train/images/tile.png", _png_bytes("RGB", (1, 2, 3)))
        archive.writestr("OpenEarthMap/train/labels/tile.png", _png_bytes("L", 0))

    stale_path = data_dir / "extracted" / "stale.txt"
    _touch(stale_path)

    result = download_and_prepare_openearthmap(data_dir, download=False, extract=True)
    samples = read_manifest(result.manifest_path)

    assert not stale_path.exists()
    assert result.samples == 1
    assert result.scanned_images == 1
    assert [sample.image_path.name for sample in samples] == ["tile.png"]
    assert (data_dir / "extracted" / ".openearthmap-extract-complete.json").exists()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), (1, 2, 3)).save(path)


def _write_label_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (1, 1), 0).save(path)


def _png_bytes(mode: str, color: float | tuple[float, ...] | str | None) -> bytes:
    buffer = BytesIO()
    Image.new(mode, (1, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()
