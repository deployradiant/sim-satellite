from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite.data.loader import read_manifest  # noqa: E402
from sim_satellite.data.openearthmap import (  # noqa: E402
    OPENEARTHMAP_CLASSES,
    OPENEARTHMAP_RGB,
    download_and_prepare_openearthmap,
    scan_openearthmap,
)

EXPECTED_OPENEARTHMAP_CLASSES = (
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "building",
)
EXPECTED_OPENEARTHMAP_RGB = {
    "bareland": (128, 0, 0),
    "rangeland": (0, 255, 36),
    "developed space": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "agriculture land": (75, 181, 73),
    "building": (222, 31, 7),
}


def test_openearthmap_taxonomy_matches_source_metadata() -> None:
    assert OPENEARTHMAP_CLASSES == EXPECTED_OPENEARTHMAP_CLASSES
    assert OPENEARTHMAP_RGB == EXPECTED_OPENEARTHMAP_RGB


def test_prepare_openearthmap_writes_manifest_and_classes(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    image_path = data_dir / "extracted" / "train" / "images" / "tile.png"
    mask_path = data_dir / "extracted" / "train" / "labels" / "tile.png"
    _touch(image_path)
    _touch(mask_path)

    result = download_and_prepare_openearthmap(data_dir, download=False, extract=False)

    assert result.samples == 1
    assert result.scanned_images == 1
    assert result.missing_masks == 0
    assert result.image_dirs_without_labels == 0

    samples = read_manifest(result.manifest_path)
    assert len(samples) == 1
    assert samples[0].image_path == image_path
    assert samples[0].mask_path == mask_path
    assert samples[0].split == "train"

    class_payload = json.loads(result.classes_path.read_text(encoding="utf-8"))
    expected_classes = [
        {
            "id": class_id,
            "name": name,
            "rgb": list(EXPECTED_OPENEARTHMAP_RGB[name]),
        }
        for class_id, name in enumerate(EXPECTED_OPENEARTHMAP_CLASSES)
    ]
    assert class_payload == {"source": "openearthmap", "classes": expected_classes}


def test_prepare_openearthmap_reports_progress(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    _touch(data_dir / "extracted" / "train" / "images" / "tile.png")
    _touch(data_dir / "extracted" / "train" / "labels" / "tile.png")
    messages: list[str] = []

    download_and_prepare_openearthmap(
        data_dir,
        download=False,
        extract=False,
        progress=messages.append,
    )

    assert messages[0].startswith("Preparing OpenEarthMap data")
    assert any(message.startswith("Finding image directories") for message in messages)
    assert any(message.startswith("Scanned 1/1 image directories") for message in messages)
    assert any(message.startswith("Writing manifest") for message in messages)
    assert messages[-1].startswith("Done: 1 samples")


def test_scan_openearthmap_reports_missing_masks(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    _touch(extracted_dir / "train" / "images" / "tile.png")
    (extracted_dir / "train" / "labels").mkdir(parents=True)

    scan = scan_openearthmap(extracted_dir)

    assert scan.scanned_images == 1
    assert scan.missing_masks == 1
    assert scan.image_dirs_without_labels == 0
    assert scan.samples == ()


def test_prepare_openearthmap_skips_missing_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    image_dir = data_dir / "extracted" / "train" / "images"
    label_dir = data_dir / "extracted" / "train" / "labels"
    _touch(image_dir / "paired.png")
    _touch(label_dir / "paired.png")
    _touch(image_dir / "unlabeled.png")

    result = download_and_prepare_openearthmap(data_dir, download=False, extract=False)
    samples = read_manifest(result.manifest_path)

    assert result.scanned_images == 2
    assert result.missing_masks == 1
    assert result.samples == 1
    assert [sample.image_path.name for sample in samples] == ["paired.png"]


def test_prepare_openearthmap_rejects_when_no_pairs_are_found(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    _touch(data_dir / "extracted" / "train" / "images" / "tile.png")
    (data_dir / "extracted" / "train" / "labels").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="No OpenEarthMap image/mask pairs"):
        download_and_prepare_openearthmap(data_dir, download=False, extract=False)


def test_prepare_openearthmap_rejects_image_dirs_without_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "openearthmap"
    _touch(data_dir / "extracted" / "train" / "images" / "tile.png")

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


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
