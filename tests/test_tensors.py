from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite.data.tensors import _mask_to_class_ids  # noqa: E402
from sim_satellite.model import ModelConfig  # noqa: E402

OPENEARTHMAP_CLASSES = (
    "unlabeled",
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "building",
)
OPENEARTHMAP_RGB = {
    "unlabeled": (0, 0, 0),
    "bareland": (128, 0, 0),
    "rangeland": (0, 255, 36),
    "developed space": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "agriculture land": (75, 181, 73),
    "building": (222, 31, 7),
}


def test_rgb_mask_decodes_openearthmap_palette() -> None:
    mask = Image.new("RGB", (len(OPENEARTHMAP_CLASSES), 1))
    mask.putdata([OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES])

    class_ids = _mask_to_class_ids(mask, ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)))

    assert class_ids == list(range(len(OPENEARTHMAP_CLASSES)))


def test_palette_mask_decodes_openearthmap_palette() -> None:
    mask = Image.new("P", (len(OPENEARTHMAP_CLASSES), 1))
    palette = [0] * 768
    for class_id, name in enumerate(OPENEARTHMAP_CLASSES):
        rgb = OPENEARTHMAP_RGB[name]
        palette[class_id * 3 : class_id * 3 + 3] = rgb
    mask.putpalette(palette)
    mask.putdata(range(len(OPENEARTHMAP_CLASSES)))

    class_ids = _mask_to_class_ids(mask, ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)))

    assert class_ids == list(range(len(OPENEARTHMAP_CLASSES)))


def test_rgb_mask_rejects_unmapped_color() -> None:
    mask = Image.new("RGB", (1, 1), (1, 2, 3))

    with pytest.raises(ValueError, match="unmapped RGB color"):
        _mask_to_class_ids(mask, ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)))


def test_rgb_mask_rejects_class_outside_mask_channels() -> None:
    mask = Image.new("RGB", (1, 1), OPENEARTHMAP_RGB["building"])

    with pytest.raises(ValueError, match="maps to class 8"):
        _mask_to_class_ids(mask, ModelConfig(mask_channels=8))


def test_grayscale_mask_decodes_class_ids_without_clamping() -> None:
    mask = Image.new("L", (2, 1))
    mask.putdata([0, 8])

    class_ids = _mask_to_class_ids(mask, ModelConfig(mask_channels=9))

    assert class_ids == [0, 8]


def test_grayscale_mask_rejects_class_id_outside_mask_channels() -> None:
    mask = Image.new("L", (1, 1), 9)

    with pytest.raises(ValueError, match="class id 9"):
        _mask_to_class_ids(mask, ModelConfig(mask_channels=9))
