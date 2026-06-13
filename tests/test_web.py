from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite.data.loader import SatelliteSample  # noqa: E402
from sim_satellite.web import WebConfig, _mask_preview_png, _sample_at  # noqa: E402


def test_mask_preview_colorizes_grayscale_class_ids(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (2, 1))
    mask.putdata([0, 8])
    mask.save(mask_path)

    png = _mask_preview_png(mask_path, WebConfig(display_size=2))

    with Image.open(BytesIO(png)) as preview:
        assert preview.mode == "RGB"
        assert preview.size == (2, 2)
        assert preview.getpixel((0, 0)) == (0, 0, 0)
        assert preview.getpixel((1, 0)) == (222, 31, 7)


def test_sample_at_rejects_negative_indexes(tmp_path: Path) -> None:
    sample = SatelliteSample(
        image_path=tmp_path / "image.tif",
        mask_path=tmp_path / "mask.tif",
        split="train",
    )

    with pytest.raises(ValueError, match="out of range"):
        _sample_at((sample,), -1)
