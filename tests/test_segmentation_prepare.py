from __future__ import annotations

import json
from pathlib import Path

import torch
from click.testing import CliRunner
from PIL import Image

from sim_satellite.cli import cli
from sim_satellite.data.loader import read_manifest
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.segmentation import (
    SegmentationPreparationConfig,
    prepare_segmentation_manifest,
)


def test_segmentation_prepare_raw_manifest_writes_prepared_training_manifest(
    tmp_path: Path,
) -> None:
    for index, _split in enumerate(("custom-train", "custom-val")):
        _write_rgb_fixture(tmp_path / "raw" / f"tile-{index}.png", index=index)
        _write_label_fixture(tmp_path / "raw" / f"tile-{index}-mask.png", index=index)
    raw_manifest_path = tmp_path / "raw.jsonl"
    raw_manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": "raw/tile-0.png",
                        "mask": "raw/tile-0-mask.png",
                        "split": "custom-train",
                        "source": "fixture",
                    }
                ),
                json.dumps(
                    {
                        "image": "raw/tile-1.png",
                        "mask": "raw/tile-1-mask.png",
                        "split": "custom-val",
                        "source": "fixture",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_manifest_path = tmp_path / "prepared-manifest.jsonl"

    result = prepare_segmentation_manifest(
        SegmentationPreparationConfig(
            raw_manifest_path=str(raw_manifest_path),
            output_manifest_path=str(output_manifest_path),
            prepared_dir=str(tmp_path / "prepared"),
        )
    )

    prepared_samples = read_manifest(output_manifest_path)
    assert result.samples == 2
    assert result.splits == {"custom-train": 1, "custom-val": 1}
    assert [sample.source for sample in prepared_samples] == ["fixture", "fixture"]
    assert all(sample.prepared_path.exists() for sample in prepared_samples)
    prepared = load_prepared_sample(prepared_samples[0].prepared_path, torch=torch)
    assert tuple(prepared.image.shape) == (16, 16, 3)
    assert tuple(prepared.mask.shape) == (16, 16)


def test_cli_exposes_segmentation_prepare_command() -> None:
    result = CliRunner().invoke(cli, ["segmentation-prepare", "--help"])

    assert result.exit_code == 0


def _write_rgb_fixture(path: Path, *, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), (index * 20, 32, 64))
    for x in range(8, 16):
        for y in range(16):
            image.putpixel((x, y), (222, 31, 7))
    image.save(path)


def _write_label_fixture(path: Path, *, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (16, 16), 5 if index % 2 else 4)
    for x in range(8, 16):
        for y in range(16):
            mask.putpixel((x, y), 8)
    mask.save(path)
