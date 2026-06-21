from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.data.sam_training import build_sam_training_data
from sim_satellite.model import ModelConfig


def test_build_sam_training_data_writes_binary_class_records(tmp_path: Path) -> None:
    sample = _write_sample(tmp_path / "OpenEarthMap" / "train" / "abancay", split="train")
    manifest_path = tmp_path / "source" / "manifest.jsonl"
    output_dir = tmp_path / "sam"
    write_manifest((sample,), manifest_path)

    result = build_sam_training_data(
        manifest_path,
        output_dir,
        min_mask_pixels=1,
        positive_points=2,
        negative_points=1,
    )

    assert result.records == 2
    assert result.mask_files == 2
    assert result.skipped_empty_or_small_masks == 6

    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["class_name"] for record in records] == ["road", "building"]
    assert records[0]["bbox_xyxy"] == [0, 0, 1, 1]
    assert records[0]["positive_points_xy"] == [[0, 0], [1, 1]]
    assert records[0]["negative_points_xy"] == [[2, 4]]
    assert records[0]["concept_prompt"] == "road"
    assert records[0]["region_key"] == "abancay"
    assert records[0]["label_source"] == "openearthmap-ground-truth"
    assert Path(records[0]["mask"]).suffix == ".png"

    road_mask = Image.open(output_dir / records[0]["mask"])
    assert road_mask.mode == "L"
    assert road_mask.size == (8, 8)
    assert road_mask.getpixel((0, 0)) == 255
    assert road_mask.getpixel((2, 0)) == 0


def test_build_sam_training_data_balances_record_limit(tmp_path: Path) -> None:
    sample_a = _write_sample(tmp_path / "OpenEarthMap" / "train" / "region-a", split="train")
    sample_b = _write_sample(tmp_path / "OpenEarthMap" / "train" / "region-b", split="train")
    manifest_path = tmp_path / "source" / "manifest.jsonl"
    write_manifest((sample_a, sample_b), manifest_path)

    result = build_sam_training_data(
        manifest_path,
        tmp_path / "sam",
        min_mask_pixels=1,
        max_records=2,
        sample_seed=7,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "sam" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result.records == 2
    assert len({record["region_key"] for record in records}) == 2


def _write_sample(root: Path, *, split: str) -> SatelliteSample:
    image_path = root / "images" / "tile.png"
    mask_path = root / "labels" / "tile.png"
    prepared_path = root / "prepared" / "tile.pt"
    _write_image(image_path)
    _write_mask(mask_path)
    sample = SatelliteSample(
        image_path=image_path,
        mask_path=mask_path,
        prepared_path=prepared_path,
        split=split,
        source="openearthmap",
    )
    write_prepared_sample(
        sample,
        torch=torch,
        config=ModelConfig(image_size=8, mask_channels=9),
    )
    return sample


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (8, 8), 0)
    for point in ((0, 0), (1, 0), (0, 1), (1, 1)):
        mask.putpixel(point, 4)
    for point in ((2, 2), (3, 2), (2, 3), (3, 3)):
        mask.putpixel(point, 8)
    mask.save(path)
