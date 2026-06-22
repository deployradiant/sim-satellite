from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.data.sam_records import SamImageRecord, write_sam_image_records
from sim_satellite.model import ModelConfig
from sim_satellite.sam_compare import (
    SamProposalConfig,
    compare_sam_models,
    compare_sam_proposals,
    evaluate_sam_proposals,
    generate_sam_proposals,
)


def test_generate_sam_proposals_from_unlabeled_openearthmap_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "OpenEarthMap" / "train" / "abancay" / "images" / "tile.png"
    _write_image(image_path)
    images_manifest = tmp_path / "unlabeled-images.jsonl"
    write_sam_image_records(
        (
            SamImageRecord(
                image_path=image_path,
                split="train",
                source="openearthmap-unlabeled",
                region_key="abancay",
                image_key=str(image_path.with_suffix("")),
            ),
        ),
        images_manifest,
    )

    result = generate_sam_proposals(
        SamProposalConfig(
            output_dir=str(tmp_path / "proposals"),
            model_id="baseline",
            model_label="baseline",
            manifest_path=None,
            images_manifest_path=str(images_manifest),
            min_mask_region_area=1,
        ),
        generator_factory=lambda config: _FakeSamGenerator(config.model_id),
    )

    records = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    assert result.images == 1
    assert result.proposals == 1
    assert records[0]["source"] == "openearthmap-unlabeled"
    assert records[0]["region_key"] == "abancay"
    assert records[0]["model_id"] == "baseline"


def test_compare_sam_models_scores_baseline_against_trained(tmp_path: Path) -> None:
    sample = _write_sample(tmp_path / "OpenEarthMap" / "train" / "abancay")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest((sample,), manifest_path)

    result = compare_sam_models(
        manifest_path=manifest_path,
        baseline_model_id="baseline",
        trained_model_id="trained",
        output_dir=tmp_path / "comparison",
        min_mask_pixels=1,
        generator_factory=lambda config: _FakeSamGenerator(config.model_id),
    )

    assert result.baseline.ground_truth_masks == 2
    assert result.baseline.mean_best_iou == 0.5
    assert result.trained.mean_best_iou == 1.0
    assert result.trained.mean_best_iou - result.baseline.mean_best_iou == 0.5
    assert (tmp_path / "comparison" / "comparison.json").exists()


def test_compare_sam_proposal_files_reports_same_delta(tmp_path: Path) -> None:
    sample = _write_sample(tmp_path / "OpenEarthMap" / "train" / "abancay")
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest((sample,), manifest_path)
    baseline = generate_sam_proposals(
        SamProposalConfig(
            output_dir=str(tmp_path / "baseline"),
            model_id="baseline",
            manifest_path=str(manifest_path),
            min_mask_region_area=1,
        ),
        generator_factory=lambda config: _FakeSamGenerator(config.model_id),
    )
    trained = generate_sam_proposals(
        SamProposalConfig(
            output_dir=str(tmp_path / "trained"),
            model_id="trained",
            manifest_path=str(manifest_path),
            min_mask_region_area=1,
        ),
        generator_factory=lambda config: _FakeSamGenerator(config.model_id),
    )

    baseline_eval = evaluate_sam_proposals(
        manifest_path=manifest_path,
        proposals_path=baseline.records_path,
        min_mask_pixels=1,
    )
    comparison = compare_sam_proposals(
        manifest_path=manifest_path,
        baseline_proposals_path=baseline.records_path,
        trained_proposals_path=trained.records_path,
        min_mask_pixels=1,
    )

    assert baseline_eval.recall_by_threshold == {"0.50": 0.5, "0.75": 0.5}
    assert comparison.trained.recall_by_threshold == {"0.50": 1.0, "0.75": 1.0}


class _FakeSamGenerator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, image: Any) -> list[dict[str, object]]:
        assert image.shape == (8, 8, 3)
        road = np.zeros((8, 8), dtype=bool)
        road[:2, :2] = True
        proposals = [_proposal(road, [0, 0, 2, 2])]
        if self.model_id == "trained":
            building = np.zeros((8, 8), dtype=bool)
            building[2:4, 2:4] = True
            proposals.append(_proposal(building, [2, 2, 2, 2]))
        return proposals


def _proposal(mask: Any, bbox: list[int]) -> dict[str, object]:
    return {
        "segmentation": mask,
        "bbox": bbox,
        "predicted_iou": 0.9,
        "stability_score": 0.95,
        "crop_box": [0, 0, 8, 8],
        "point_coords": [[1, 1]],
    }


def _write_sample(root: Path) -> SatelliteSample:
    image_path = root / "images" / "tile.png"
    mask_path = root / "labels" / "tile.png"
    prepared_path = root / "prepared" / "tile.pt"
    _write_image(image_path)
    _write_mask(mask_path)
    sample = SatelliteSample(
        image_path=image_path,
        mask_path=mask_path,
        prepared_path=prepared_path,
        split="train",
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
