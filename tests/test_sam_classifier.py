from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.sam_records import image_key
from sim_satellite.sam_classifier import (
    SamClassifierLabelConfig,
    SamClassifierPredictConfig,
    SamClassifierTrainConfig,
    classify_sam_proposals,
    label_sam_proposals_for_classifier,
    train_sam_mask_classifier,
)


def test_sam_classifier_labels_proposals_from_ground_truth_overlap(tmp_path: Path) -> None:
    manifest_path, proposals_path = _write_classifier_fixture(tmp_path)

    result = label_sam_proposals_for_classifier(
        SamClassifierLabelConfig(
            manifest_path=str(manifest_path),
            proposals_path=str(proposals_path),
            output_path=str(tmp_path / "classifier-records.jsonl"),
            min_mask_pixels=1,
            min_class_fraction=0.8,
        )
    )

    records = _jsonl(result.output_path)
    assert result.proposals == 3
    assert result.labeled_records == 2
    assert result.skipped_low_purity == 1
    assert result.class_counts == {"building": 1, "road": 1}
    assert [record["class_name"] for record in records] == ["road", "building"]
    assert records[0]["label_source"] == "openearthmap-majority-overlap"


def test_sam_classifier_trains_and_predicts(tmp_path: Path) -> None:
    manifest_path, proposals_path = _write_classifier_fixture(tmp_path)
    records_path = tmp_path / "classifier-records.jsonl"
    label_sam_proposals_for_classifier(
        SamClassifierLabelConfig(
            manifest_path=str(manifest_path),
            proposals_path=str(proposals_path),
            output_path=str(records_path),
            min_mask_pixels=1,
            min_class_fraction=0.8,
        )
    )

    checkpoint_path = tmp_path / "classifier.pt"
    train_result = train_sam_mask_classifier(
        SamClassifierTrainConfig(
            records_path=str(records_path),
            checkpoint_path=str(checkpoint_path),
            epochs=2,
            batch_size=2,
            input_size=8,
            validation_fraction=0.0,
        )
    )
    prediction_path = tmp_path / "predictions.jsonl"
    predict_result = classify_sam_proposals(
        SamClassifierPredictConfig(
            proposals_path=str(proposals_path),
            checkpoint_path=str(checkpoint_path),
            output_path=str(prediction_path),
            input_size=8,
        )
    )

    predictions = _jsonl(prediction_path)
    assert train_result.records == 2
    assert train_result.train_records == 2
    assert train_result.validation_records == 0
    assert checkpoint_path.exists()
    assert predict_result.proposals == 3
    assert predict_result.predictions == 3
    assert Path(str(predictions[0]["mask"])).is_absolute()
    assert Path(str(predictions[0]["mask"])).exists()
    assert {record["predicted_class_name"] for record in predictions} <= {
        "unlabeled",
        "bareland",
        "rangeland",
        "developed space",
        "road",
        "tree",
        "water",
        "agriculture land",
        "building",
    }


def _write_classifier_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "OpenEarthMap" / "train" / "region" / "images" / "tile.png"
    mask_path = tmp_path / "OpenEarthMap" / "train" / "region" / "labels" / "tile.png"
    prepared_path = tmp_path / "prepared" / "tile.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    proposals_path = tmp_path / "sam" / "proposals.jsonl"

    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    for x in range(0, 4):
        for y in range(0, 4):
            image.putpixel((x, y), (200, 200, 200))
    for x in range(4, 8):
        for y in range(4, 8):
            image.putpixel((x, y), (170, 70, 50))
    image.save(image_path)

    mask = Image.new("L", (8, 8), 0)
    for x in range(0, 4):
        for y in range(0, 4):
            mask.putpixel((x, y), 4)
    for x in range(4, 8):
        for y in range(4, 8):
            mask.putpixel((x, y), 8)
    mask.save(mask_path)

    write_manifest(
        (
            SatelliteSample(
                image_path=image_path,
                mask_path=mask_path,
                prepared_path=prepared_path,
                split="train",
                source="openearthmap",
            ),
        ),
        manifest_path,
    )
    _write_proposal(
        proposals_path,
        image_path,
        "road.png",
        (0, 0, 3, 3),
        [(x, y) for x in range(4) for y in range(4)],
    )
    _write_proposal(
        proposals_path,
        image_path,
        "building.png",
        (4, 4, 7, 7),
        [(x, y) for x in range(4, 8) for y in range(4, 8)],
        append=True,
    )
    _write_proposal(
        proposals_path,
        image_path,
        "mixed.png",
        (0, 0, 7, 7),
        [(x, y) for x in range(8) for y in range(8)],
        append=True,
    )
    return manifest_path, proposals_path


def _write_proposal(
    proposals_path: Path,
    image_path: Path,
    name: str,
    bbox_xyxy: tuple[int, int, int, int],
    pixels: list[tuple[int, int]],
    *,
    append: bool = False,
) -> None:
    mask_path = proposals_path.parent / "masks" / name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (8, 8), 0)
    for point in pixels:
        mask.putpixel(point, 255)
    mask.save(mask_path)
    record = {
        "image": str(image_path),
        "mask": str(mask_path.relative_to(proposals_path.parent)),
        "image_key": image_key(image_path),
        "bbox_xyxy": list(bbox_xyxy),
        "area": len(pixels),
    }
    mode = "a" if append else "w"
    with proposals_path.open(mode, encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
