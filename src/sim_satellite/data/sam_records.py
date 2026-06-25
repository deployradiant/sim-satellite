"""Shared records for SAM proposal generation and comparison.

The SAM comparison path uses two OpenEarthMap subsets:

- labeled pairs from `data/openearthmap/manifest.jsonl` for training-data export
  and recall evaluation
- unlabeled images from `data/openearthmap/unlabeled-images.jsonl` for running
  the winning SAM model on images that do not have masks

Proposal records are class-agnostic on purpose. Baseline SAM and trained SAM
write the same shape so evaluation can compare only mask quality, not side
format differences.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAM_RECORD_VERSION = 1


@dataclass(frozen=True)
class SamImageRecord:
    image_path: Path
    split: str
    source: str
    region_key: str
    image_key: str

    @classmethod
    def from_json(cls, value: dict[str, str], root: Path) -> SamImageRecord:
        image_path = _resolve(root, value["image"])
        return cls(
            image_path=image_path,
            split=value["split"],
            source=value.get("source", "openearthmap-unlabeled"),
            region_key=value.get("region_key", infer_region_key(image_path)),
            image_key=value.get("image_key", image_key(image_path)),
        )

    def to_json(self, root: Path) -> dict[str, str]:
        return {
            "image": _relative_or_absolute(self.image_path, root),
            "split": self.split,
            "source": self.source,
            "region_key": self.region_key,
            "image_key": self.image_key,
        }


@dataclass(frozen=True)
class SamMaskRecord:
    image_path: Path
    mask_path: Path
    split: str
    source: str
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    image_size: tuple[int, int]
    region_key: str
    image_key: str
    label_source: str
    model_id: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    prepared_path: Path | None = None
    source_mask_path: Path | None = None
    parent_image_path: Path | None = None
    positive_points_xy: tuple[tuple[int, int], ...] = ()
    negative_points_xy: tuple[tuple[int, int], ...] = ()
    tile_row: int | None = None
    tile_col: int | None = None
    tile_grid: int | None = None
    tile_xyxy: tuple[int, int, int, int] | None = None
    predicted_iou: float | None = None
    stability_score: float | None = None
    crop_box_xywh: tuple[float, float, float, float] | None = None
    point_coords_xy: tuple[tuple[float, float], ...] = ()

    def to_json(self, root: Path) -> dict[str, object]:
        value: dict[str, object] = {
            "version": SAM_RECORD_VERSION,
            "image": _relative_or_absolute(self.image_path, root),
            "mask": _relative_or_absolute(self.mask_path, root),
            "split": self.split,
            "source": self.source,
            "label_source": self.label_source,
            "bbox_xyxy": list(self.bbox_xyxy),
            "area": self.area,
            "image_size": list(self.image_size),
            "region_key": self.region_key,
            "image_key": self.image_key,
        }
        if self.model_id is not None:
            value["model_id"] = self.model_id
        if self.class_id is not None:
            value["class_id"] = self.class_id
        if self.class_name is not None:
            value["class_name"] = self.class_name
            value["concept_prompt"] = self.class_name
        if self.prepared_path is not None:
            value["prepared"] = _relative_or_absolute(self.prepared_path, root)
        if self.source_mask_path is not None:
            value["source_mask"] = _relative_or_absolute(self.source_mask_path, root)
        if self.parent_image_path is not None:
            value["parent_image"] = _relative_or_absolute(self.parent_image_path, root)
        if self.positive_points_xy:
            value["positive_points_xy"] = [list(point) for point in self.positive_points_xy]
        if self.negative_points_xy:
            value["negative_points_xy"] = [list(point) for point in self.negative_points_xy]
        if self.tile_row is not None:
            value["tile_row"] = self.tile_row
        if self.tile_col is not None:
            value["tile_col"] = self.tile_col
        if self.tile_grid is not None:
            value["tile_grid"] = self.tile_grid
        if self.tile_xyxy is not None:
            value["tile_xyxy"] = list(self.tile_xyxy)
        if self.predicted_iou is not None:
            value["predicted_iou"] = self.predicted_iou
        if self.stability_score is not None:
            value["stability_score"] = self.stability_score
        if self.crop_box_xywh is not None:
            value["crop_box_xywh"] = list(self.crop_box_xywh)
        if self.point_coords_xy:
            value["point_coords_xy"] = [list(point) for point in self.point_coords_xy]
        return value


def write_sam_image_records(records: Sequence[SamImageRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record.to_json(root), handle, sort_keys=True)
            handle.write("\n")
    return output


def read_sam_image_records(path: str | Path) -> tuple[SamImageRecord, ...]:
    input_path = Path(path)
    root = input_path.parent
    records: list[SamImageRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = _json_line(input_path, line_number, line)
            if value is None:
                continue
            records.append(SamImageRecord.from_json(_string_mapping(value), root))
    return tuple(records)


def write_sam_mask_records(records: Sequence[SamMaskRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record.to_json(root), handle, sort_keys=True)
            handle.write("\n")
    return output


def records_from_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = _json_line(input_path, line_number, line)
            if value is not None:
                records.append(value)
    return tuple(records)


def balanced_sam_records(
    records: Sequence[SamMaskRecord],
    *,
    max_records: int | None,
    seed: int,
    keys: tuple[str, ...] = ("class", "split", "region", "image"),
) -> tuple[SamMaskRecord, ...]:
    """Return deterministic round-robin samples across diversity buckets."""

    if max_records is None or len(records) <= max_records:
        return tuple(records)
    if max_records < 1:
        raise ValueError("max_records must be at least 1 when provided")

    rng = random.Random(seed)
    buckets: dict[tuple[object, ...], list[SamMaskRecord]] = defaultdict(list)
    for record in records:
        buckets[_bucket_key(record, keys)].append(record)

    ordered_buckets = list(buckets.values())
    for bucket in ordered_buckets:
        rng.shuffle(bucket)
    rng.shuffle(ordered_buckets)

    selected: list[SamMaskRecord] = []
    while len(selected) < max_records and ordered_buckets:
        next_buckets: list[list[SamMaskRecord]] = []
        for bucket in ordered_buckets:
            selected.append(bucket.pop())
            if len(selected) >= max_records:
                break
            if bucket:
                next_buckets.append(bucket)
        ordered_buckets = next_buckets
    return tuple(selected)


def infer_region_key(path: Path) -> str:
    if path.parent.name.lower() in {"image", "images", "img", "imgs", "label", "labels"}:
        return path.parent.parent.name
    parts = path.parts
    for index, part in enumerate(parts):
        if part.lower() in {"train", "val", "valid", "validation", "test"} and index + 1 < len(
            parts
        ):
            return parts[index + 1]
    if len(parts) >= 2:
        return parts[-2]
    return path.stem


def image_key(path: Path) -> str:
    return str(path.with_suffix(""))


def _bucket_key(record: SamMaskRecord, keys: Iterable[str]) -> tuple[object, ...]:
    values: list[object] = []
    for key in keys:
        if key == "class":
            values.append(record.class_id)
        elif key == "split":
            values.append(record.split)
        elif key == "region":
            values.append(record.region_key)
        elif key == "image":
            values.append(record.image_key)
        else:
            raise ValueError(f"Unsupported SAM diversity key: {key}")
    return tuple(values)


def _json_line(path: Path, line_number: int, line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: record must be a JSON object")
    return value


def _string_mapping(value: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise ValueError(f"SAM image record field {key!r} must be a string")
        output[key] = item
    return output


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
