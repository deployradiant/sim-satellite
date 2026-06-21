"""Shared SAM-style mask record helpers.

The SAM-assisted labeling path uses a single JSONL shape for both ground-truth
OpenEarthMap binary masks and later SAM 2 automatic-mask proposals. Keeping the
record shape here makes the pipeline boring on purpose: proposal generation,
mask classification, evaluation, and pseudo-label merging can share provenance,
geometry, confidence, region, and image keys instead of each step inventing its
own side format.

`region_key` and `image_key` are first-class because OpenEarthMap has many
locations and strong class imbalance. Any record limit used for training should
sample across class, split, region, and image so a small run is still diverse.
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
class SamMaskRecord:
    image_path: Path
    mask_path: Path
    split: str
    source: str
    class_id: int | None
    class_name: str | None
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    image_size: tuple[int, int]
    region_key: str
    image_key: str
    label_source: str
    prepared_path: Path | None = None
    source_mask_path: Path | None = None
    positive_points_xy: tuple[tuple[int, int], ...] = ()
    negative_points_xy: tuple[tuple[int, int], ...] = ()
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
        if self.class_id is not None:
            value["class_id"] = self.class_id
        if self.class_name is not None:
            value["class_name"] = self.class_name
            value["concept_prompt"] = self.class_name
        if self.prepared_path is not None:
            value["prepared"] = _relative_or_absolute(self.prepared_path, root)
        if self.source_mask_path is not None:
            value["source_mask"] = _relative_or_absolute(self.source_mask_path, root)
        if self.positive_points_xy:
            value["positive_points_xy"] = [list(point) for point in self.positive_points_xy]
        if self.negative_points_xy:
            value["negative_points_xy"] = [list(point) for point in self.negative_points_xy]
        if self.predicted_iou is not None:
            value["predicted_iou"] = self.predicted_iou
        if self.stability_score is not None:
            value["stability_score"] = self.stability_score
        if self.crop_box_xywh is not None:
            value["crop_box_xywh"] = list(self.crop_box_xywh)
        if self.point_coords_xy:
            value["point_coords_xy"] = [list(point) for point in self.point_coords_xy]
        return value


def write_sam_records(records: Sequence[SamMaskRecord], records_path: str | Path) -> Path:
    path = Path(records_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record.to_json(root), handle, sort_keys=True)
            handle.write("\n")
    return path


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


def records_from_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(value)
    return tuple(records)


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


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
