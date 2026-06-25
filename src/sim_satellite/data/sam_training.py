"""Build SAM-style training records from labeled OpenEarthMap masks.

OpenEarthMap's labeled subset is dense semantic segmentation. SAM comparison
work needs binary promptable masks. This module bridges those shapes by splitting
each prepared semantic mask into one binary PNG per non-empty class and writing a
SAM JSONL record with class prompt, bbox, points, source paths, region key, and
image key.

Use these records to adapt or fine-tune SAM. Evaluate the resulting trained SAM
against the baseline SAM on held-out labeled OpenEarthMap images with
`sam-compare`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.sam_records import (
    SamMaskRecord,
    balanced_sam_records,
    image_key,
    infer_region_key,
    write_sam_mask_records,
)

DEFAULT_POSITIVE_POINTS = 3
DEFAULT_NEGATIVE_POINTS = 3
DEFAULT_SAMPLE_SEED = 2026
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class SamTrainingDataResult:
    output_dir: Path
    records_path: Path
    records: int
    samples: int
    tile_images: int
    mask_files: int
    skipped_empty_or_small_masks: int


def build_sam_training_data(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    include_unlabeled: bool = False,
    min_mask_pixels: int = 64,
    positive_points: int = DEFAULT_POSITIVE_POINTS,
    negative_points: int = DEFAULT_NEGATIVE_POINTS,
    max_records: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    class_ids: Sequence[int] | None = None,
    tile_grid: int = 1,
    progress: ProgressReporter | None = None,
) -> SamTrainingDataResult:
    if min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be at least 1")
    if positive_points < 1:
        raise ValueError("positive_points must be at least 1")
    if negative_points < 0:
        raise ValueError("negative_points must be non-negative")
    if tile_grid < 1:
        raise ValueError("tile_grid must be positive")

    selected_class_ids = _selected_class_ids(
        include_unlabeled=include_unlabeled,
        class_ids=class_ids,
    )
    output = Path(output_dir)
    image_dir = output / "images"
    mask_dir = output / "masks"
    records_path = output / "records.jsonl"
    samples = read_manifest(manifest_path)
    records: list[SamMaskRecord] = []
    skipped = 0
    tile_images = 0

    _report(progress, f"Reading {len(samples)} prepared samples from {manifest_path}")
    import torch

    for sample_index, sample in enumerate(samples, start=1):
        prepared = load_prepared_sample(sample.prepared_path, torch=torch)
        label_mask = prepared.mask.long()
        _validate_mask_shape(label_mask, prepared.size, sample.prepared_path)
        region_key = infer_region_key(sample.image_path)
        tile_windows = _tile_windows(prepared.size, tile_grid)
        source_image = (
            _sample_image_for_tiling(sample.image_path, prepared.size) if tile_grid > 1 else None
        )

        for tile_row, tile_col, tile_xyxy in tile_windows:
            x0, y0, x1, y1 = tile_xyxy
            tile_labels = label_mask[y0:y1, x0:x1]
            tile_size = (x1 - x0, y1 - y0)
            tile_image_path = sample.image_path
            if tile_grid > 1:
                if source_image is None:
                    raise RuntimeError("source image is required when tile_grid > 1")
                tile_image_path = image_dir / _tile_image_name(
                    sample_index=sample_index,
                    sample_path=sample.image_path,
                    tile_row=tile_row,
                    tile_col=tile_col,
                )
                _write_tile_image(tile_image_path, source_image, tile_xyxy)
                tile_images += 1

            for class_id in selected_class_ids:
                class_mask = tile_labels == class_id
                pixel_count = int(class_mask.sum().item())
                if pixel_count < min_mask_pixels:
                    skipped += 1
                    continue

                class_name = OPENEARTHMAP_CLASSES[class_id]
                binary_mask_path = mask_dir / _record_mask_name(
                    sample_index=sample_index,
                    sample_path=sample.image_path,
                    class_id=class_id,
                    class_name=class_name,
                    tile_row=tile_row if tile_grid > 1 else None,
                    tile_col=tile_col if tile_grid > 1 else None,
                )
                _write_binary_mask(binary_mask_path, class_mask)
                records.append(
                    SamMaskRecord(
                        image_path=tile_image_path,
                        source_mask_path=sample.mask_path,
                        prepared_path=sample.prepared_path,
                        parent_image_path=sample.image_path if tile_grid > 1 else None,
                        mask_path=binary_mask_path,
                        split=sample.split,
                        source=sample.source,
                        class_id=class_id,
                        class_name=class_name,
                        area=pixel_count,
                        image_size=tile_size,
                        bbox_xyxy=_bbox_xyxy(class_mask),
                        positive_points_xy=_sample_points(class_mask, positive_points),
                        negative_points_xy=_sample_points(~class_mask, negative_points),
                        region_key=region_key,
                        image_key=image_key(tile_image_path),
                        label_source=(
                            "openearthmap-ground-truth-tile"
                            if tile_grid > 1
                            else "openearthmap-ground-truth"
                        ),
                        tile_row=tile_row if tile_grid > 1 else None,
                        tile_col=tile_col if tile_grid > 1 else None,
                        tile_grid=tile_grid if tile_grid > 1 else None,
                        tile_xyxy=tile_xyxy if tile_grid > 1 else None,
                    )
                )

        if source_image is not None:
            source_image.close()

        if _should_report(sample_index, len(samples)):
            _report(progress, f"Processed {sample_index}/{len(samples)} samples")

    selected_records = balanced_sam_records(records, max_records=max_records, seed=sample_seed)
    write_sam_mask_records(selected_records, records_path)
    return SamTrainingDataResult(
        output_dir=output,
        records_path=records_path,
        records=len(selected_records),
        samples=len(samples),
        tile_images=tile_images,
        mask_files=len(records),
        skipped_empty_or_small_masks=skipped,
    )


def _selected_class_ids(
    *,
    include_unlabeled: bool,
    class_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    selected = tuple(range(len(OPENEARTHMAP_CLASSES))) if class_ids is None else tuple(class_ids)
    invalid = [
        class_id for class_id in selected if class_id < 0 or class_id >= len(OPENEARTHMAP_CLASSES)
    ]
    if invalid:
        raise ValueError(f"class_ids contains unsupported OpenEarthMap class ids: {invalid}")
    if include_unlabeled:
        return selected
    return tuple(class_id for class_id in selected if class_id != 0)


def _validate_mask_shape(mask: Any, image_size: tuple[int, int], prepared_path: Path) -> None:
    height, width = tuple(mask.shape)
    expected_width, expected_height = image_size
    if width != expected_width or height != expected_height:
        raise RuntimeError(
            f"Prepared mask shape does not match image size for {prepared_path}: "
            f"mask={(width, height)} image={image_size}"
        )


def _write_binary_mask(path: Path, mask: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = tuple(mask.shape)
    values = mask.to(device="cpu", dtype=torch.uint8).mul(255).contiguous()
    Image.frombytes("L", (width, height), values.numpy().tobytes()).save(path)


def _tile_windows(
    image_size: tuple[int, int],
    tile_grid: int,
) -> tuple[tuple[int, int, tuple[int, int, int, int]], ...]:
    width, height = image_size
    if tile_grid > min(width, height):
        raise ValueError(f"tile_grid={tile_grid} is too large for image size {image_size}")
    windows: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for row in range(tile_grid):
        y0 = round(row * height / tile_grid)
        y1 = round((row + 1) * height / tile_grid)
        for col in range(tile_grid):
            x0 = round(col * width / tile_grid)
            x1 = round((col + 1) * width / tile_grid)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"tile_grid={tile_grid} produced an empty tile for {image_size}")
            windows.append((row, col, (x0, y0, x1, y1)))
    return tuple(windows)


def _sample_image_for_tiling(image_path: Path, image_size: tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.size != image_size:
        return image.resize(image_size, Image.Resampling.BILINEAR)
    return image


def _write_tile_image(
    path: Path,
    source_image: Image.Image,
    tile_xyxy: tuple[int, int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_image.crop(tile_xyxy).save(path)


def _tile_image_name(
    *,
    sample_index: int,
    sample_path: Path,
    tile_row: int,
    tile_col: int,
) -> str:
    fingerprint = hashlib.sha1(str(sample_path).encode("utf-8")).hexdigest()[:12]
    return (
        f"{sample_index:06d}-"
        f"{_slug(sample_path.stem)}-"
        f"tile-r{tile_row:02d}-c{tile_col:02d}-"
        f"{fingerprint}.png"
    )


def _record_mask_name(
    *,
    sample_index: int,
    sample_path: Path,
    class_id: int,
    class_name: str,
    tile_row: int | None = None,
    tile_col: int | None = None,
) -> str:
    fingerprint = hashlib.sha1(str(sample_path).encode("utf-8")).hexdigest()[:12]
    tile = f"tile-r{tile_row:02d}-c{tile_col:02d}-" if tile_row is not None else ""
    return (
        f"{sample_index:06d}-"
        f"{_slug(sample_path.stem)}-"
        f"{tile}"
        f"class-{class_id:02d}-{_slug(class_name)}-"
        f"{fingerprint}.png"
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "sample"


def _bbox_xyxy(mask: Any) -> tuple[int, int, int, int]:
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("cannot compute a bounding box for an empty mask")
    ys = coordinates[:, 0]
    xs = coordinates[:, 1]
    return (
        int(xs.min().item()),
        int(ys.min().item()),
        int(xs.max().item()),
        int(ys.max().item()),
    )


def _sample_points(mask: Any, count: int) -> tuple[tuple[int, int], ...]:
    if count == 0:
        return ()
    coordinates = mask.nonzero(as_tuple=False)
    total = int(coordinates.shape[0])
    if total == 0:
        return ()
    indexes = (
        (total // 2,)
        if count == 1
        else tuple(round(index * (total - 1) / (count - 1)) for index in range(count))
    )
    return tuple(
        (int(coordinates[index, 1].item()), int(coordinates[index, 0].item())) for index in indexes
    )


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == 1 or current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _build_click_command():
    import click

    @click.command(
        "sam-data",
        help="Split labeled OpenEarthMap masks into SAM-style binary training records.",
    )
    @click.option(
        "--manifest",
        type=click.Path(path_type=Path),
        default=Path("data/openearthmap/manifest.jsonl"),
        help="Prepared labeled OpenEarthMap manifest.",
    )
    @click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        default=Path("data/sam/openearthmap"),
        help="Directory for binary masks and records.jsonl.",
    )
    @click.option("--include-unlabeled", is_flag=True)
    @click.option("--min-mask-pixels", type=int, default=64, show_default=True)
    @click.option("--positive-points", type=int, default=DEFAULT_POSITIVE_POINTS, show_default=True)
    @click.option("--negative-points", type=int, default=DEFAULT_NEGATIVE_POINTS, show_default=True)
    @click.option("--max-records", type=int, default=None)
    @click.option("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, show_default=True)
    @click.option(
        "--tile-grid",
        type=int,
        default=1,
        show_default=True,
        help="Split each image into an NxN grid before writing SAM records.",
    )
    def command(
        manifest: Path,
        output_dir: Path,
        include_unlabeled: bool,
        min_mask_pixels: int,
        positive_points: int,
        negative_points: int,
        max_records: int | None,
        sample_seed: int,
        tile_grid: int,
    ) -> None:
        try:
            result = build_sam_training_data(
                manifest,
                output_dir,
                include_unlabeled=include_unlabeled,
                min_mask_pixels=min_mask_pixels,
                positive_points=positive_points,
                negative_points=negative_points,
                max_records=max_records,
                sample_seed=sample_seed,
                tile_grid=tile_grid,
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"records={result.records}")
        click.echo(f"records_path={result.records_path}")
        click.echo(f"tile_images={result.tile_images}")
        click.echo(f"mask_files={result.mask_files}")
        click.echo(f"skipped_empty_or_small_masks={result.skipped_empty_or_small_masks}")

    return command


cli = _build_click_command()
