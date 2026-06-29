"""Prepare custom labelled segmentation data for training."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import torch

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.model import ModelConfig

ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class SegmentationPreparationConfig:
    raw_manifest_path: str
    output_manifest_path: str
    prepared_dir: str
    source: str = "custom"
    force_prepared: bool = False


@dataclass(frozen=True)
class SegmentationPreparationResult:
    raw_manifest_path: str
    output_manifest_path: str
    prepared_dir: str
    samples: int
    splits: dict[str, int]
    sources: dict[str, int]


@dataclass(frozen=True)
class RawSegmentationRecord:
    image_path: Path
    mask_path: Path
    split: str
    source: str


def prepare_segmentation_manifest(
    config: SegmentationPreparationConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationPreparationResult:
    _validate_preparation_config(config)
    raw_manifest_path = Path(config.raw_manifest_path)
    output_manifest_path = Path(config.output_manifest_path)
    prepared_dir = Path(config.prepared_dir)
    records = _read_raw_segmentation_records(raw_manifest_path, default_source=config.source)
    if not records:
        raise RuntimeError(f"No raw segmentation records found in {raw_manifest_path}")

    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    samples: list[SatelliteSample] = []
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        sample = SatelliteSample(
            image_path=record.image_path,
            mask_path=record.mask_path,
            prepared_path=_prepared_path_for_raw_record(record, prepared_dir, index=index),
            split=record.split,
            source=record.source,
        )
        write_prepared_sample(
            sample,
            torch=torch,
            config=model_config,
            force=config.force_prepared,
        )
        samples.append(sample)
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
        if _should_report(index, len(records)):
            _report(progress, f"Prepared segmentation sample {index}/{len(records)}")

    write_manifest(samples, output_manifest_path)
    return SegmentationPreparationResult(
        raw_manifest_path=str(raw_manifest_path),
        output_manifest_path=str(output_manifest_path),
        prepared_dir=str(prepared_dir),
        samples=len(samples),
        splits=split_counts,
        sources=source_counts,
    )


def _read_raw_segmentation_records(
    raw_manifest_path: Path,
    *,
    default_source: str,
) -> tuple[RawSegmentationRecord, ...]:
    root = raw_manifest_path.parent
    records: list[RawSegmentationRecord] = []
    with raw_manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{raw_manifest_path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{raw_manifest_path}:{line_number}: record must be an object")
            image = _required_manifest_string(payload, "image", raw_manifest_path, line_number)
            mask = _required_manifest_string(payload, "mask", raw_manifest_path, line_number)
            split = str(payload.get("split", "train")).strip() or "train"
            source = str(payload.get("source", default_source)).strip() or default_source
            records.append(
                RawSegmentationRecord(
                    image_path=_resolve_manifest_path(root, image),
                    mask_path=_resolve_manifest_path(root, mask),
                    split=split,
                    source=source,
                )
            )
    return tuple(records)


def _required_manifest_string(
    payload: Mapping[str, Any],
    name: str,
    path: Path,
    line_number: int,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number}: {name} is required")
    return value


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _prepared_path_for_raw_record(
    record: RawSegmentationRecord,
    prepared_dir: Path,
    *,
    index: int,
) -> Path:
    source = _safe_path_component(record.source)
    split = _safe_path_component(record.split)
    stem = _safe_path_component(record.image_path.stem)
    return prepared_dir / source / split / f"{index:06d}-{stem}.pt"


def _safe_path_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )
    return safe.strip(".-") or "sample"


def _validate_preparation_config(config: SegmentationPreparationConfig) -> None:
    if not config.raw_manifest_path:
        raise ValueError("raw_manifest_path is required")
    if not config.output_manifest_path:
        raise ValueError("output_manifest_path is required")
    if not config.prepared_dir:
        raise ValueError("prepared_dir is required")
    if not config.source:
        raise ValueError("source is required")


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


@click.command("segmentation-prepare", help="Prepare a raw segmentation manifest for training.")
@click.argument("raw_manifest_path", type=click.Path(path_type=Path))
@click.option(
    "--output-manifest",
    "output_manifest_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--prepared-dir", type=click.Path(path_type=Path), required=True)
@click.option("--source", default="custom", show_default=True)
@click.option("--force-prepared", is_flag=True, help="Rewrite prepared sample data files.")
def cli(
    raw_manifest_path: Path,
    output_manifest_path: Path,
    prepared_dir: Path,
    source: str,
    force_prepared: bool,
) -> None:
    try:
        result = prepare_segmentation_manifest(
            SegmentationPreparationConfig(
                raw_manifest_path=str(raw_manifest_path),
                output_manifest_path=str(output_manifest_path),
                prepared_dir=str(prepared_dir),
                source=source,
                force_prepared=force_prepared,
            ),
            progress=click.echo,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
