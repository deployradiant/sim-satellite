"""Download, tile, and prepare LandCover.ai for non-commercial POC training.

LandCover.ai is licensed CC BY-NC-SA 4.0. Its label ``0`` is background /
unknown land cover, so this importer maps it to ``unlabeled`` rather than using
it as negative supervision for the other OpenEarthMap classes.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS

LANDCOVERAI_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
LANDCOVERAI_LICENSE = "CC BY-NC-SA 4.0; proof-of-concept/non-commercial use only."
LANDCOVERAI_TILE_SIZE = 512
LANDCOVERAI_EXPECTED_SPLIT_COUNTS = {"train": 7470, "val": 1602, "test": 1602}
LANDCOVERAI_EXPECTED_TOTAL_TILES = sum(LANDCOVERAI_EXPECTED_SPLIT_COUNTS.values())
LANDCOVERAI_EXPECTED_SOURCE_SCENES = 41
_EXTRACT_MARKER = ".landcoverai-extract-complete.json"
ProgressReporter = Callable[[str], None]

LANDCOVERAI_TO_OPENEARTHMAP = {
    0: OPENEARTHMAP_CLASS_IDS["unlabeled"],  # background / unknown
    1: OPENEARTHMAP_CLASS_IDS["building"],
    2: OPENEARTHMAP_CLASS_IDS["tree"],
    3: OPENEARTHMAP_CLASS_IDS["water"],
    4: OPENEARTHMAP_CLASS_IDS["road"],
}
LANDCOVERAI_CLASS_MAP = {
    "background": "unlabeled",
    "building": "building",
    "woodland": "tree",
    "water": "water",
    "road": "road",
}


@dataclass(frozen=True)
class LandCoverAIResult:
    data_dir: Path
    extracted_dir: Path
    tile_dir: Path
    prepared_dir: Path
    manifest_path: Path
    samples: int
    prepared_samples: int
    total_tiles: int
    skipped_tiles: int
    supervised_tiles: int
    min_labeled_fraction: float
    split_counts: dict[str, int]


@dataclass(frozen=True)
class LandCoverAIScan:
    samples: tuple[_LandCoverAITile, ...]
    total_tiles: int
    skipped_tiles: int
    supervised_tiles: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class _LandCoverAITile:
    tile_id: str
    split: str
    image_path: Path
    mask_path: Path
    labeled_fraction: float


@dataclass(frozen=True)
class _LandCoverAIPaths:
    data_dir: Path
    archive_path: Path
    extracted_dir: Path
    tile_dir: Path
    prepared_dir: Path
    manifest_path: Path


def download_and_prepare_landcoverai(
    data_dir: str | Path = "data/landcoverai",
    *,
    download: bool = True,
    extract: bool = True,
    force_download: bool = False,
    force_extract: bool = False,
    force_prepared: bool = False,
    min_labeled_fraction: float = 0.0,
    validate_official_counts: bool = True,
    progress: ProgressReporter | None = None,
) -> LandCoverAIResult:
    if not 0.0 <= min_labeled_fraction <= 1.0:
        raise ValueError("min_labeled_fraction must be between 0 and 1")

    paths = _paths(data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Preparing LandCover.ai data in {paths.data_dir}")
    _report(progress, f"License: {LANDCOVERAI_LICENSE}")

    if download:
        _download(paths.archive_path, force=force_download, progress=progress)
    elif not paths.archive_path.exists() and extract:
        raise FileNotFoundError(f"LandCover.ai archive not found: {paths.archive_path}")
    else:
        _report(progress, "Skipping download")

    if extract:
        _extract(paths.archive_path, paths.extracted_dir, force=force_extract, progress=progress)
    elif not paths.extracted_dir.exists():
        raise FileNotFoundError(
            f"LandCover.ai extracted directory not found: {paths.extracted_dir}"
        )
    else:
        _report(progress, f"Skipping extraction; using {paths.extracted_dir}")

    scan = scan_landcoverai(
        paths.extracted_dir,
        tile_dir=paths.tile_dir,
        force=force_prepared,
        min_labeled_fraction=min_labeled_fraction,
        validate_official_counts=validate_official_counts,
        progress=progress,
    )
    if not scan.samples:
        raise RuntimeError(f"No LandCover.ai tiles found under {paths.extracted_dir}")

    prepared_samples = _prepare_sample_data(
        scan.samples,
        prepared_dir=paths.prepared_dir,
        tile_dir=paths.tile_dir,
        force=force_prepared,
        progress=progress,
    )
    write_manifest(prepared_samples, paths.manifest_path)
    _report(
        progress,
        "Done: "
        f"{len(prepared_samples)} kept tiles, "
        f"{scan.total_tiles} total split tiles, "
        f"{scan.supervised_tiles} with supervised pixels, "
        f"{scan.skipped_tiles} skipped",
    )

    return LandCoverAIResult(
        data_dir=paths.data_dir,
        extracted_dir=paths.extracted_dir,
        tile_dir=paths.tile_dir,
        prepared_dir=paths.prepared_dir,
        manifest_path=paths.manifest_path,
        samples=len(prepared_samples),
        prepared_samples=len(prepared_samples),
        total_tiles=scan.total_tiles,
        skipped_tiles=scan.skipped_tiles,
        supervised_tiles=scan.supervised_tiles,
        min_labeled_fraction=min_labeled_fraction,
        split_counts=scan.split_counts,
    )


def scan_landcoverai(
    extracted_dir: str | Path,
    *,
    tile_dir: str | Path,
    force: bool = False,
    min_labeled_fraction: float = 0.0,
    validate_official_counts: bool = True,
    progress: ProgressReporter | None = None,
) -> LandCoverAIScan:
    root = Path(extracted_dir)
    output_root = Path(tile_dir)
    samples: list[_LandCoverAITile] = []
    split_counts: Counter[str] = Counter()
    total_tiles = 0
    skipped_tiles = 0
    supervised_tiles = 0
    split_ids_by_split = {
        split: _read_split_ids(root / f"{split}.txt", required=validate_official_counts)
        for split in ("train", "val", "test")
    }
    if validate_official_counts:
        _validate_official_split_counts(split_ids_by_split)
    expected_total_tiles = sum(len(tile_ids) for tile_ids in split_ids_by_split.values())

    for split, tile_ids in split_ids_by_split.items():
        for tile_id in tile_ids:
            total_tiles += 1
            scene, tile_index = _parse_tile_id(tile_id)
            image_path = root / "images" / f"{scene}.tif"
            mask_path = root / "masks" / f"{scene}.tif"
            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError(
                    f"LandCover.ai split references missing source files for {tile_id}: "
                    f"{image_path} / {mask_path}"
                )
            tile = _write_tile(
                tile_id,
                split=split,
                image_path=image_path,
                mask_path=mask_path,
                tile_index=tile_index,
                output_root=output_root,
                force=force,
            )
            if tile.labeled_fraction > 0:
                supervised_tiles += 1
            if tile.labeled_fraction < min_labeled_fraction:
                skipped_tiles += 1
                continue
            samples.append(tile)
            split_counts[split] += 1
            if _should_report(total_tiles, expected_total_tiles):
                _report(
                    progress,
                    f"Scanned {total_tiles}/{expected_total_tiles} LandCover.ai split tiles; "
                    f"kept {len(samples)}",
                )

    return LandCoverAIScan(
        samples=tuple(samples),
        total_tiles=total_tiles,
        skipped_tiles=skipped_tiles,
        supervised_tiles=supervised_tiles,
        split_counts=dict(sorted(split_counts.items())),
    )


def remap_landcoverai_mask(mask: Image.Image) -> Image.Image:
    table = [
        LANDCOVERAI_TO_OPENEARTHMAP.get(value, OPENEARTHMAP_CLASS_IDS["unlabeled"])
        for value in range(256)
    ]
    return mask.convert("L").point(table, mode="L")


def _write_tile(
    tile_id: str,
    *,
    split: str,
    image_path: Path,
    mask_path: Path,
    tile_index: int,
    output_root: Path,
    force: bool,
) -> _LandCoverAITile:
    image_output = output_root / split / "images" / f"{tile_id}.png"
    mask_output = output_root / split / "masks" / f"{tile_id}.png"
    if image_output.exists() and mask_output.exists() and not force:
        with Image.open(mask_output) as cached_mask:
            labeled_fraction = _labeled_fraction(cached_mask)
        return _LandCoverAITile(
            tile_id=tile_id,
            split=split,
            image_path=image_output,
            mask_path=mask_output,
            labeled_fraction=labeled_fraction,
        )

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        crop_box = _crop_box_for_tile(image.size, tile_index)
        if crop_box[2] > image.width or crop_box[3] > image.height:
            raise RuntimeError(
                f"LandCover.ai split tile {tile_id} is outside source image bounds: "
                f"crop={crop_box}, image={image.size}"
            )
        image_tile = image.crop(crop_box).convert("RGB")
        mask_tile = mask.crop(crop_box)
        remapped_mask = remap_landcoverai_mask(mask_tile)
        labeled_fraction = _labeled_fraction(remapped_mask)

        image_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        image_tile.save(image_output)
        remapped_mask.save(mask_output)

    return _LandCoverAITile(
        tile_id=tile_id,
        split=split,
        image_path=image_output,
        mask_path=mask_output,
        labeled_fraction=labeled_fraction,
    )


def _prepare_sample_data(
    samples: tuple[_LandCoverAITile, ...],
    *,
    prepared_dir: Path,
    tile_dir: Path,
    force: bool,
    progress: ProgressReporter | None,
) -> tuple[SatelliteSample, ...]:
    import torch

    from sim_satellite.data.prepared import write_prepared_sample
    from sim_satellite.model import ModelConfig

    config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASS_IDS))
    prepared_samples: list[SatelliteSample] = []
    for index, tile in enumerate(samples, start=1):
        prepared = SatelliteSample(
            image_path=tile.image_path,
            mask_path=tile.mask_path,
            prepared_path=_derived_path(
                tile.image_path, source_dir=tile_dir, target_dir=prepared_dir
            ).with_suffix(".pt"),
            split=tile.split,
            source="landcoverai",
        )
        write_prepared_sample(prepared, torch=torch, config=config, force=force)
        prepared_samples.append(prepared)
        if _should_report(index, len(samples)):
            _report(progress, f"Prepared tensors for {index}/{len(samples)} LandCover.ai tiles")
    return tuple(prepared_samples)


def _download(
    archive_path: Path,
    *,
    force: bool,
    progress: ProgressReporter | None,
) -> None:
    if archive_path.exists() and not force:
        _report(progress, f"Using cached archive {archive_path}")
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    _report(progress, f"Downloading LandCover.ai archive to {archive_path}")
    with urllib.request.urlopen(LANDCOVERAI_URL) as response, partial_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial_path.replace(archive_path)


def _extract(
    archive_path: Path,
    extracted_dir: Path,
    *,
    force: bool,
    progress: ProgressReporter | None,
) -> None:
    marker_path = extracted_dir / _EXTRACT_MARKER
    if extracted_dir.exists() and force:
        shutil.rmtree(extracted_dir)
    if marker_path.exists() and not force:
        _validate_extract_marker(marker_path, archive_path)
        _report(progress, f"Using cached extracted data {extracted_dir}")
        return
    if extracted_dir.exists() and any(extracted_dir.iterdir()) and not force:
        raise RuntimeError(
            f"LandCover.ai extracted directory exists without a valid completion marker: "
            f"{extracted_dir}. Re-run with --force-extract."
        )
    with zipfile.ZipFile(archive_path) as archive:
        _assert_safe_zip_members(archive, extracted_dir)
        _validate_archive_members(archive)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extracted_dir.resolve())
    _write_extract_marker(marker_path, archive_path)


def _assert_safe_zip_members(archive: zipfile.ZipFile, extracted_dir: Path) -> None:
    destination = extracted_dir.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")


def _validate_archive_members(archive: zipfile.ZipFile) -> None:
    names = {member.filename for member in archive.infolist()}
    missing = sorted(
        f"{split}.txt" for split in ("train", "val", "test") if f"{split}.txt" not in names
    )
    if missing:
        raise RuntimeError(f"LandCover.ai archive is missing required split files: {missing}")

    images = {
        Path(name).stem for name in names if name.startswith("images/") and name.endswith(".tif")
    }
    masks = {
        Path(name).stem for name in names if name.startswith("masks/") and name.endswith(".tif")
    }
    if len(images) != LANDCOVERAI_EXPECTED_SOURCE_SCENES:
        raise RuntimeError(
            f"Expected {LANDCOVERAI_EXPECTED_SOURCE_SCENES} LandCover.ai source images, "
            f"found {len(images)}"
        )
    if len(masks) != LANDCOVERAI_EXPECTED_SOURCE_SCENES:
        raise RuntimeError(
            f"Expected {LANDCOVERAI_EXPECTED_SOURCE_SCENES} LandCover.ai source masks, "
            f"found {len(masks)}"
        )
    if images != masks:
        missing_masks = sorted(images - masks)
        missing_images = sorted(masks - images)
        raise RuntimeError(
            "LandCover.ai archive image/mask scenes do not match: "
            f"missing_masks={missing_masks}, missing_images={missing_images}"
        )


def _write_extract_marker(marker_path: Path, archive_path: Path) -> None:
    payload = {
        "archive": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "dataset": "landcoverai-v1",
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate_extract_marker(marker_path: Path, archive_path: Path) -> None:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid LandCover.ai extraction marker: {marker_path}") from exc

    expected = {
        "archive": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "dataset": "landcoverai-v1",
    }
    if payload != expected:
        raise RuntimeError(
            f"LandCover.ai extraction marker does not match {archive_path}. "
            "Re-run with --force-extract."
        )


def _paths(data_dir: str | Path) -> _LandCoverAIPaths:
    root = Path(data_dir)
    return _LandCoverAIPaths(
        data_dir=root,
        archive_path=root / "raw" / "landcover.ai.v1.zip",
        extracted_dir=root / "extracted",
        tile_dir=root / "tiles",
        prepared_dir=root / "prepared",
        manifest_path=root / "manifest.jsonl",
    )


def _read_split_ids(path: Path, *, required: bool) -> list[str]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"LandCover.ai split file not found: {path}")
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_official_split_counts(split_ids_by_split: dict[str, list[str]]) -> None:
    counts = {split: len(tile_ids) for split, tile_ids in split_ids_by_split.items()}
    if counts != LANDCOVERAI_EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Unexpected LandCover.ai split counts. "
            f"expected={LANDCOVERAI_EXPECTED_SPLIT_COUNTS}, found={counts}"
        )


def _parse_tile_id(tile_id: str) -> tuple[str, int]:
    scene, index = tile_id.rsplit("_", 1)
    return scene, int(index)


def _crop_box_for_tile(size: tuple[int, int], tile_index: int) -> tuple[int, int, int, int]:
    width, _height = size
    columns = (width + LANDCOVERAI_TILE_SIZE - 1) // LANDCOVERAI_TILE_SIZE
    x = (tile_index % columns) * LANDCOVERAI_TILE_SIZE
    y = (tile_index // columns) * LANDCOVERAI_TILE_SIZE
    return (x, y, x + LANDCOVERAI_TILE_SIZE, y + LANDCOVERAI_TILE_SIZE)


def _labeled_fraction(mask: Image.Image) -> float:
    histogram = mask.convert("L").histogram()[:256]
    total = sum(histogram)
    if total == 0:
        return 0.0
    labeled = total - histogram[OPENEARTHMAP_CLASS_IDS["unlabeled"]]
    return labeled / total


def _derived_path(path: Path, *, source_dir: Path, target_dir: Path) -> Path:
    try:
        relative_path = path.relative_to(source_dir)
    except ValueError:
        relative_path = path.name
    return target_dir / relative_path


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == total or total <= 10:
        return True
    return current % max(1, total // 10) == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _build_click_command():
    import click

    @click.command("landcoverai", help="Download and prepare the LandCover.ai manifest.")
    @click.option(
        "--data-dir",
        type=click.Path(path_type=Path),
        default=Path("data/landcoverai"),
    )
    @click.option("--skip-download", is_flag=True, help="Use an existing raw/landcover.ai.v1.zip.")
    @click.option("--skip-extract", is_flag=True, help="Use an existing extracted directory.")
    @click.option("--force-download", is_flag=True, help="Replace the existing archive.")
    @click.option("--force-extract", is_flag=True, help="Replace the extracted directory.")
    @click.option("--force-prepared", is_flag=True, help="Rewrite tiles and prepared tensors.")
    @click.option(
        "--min-labeled-fraction",
        type=float,
        default=0.0,
        show_default=True,
        help="Skip tiles below this supervised-pixel fraction.",
    )
    @click.option(
        "--skip-official-count-validation",
        is_flag=True,
        help="Allow non-official miniature fixtures or custom split counts.",
    )
    def command(
        data_dir: Path,
        skip_download: bool,
        skip_extract: bool,
        force_download: bool,
        force_extract: bool,
        force_prepared: bool,
        min_labeled_fraction: float,
        skip_official_count_validation: bool,
    ) -> None:
        try:
            result = download_and_prepare_landcoverai(
                data_dir,
                download=not skip_download,
                extract=not skip_extract,
                force_download=force_download,
                force_extract=force_extract,
                force_prepared=force_prepared,
                min_labeled_fraction=min_labeled_fraction,
                validate_official_counts=not skip_official_count_validation,
                progress=click.echo,
            )
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"manifest={result.manifest_path}")
        click.echo(f"prepared={result.prepared_dir}")
        click.echo(f"samples={result.samples}")
        click.echo(f"prepared_samples={result.prepared_samples}")
        click.echo(f"total_tiles={result.total_tiles}")
        click.echo(f"supervised_tiles={result.supervised_tiles}")
        click.echo(f"skipped_tiles={result.skipped_tiles}")
        click.echo(f"split_counts={result.split_counts}")

    return command


cli = _build_click_command()
