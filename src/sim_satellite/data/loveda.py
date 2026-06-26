"""Download and prepare LoveDA for non-commercial proof-of-concept training.

LoveDA is described by its Zenodo record as academic-only, non-commercial data
with Google Earth terms applying to the source imagery. Keep downloaded archives,
extracted images, remapped masks, and prepared tensors under ``data/loveda/``.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sim_satellite.data.classes import OPENEARTHMAP_CLASS_IDS
from sim_satellite.data.loader import SatelliteSample, write_manifest

LOVEDA_URLS = {
    "train": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
    "val": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
}
LOVEDA_EXPECTED_LABELED_SAMPLES = 4191
LOVEDA_EXPECTED_SPLIT_COUNTS = {"train": 2522, "val": 1669}
LOVEDA_LICENSE = (
    "Academic-only/non-commercial; Zenodo description states CC BY-NC-SA 4.0 "
    "and Google Earth terms apply to source imagery."
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".tif", ".tiff"}
EXTRACT_COMPLETE_MARKER = ".loveda-extract-complete.json"
ProgressReporter = Callable[[str], None]

LOVEDA_TO_OPENEARTHMAP = {
    0: OPENEARTHMAP_CLASS_IDS["unlabeled"],  # no-data
    1: OPENEARTHMAP_CLASS_IDS["unlabeled"],  # background / unknown
    2: OPENEARTHMAP_CLASS_IDS["building"],
    3: OPENEARTHMAP_CLASS_IDS["road"],
    4: OPENEARTHMAP_CLASS_IDS["water"],
    5: OPENEARTHMAP_CLASS_IDS["bareland"],
    6: OPENEARTHMAP_CLASS_IDS["tree"],
    7: OPENEARTHMAP_CLASS_IDS["agriculture land"],
}
LOVEDA_MASK_VALUES = frozenset(LOVEDA_TO_OPENEARTHMAP)


@dataclass(frozen=True)
class LoveDAResult:
    data_dir: Path
    extracted_dir: Path
    remapped_dir: Path
    prepared_dir: Path
    manifest_path: Path
    samples: int
    prepared_samples: int
    scanned_images: int
    missing_masks: int
    image_dirs_without_masks: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class LoveDAScan:
    samples: tuple[_LoveDAPair, ...]
    scanned_images: int
    missing_masks: int
    image_dirs_without_masks: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class _LoveDAPair:
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class _LoveDAPaths:
    data_dir: Path
    archive_paths: dict[str, Path]
    extracted_dir: Path
    remapped_dir: Path
    prepared_dir: Path
    manifest_path: Path


def download_and_prepare_loveda(
    data_dir: str | Path = "data/loveda",
    *,
    download: bool = True,
    extract: bool = True,
    force_download: bool = False,
    force_extract: bool = False,
    force_prepared: bool = False,
    validate_official_counts: bool = False,
    progress: ProgressReporter | None = None,
) -> LoveDAResult:
    paths = _paths(data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Preparing LoveDA data in {paths.data_dir}")
    _report(progress, f"License: {LOVEDA_LICENSE}")

    if download:
        for split, url in LOVEDA_URLS.items():
            _download(paths.archive_paths[split], url=url, force=force_download, progress=progress)
    elif extract:
        for archive_path in paths.archive_paths.values():
            if not archive_path.exists():
                raise FileNotFoundError(f"LoveDA archive not found: {archive_path}")
    else:
        _report(progress, "Skipping download")

    if extract:
        _extract(
            paths.archive_paths.values(),
            paths.extracted_dir,
            force=force_extract,
            progress=progress,
        )
    elif not paths.extracted_dir.exists():
        raise FileNotFoundError(f"LoveDA extracted directory not found: {paths.extracted_dir}")
    else:
        _report(progress, f"Skipping extraction; using {paths.extracted_dir}")

    scan = scan_loveda(paths.extracted_dir, progress=progress)
    if scan.image_dirs_without_masks:
        raise RuntimeError(
            "LoveDA scan found image directories without matching masks: "
            f"{scan.image_dirs_without_masks}"
        )
    if not scan.samples:
        raise RuntimeError(f"No LoveDA image/mask pairs found under {paths.extracted_dir}")
    if validate_official_counts:
        _validate_official_counts(scan.split_counts)

    prepared_samples = _prepare_sample_data(
        scan.samples,
        remapped_dir=paths.remapped_dir,
        prepared_dir=paths.prepared_dir,
        extracted_dir=paths.extracted_dir,
        force=force_prepared,
        progress=progress,
    )
    write_manifest(prepared_samples, paths.manifest_path)
    _report(
        progress,
        "Done: "
        f"{len(prepared_samples)} samples, "
        f"{scan.scanned_images} images scanned, "
        f"{scan.missing_masks} images skipped without masks",
    )

    return LoveDAResult(
        data_dir=paths.data_dir,
        extracted_dir=paths.extracted_dir,
        remapped_dir=paths.remapped_dir,
        prepared_dir=paths.prepared_dir,
        manifest_path=paths.manifest_path,
        samples=len(prepared_samples),
        prepared_samples=len(prepared_samples),
        scanned_images=scan.scanned_images,
        missing_masks=scan.missing_masks,
        image_dirs_without_masks=scan.image_dirs_without_masks,
        split_counts=scan.split_counts,
    )


def scan_loveda(
    extracted_dir: str | Path,
    *,
    progress: ProgressReporter | None = None,
) -> LoveDAScan:
    root = Path(extracted_dir)
    pairs: list[_LoveDAPair] = []
    split_counts: Counter[str] = Counter()
    scanned_images = 0
    missing_masks = 0
    image_dirs_without_masks = 0

    image_dirs = sorted(path for path in root.rglob("*") if _is_image_dir(path))
    _report(progress, f"Found {len(image_dirs)} LoveDA image directories")
    for image_dir in image_dirs:
        mask_dir = _matching_mask_dir(image_dir)
        if mask_dir is None:
            image_dirs_without_masks += 1
            continue
        for image_path in sorted(_iter_files(image_dir, IMAGE_SUFFIXES)):
            scanned_images += 1
            mask_path = _matching_mask(image_path, image_dir=image_dir, mask_dir=mask_dir)
            if mask_path is None:
                missing_masks += 1
                continue
            split = _split_from_path(image_path)
            pairs.append(_LoveDAPair(image_path=image_path, mask_path=mask_path, split=split))
            split_counts[split] += 1

    return LoveDAScan(
        samples=tuple(pairs),
        scanned_images=scanned_images,
        missing_masks=missing_masks,
        image_dirs_without_masks=image_dirs_without_masks,
        split_counts=dict(sorted(split_counts.items())),
    )


def remap_loveda_mask(mask_path: Path, output_path: Path, *, force: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = [LOVEDA_TO_OPENEARTHMAP.get(value, 0) for value in range(256)]
    with Image.open(mask_path) as mask:
        class_id_mask = _loveda_class_id_mask(mask, mask_path=mask_path)
        remapped = class_id_mask.point(table, mode="L")
        remapped.save(output_path)
    return output_path


def _loveda_class_id_mask(mask: Image.Image, *, mask_path: Path) -> Image.Image:
    if mask.mode == "L":
        class_id_mask = mask.copy()
    elif mask.mode == "P":
        class_id_mask = Image.new("L", mask.size)
        class_id_mask.putdata(mask.getdata())
    else:
        raise ValueError(
            f"LoveDA mask must store class ids in L or P mode: {mask_path} has mode {mask.mode!r}"
        )

    unexpected = _unexpected_loveda_values(class_id_mask)
    if unexpected:
        values = ", ".join(str(value) for value in unexpected[:8])
        suffix = "" if len(unexpected) <= 8 else ", ..."
        raise ValueError(
            f"LoveDA mask contains unexpected class ids in {mask_path}: {values}{suffix}"
        )
    return class_id_mask


def _unexpected_loveda_values(mask: Image.Image) -> list[int]:
    histogram = mask.histogram()[:256]
    return [
        value for value, count in enumerate(histogram) if count and value not in LOVEDA_MASK_VALUES
    ]


def _prepare_sample_data(
    samples: tuple[_LoveDAPair, ...],
    *,
    remapped_dir: Path,
    prepared_dir: Path,
    extracted_dir: Path,
    force: bool,
    progress: ProgressReporter | None,
) -> tuple[SatelliteSample, ...]:
    import torch

    from sim_satellite.data.prepared import write_prepared_sample
    from sim_satellite.model import ModelConfig

    config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASS_IDS))
    prepared_samples: list[SatelliteSample] = []
    for index, sample in enumerate(samples, start=1):
        remapped_mask = remap_loveda_mask(
            sample.mask_path,
            _derived_path(sample.mask_path, source_dir=extracted_dir, target_dir=remapped_dir),
            force=force,
        )
        prepared = SatelliteSample(
            image_path=sample.image_path,
            mask_path=remapped_mask,
            prepared_path=_derived_path(
                sample.image_path, source_dir=extracted_dir, target_dir=prepared_dir
            ).with_suffix(".pt"),
            split=sample.split,
            source="loveda",
        )
        write_prepared_sample(prepared, torch=torch, config=config, force=force)
        prepared_samples.append(prepared)
        if _should_report(index, len(samples)):
            _report(progress, f"Prepared {index}/{len(samples)} LoveDA samples")
    return tuple(prepared_samples)


def _download(
    archive_path: Path,
    *,
    url: str,
    force: bool,
    progress: ProgressReporter | None,
) -> None:
    if archive_path.exists() and not force:
        _report(progress, f"Using cached archive {archive_path}")
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    _report(progress, f"Downloading {url} to {archive_path}")
    with urllib.request.urlopen(url) as response, partial_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial_path.replace(archive_path)


def _extract(
    archive_paths: Iterable[Path],
    extracted_dir: Path,
    *,
    force: bool,
    progress: ProgressReporter | None,
) -> None:
    archive_path_tuple = tuple(archive_paths)
    replace_existing = extracted_dir.exists() and any(extracted_dir.iterdir())
    with ExitStack() as stack:
        archives = tuple(stack.enter_context(zipfile.ZipFile(path)) for path in archive_path_tuple)
        if replace_existing and not force:
            if _extraction_complete(extracted_dir, archives):
                _report(progress, f"Using cached extracted data {extracted_dir}")
                return
            _report(progress, f"Cached extracted data is incomplete; replacing {extracted_dir}")
        elif force and extracted_dir.exists():
            _report(progress, f"Replacing extracted data {extracted_dir}")

        if replace_existing:
            shutil.rmtree(extracted_dir)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            _report(progress, f"Extracting {archive.filename} to {extracted_dir}")
            _safe_extract(archive, extracted_dir)
        _write_extraction_marker(extracted_dir, archives)


def _safe_extract(archive: zipfile.ZipFile, extracted_dir: Path) -> None:
    destination = extracted_dir.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
    archive.extractall(destination)


def _extraction_complete(
    extracted_dir: Path,
    archives: tuple[zipfile.ZipFile, ...],
) -> bool:
    marker_path = extracted_dir / EXTRACT_COMPLETE_MARKER
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload != _archive_marker_payload(archives):
        return False
    return all(
        (extracted_dir / member.filename).exists()
        for archive in archives
        for member in archive.infolist()
        if not member.is_dir()
    )


def _write_extraction_marker(
    extracted_dir: Path,
    archives: tuple[zipfile.ZipFile, ...],
) -> None:
    marker_path = extracted_dir / EXTRACT_COMPLETE_MARKER
    marker_path.write_text(
        json.dumps(_archive_marker_payload(archives), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _archive_marker_payload(archives: tuple[zipfile.ZipFile, ...]) -> dict[str, object]:
    archive_payloads = []
    for archive in archives:
        if archive.filename is None:
            raise RuntimeError("Archive filename is unavailable")
        archive_path = Path(archive.filename)
        archive_payloads.append(
            {
                "archive": archive_path.name,
                "archive_size": archive_path.stat().st_size,
                "member_count": len(archive.infolist()),
            }
        )
    return {"archives": archive_payloads}


def _paths(data_dir: str | Path) -> _LoveDAPaths:
    root = Path(data_dir)
    return _LoveDAPaths(
        data_dir=root,
        archive_paths={
            "train": root / "raw" / "Train.zip",
            "val": root / "raw" / "Val.zip",
        },
        extracted_dir=root / "extracted",
        remapped_dir=root / "remapped",
        prepared_dir=root / "prepared",
        manifest_path=root / "manifest.jsonl",
    )


def _validate_official_counts(split_counts: dict[str, int]) -> None:
    if sum(split_counts.values()) != LOVEDA_EXPECTED_LABELED_SAMPLES:
        raise RuntimeError(
            "LoveDA labeled sample count does not match the official train+val layout: "
            f"expected {LOVEDA_EXPECTED_LABELED_SAMPLES}, got {sum(split_counts.values())}"
        )
    if split_counts != LOVEDA_EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "LoveDA split counts do not match the official train/val layout: "
            f"expected {LOVEDA_EXPECTED_SPLIT_COUNTS}, got {split_counts}"
        )


def _is_image_dir(path: Path) -> bool:
    return path.is_dir() and path.name.lower() in {"image", "images", "images_png"}


def _matching_mask_dir(image_dir: Path) -> Path | None:
    for name in ("masks_png", "masks", "mask", "labels", "labels_png"):
        candidate = image_dir.parent / name
        if candidate.is_dir():
            return candidate
    return None


def _iter_files(root: Path, suffixes: set[str]) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def _matching_mask(image_path: Path, *, image_dir: Path, mask_dir: Path) -> Path | None:
    relative_stem = image_path.relative_to(image_dir).with_suffix("")
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / relative_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return "train"


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

    @click.command("loveda", help="Download and prepare the LoveDA manifest.")
    @click.option("--data-dir", type=click.Path(path_type=Path), default=Path("data/loveda"))
    @click.option(
        "--skip-download", is_flag=True, help="Use existing raw/Train.zip and raw/Val.zip."
    )
    @click.option("--skip-extract", is_flag=True, help="Use an existing extracted directory.")
    @click.option("--force-download", is_flag=True, help="Replace existing archives.")
    @click.option("--force-extract", is_flag=True, help="Replace the extracted directory.")
    @click.option(
        "--force-prepared", is_flag=True, help="Rewrite remapped masks and prepared tensors."
    )
    @click.option(
        "--validate-official-counts",
        is_flag=True,
        help="Require the official 2,522 train and 1,669 val labeled sample counts.",
    )
    def command(
        data_dir: Path,
        skip_download: bool,
        skip_extract: bool,
        force_download: bool,
        force_extract: bool,
        force_prepared: bool,
        validate_official_counts: bool,
    ) -> None:
        try:
            result = download_and_prepare_loveda(
                data_dir,
                download=not skip_download,
                extract=not skip_extract,
                force_download=force_download,
                force_extract=force_extract,
                force_prepared=force_prepared,
                validate_official_counts=validate_official_counts,
                progress=click.echo,
            )
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"manifest={result.manifest_path}")
        click.echo(f"prepared={result.prepared_dir}")
        click.echo(f"samples={result.samples}")
        click.echo(f"prepared_samples={result.prepared_samples}")
        click.echo(f"scanned_images={result.scanned_images}")
        click.echo(f"missing_masks={result.missing_masks}")
        click.echo(f"split_counts={result.split_counts}")

    return command


cli = _build_click_command()
