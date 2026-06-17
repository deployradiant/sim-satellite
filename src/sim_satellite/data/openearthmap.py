"""Download and prepare OpenEarthMap for the first training runs."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sim_satellite.data.loader import SatelliteSample, write_manifest

OPENEARTHMAP_URL = "https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".tif", ".tiff"}
OPENEARTHMAP_CLASSES = (
    "unlabeled",
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "building",
)
OPENEARTHMAP_CLASS_IDS = {name: index for index, name in enumerate(OPENEARTHMAP_CLASSES)}
OPENEARTHMAP_RGB = {
    "unlabeled": (0, 0, 0),
    "bareland": (128, 0, 0),
    "rangeland": (0, 255, 36),
    "developed space": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "agriculture land": (75, 181, 73),
    "building": (222, 31, 7),
}
EXTRACT_COMPLETE_MARKER = ".openearthmap-extract-complete.json"
EXTRACT_FREE_SPACE_MARGIN_BYTES = 256 * 1024 * 1024
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class OpenEarthMapResult:
    data_dir: Path
    archive_path: Path
    extracted_dir: Path
    prepared_dir: Path
    manifest_path: Path
    samples: int
    prepared_samples: int
    scanned_images: int
    missing_masks: int
    image_dirs_without_labels: int


@dataclass(frozen=True)
class OpenEarthMapScan:
    samples: tuple[_OpenEarthMapPair, ...]
    scanned_images: int
    missing_masks: int
    image_dirs_without_labels: int


@dataclass(frozen=True)
class _OpenEarthMapPair:
    image_path: Path
    mask_path: Path
    split: str


def download_and_prepare_openearthmap(
    data_dir: str | Path = "data/openearthmap",
    *,
    download: bool = True,
    extract: bool = True,
    force_download: bool = False,
    force_extract: bool = False,
    force_prepared: bool = False,
    progress: ProgressReporter | None = None,
) -> OpenEarthMapResult:
    paths = _paths(data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Preparing OpenEarthMap data in {paths.data_dir}")

    if download:
        _download(paths.archive_path, force=force_download, progress=progress)
    elif not paths.archive_path.exists() and extract:
        raise FileNotFoundError(f"OpenEarthMap archive not found: {paths.archive_path}")
    else:
        _report(progress, "Skipping download")

    if extract:
        _extract(paths.archive_path, paths.extracted_dir, force=force_extract, progress=progress)
    elif not paths.extracted_dir.exists():
        raise FileNotFoundError(
            f"OpenEarthMap extracted directory not found: {paths.extracted_dir}"
        )
    else:
        _report(progress, f"Skipping extraction; using {paths.extracted_dir}")

    scan = scan_openearthmap(paths.extracted_dir, progress=progress)
    if scan.image_dirs_without_labels:
        raise RuntimeError(
            "OpenEarthMap scan found image directories without matching labels: "
            f"{scan.image_dirs_without_labels}"
        )
    if not scan.samples:
        raise RuntimeError(f"No OpenEarthMap image/mask pairs found under {paths.extracted_dir}")

    _report(progress, f"Preparing sample data in {paths.prepared_dir}")
    prepared_samples = _prepare_sample_data(
        scan.samples,
        prepared_dir=paths.prepared_dir,
        extracted_dir=paths.extracted_dir,
        force=force_prepared,
        progress=progress,
    )
    _report(progress, f"Writing manifest to {paths.manifest_path}")
    write_manifest(prepared_samples, paths.manifest_path)
    _report(
        progress,
        "Done: "
        f"{len(scan.samples)} samples, "
        f"{scan.scanned_images} images scanned, "
        f"{scan.missing_masks} images skipped without masks",
    )

    return OpenEarthMapResult(
        data_dir=paths.data_dir,
        archive_path=paths.archive_path,
        extracted_dir=paths.extracted_dir,
        prepared_dir=paths.prepared_dir,
        manifest_path=paths.manifest_path,
        samples=len(prepared_samples),
        prepared_samples=len(prepared_samples),
        scanned_images=scan.scanned_images,
        missing_masks=scan.missing_masks,
        image_dirs_without_labels=scan.image_dirs_without_labels,
    )


def scan_openearthmap(
    extracted_dir: str | Path,
    *,
    progress: ProgressReporter | None = None,
) -> OpenEarthMapScan:
    root = Path(extracted_dir)
    pairs: list[_OpenEarthMapPair] = []
    scanned_images = 0
    missing_masks = 0
    image_dirs_without_labels = 0

    _report(progress, f"Finding image directories under {root}")
    image_dirs = sorted(path for path in root.rglob("*") if _is_image_dir(path))
    _report(progress, f"Found {len(image_dirs)} image directories")

    for index, image_dir in enumerate(image_dirs, start=1):
        label_dir = _matching_label_dir(image_dir)
        if label_dir is None:
            image_dirs_without_labels += 1
            if _should_report(index, len(image_dirs)):
                _report_scan_progress(
                    progress,
                    current=index,
                    total=len(image_dirs),
                    scanned_images=scanned_images,
                    pairs=len(pairs),
                )
            continue

        for image_path in sorted(_iter_files(image_dir, IMAGE_SUFFIXES)):
            scanned_images += 1
            mask_path = _matching_mask(image_path, image_dir, label_dir)
            if mask_path is None:
                missing_masks += 1
                continue
            pairs.append(
                _OpenEarthMapPair(
                    image_path=image_path,
                    mask_path=mask_path,
                    split=_split_from_path(image_path),
                )
            )
        if _should_report(index, len(image_dirs)):
            _report_scan_progress(
                progress,
                current=index,
                total=len(image_dirs),
                scanned_images=scanned_images,
                pairs=len(pairs),
            )

    return OpenEarthMapScan(
        samples=tuple(pairs),
        scanned_images=scanned_images,
        missing_masks=missing_masks,
        image_dirs_without_labels=image_dirs_without_labels,
    )


def _prepare_sample_data(
    samples: tuple[_OpenEarthMapPair, ...],
    *,
    prepared_dir: Path,
    extracted_dir: Path,
    force: bool,
    progress: ProgressReporter | None,
) -> tuple[SatelliteSample, ...]:
    import torch

    from sim_satellite.data.prepared import write_prepared_sample
    from sim_satellite.model import ModelConfig

    config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    prepared_samples: list[SatelliteSample] = []
    for index, sample in enumerate(samples, start=1):
        prepared = SatelliteSample(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            prepared_path=_prepared_path_for_sample(
                sample,
                prepared_dir=prepared_dir,
                extracted_dir=extracted_dir,
            ),
            split=sample.split,
            source="openearthmap",
        )
        write_prepared_sample(prepared, torch=torch, config=config, force=force)
        prepared_samples.append(prepared)
        if _should_report(index, len(samples)):
            _report(
                progress,
                f"Prepared {index}/{len(samples)} samples ({_percent(index, len(samples))}%)",
            )
    return tuple(prepared_samples)


def _prepared_path_for_sample(
    sample: _OpenEarthMapPair,
    *,
    prepared_dir: Path,
    extracted_dir: Path,
) -> Path:
    try:
        relative_path = sample.image_path.relative_to(extracted_dir)
    except ValueError:
        relative_path = Path(sample.split) / sample.image_path.name
    return prepared_dir / relative_path.with_suffix(".pt")


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
    _report(progress, f"Downloading OpenEarthMap archive to {archive_path}")

    with urllib.request.urlopen(OPENEARTHMAP_URL) as response:
        total_bytes = _content_length(response)
        downloaded_bytes = 0
        last_percent = -5
        last_reported_bytes = 0
        with partial_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_bytes += len(chunk)

                if total_bytes is not None:
                    percent = downloaded_bytes * 100 // total_bytes
                    if percent >= last_percent + 5 or downloaded_bytes == total_bytes:
                        _report(
                            progress,
                            "Downloaded "
                            f"{_format_bytes(downloaded_bytes)} / {_format_bytes(total_bytes)} "
                            f"({percent}%)",
                        )
                        last_percent = percent
                elif downloaded_bytes - last_reported_bytes >= 256 * 1024 * 1024:
                    _report(progress, f"Downloaded {_format_bytes(downloaded_bytes)}")
                    last_reported_bytes = downloaded_bytes
    partial_path.replace(archive_path)
    _report(progress, f"Download complete: {archive_path}")


def _extract(
    archive_path: Path,
    extracted_dir: Path,
    *,
    force: bool,
    progress: ProgressReporter | None,
) -> None:
    replace_existing = extracted_dir.exists() and any(extracted_dir.iterdir())
    with zipfile.ZipFile(archive_path) as archive:
        if replace_existing and not force:
            if _extraction_complete(extracted_dir, archive):
                _report(progress, f"Using cached extracted data {extracted_dir}")
                return
            _report(progress, f"Cached extracted data is incomplete; replacing {extracted_dir}")
        elif force and extracted_dir.exists():
            _report(progress, f"Replacing extracted data {extracted_dir}")

        _ensure_extraction_space(extracted_dir, archive, replacing=replace_existing)
        if replace_existing:
            shutil.rmtree(extracted_dir)

        extracted_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, extracted_dir, progress=progress)
        _write_extraction_marker(extracted_dir, archive)


def _ensure_extraction_space(
    extracted_dir: Path,
    archive: zipfile.ZipFile,
    *,
    replacing: bool,
) -> None:
    parent = extracted_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    available_bytes = shutil.disk_usage(parent).free
    if replacing and extracted_dir.exists():
        available_bytes += _path_size(extracted_dir)
    required_bytes = _archive_uncompressed_size(archive) + EXTRACT_FREE_SPACE_MARGIN_BYTES
    if available_bytes < required_bytes:
        raise OSError(
            "Not enough disk space to extract OpenEarthMap: "
            f"need at least {_format_bytes(required_bytes)}, "
            f"but only {_format_bytes(available_bytes)} is available after cleanup. "
            "Free more space or use a different data directory."
        )


def _archive_uncompressed_size(archive: zipfile.ZipFile) -> int:
    return sum(member.file_size for member in archive.infolist())


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _extraction_complete(extracted_dir: Path, archive: zipfile.ZipFile) -> bool:
    marker_path = extracted_dir / EXTRACT_COMPLETE_MARKER
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload != _archive_marker_payload(archive):
        return False
    return all(
        (extracted_dir / member.filename).exists()
        for member in archive.infolist()
        if not member.is_dir()
    )


def _write_extraction_marker(extracted_dir: Path, archive: zipfile.ZipFile) -> None:
    marker_path = extracted_dir / EXTRACT_COMPLETE_MARKER
    marker_path.write_text(
        json.dumps(_archive_marker_payload(archive), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _archive_marker_payload(archive: zipfile.ZipFile) -> dict[str, object]:
    return {
        "archive": Path(archive.filename).name,
        "archive_size": Path(archive.filename).stat().st_size,
        "member_count": len(archive.infolist()),
    }


def _safe_extract(
    archive: zipfile.ZipFile,
    extracted_dir: Path,
    *,
    progress: ProgressReporter | None,
) -> None:
    destination = extracted_dir.resolve()
    members = archive.infolist()
    _report(progress, f"Extracting {len(members)} archive members to {destination}")
    for member in members:
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")

    for index, member in enumerate(members, start=1):
        archive.extract(member, destination)
        if _should_report(index, len(members)):
            _report(
                progress,
                f"Extracted {index}/{len(members)} members ({_percent(index, len(members))}%)",
            )


def _is_image_dir(path: Path) -> bool:
    return path.is_dir() and path.name.lower() in {"image", "images", "img", "imgs"}


def _matching_label_dir(image_dir: Path) -> Path | None:
    for name in ("labels", "label", "masks", "mask"):
        candidate = image_dir.parent / name
        if candidate.is_dir():
            return candidate
    return None


def _iter_files(root: Path, suffixes: set[str]) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def _matching_mask(image_path: Path, image_dir: Path, label_dir: Path) -> Path | None:
    relative_stem = image_path.relative_to(image_dir).with_suffix("")
    for suffix in MASK_SUFFIXES:
        candidate = label_dir / relative_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _report_scan_progress(
    progress: ProgressReporter | None,
    *,
    current: int,
    total: int,
    scanned_images: int,
    pairs: int,
) -> None:
    _report(
        progress,
        "Scanned "
        f"{current}/{total} image directories "
        f"({_percent(current, total)}%): "
        f"{scanned_images} images, {pairs} pairs",
    )


def _content_length(response) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def _percent(current: int, total: int) -> int:
    if total <= 0:
        return 100
    return current * 100 // total


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{value} B"
        size /= 1024
    return f"{value} B"


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "test" in parts:
        return "test"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return "train"


@dataclass(frozen=True)
class _OpenEarthMapPaths:
    data_dir: Path
    archive_path: Path
    extracted_dir: Path
    prepared_dir: Path
    manifest_path: Path


def _paths(data_dir: str | Path) -> _OpenEarthMapPaths:
    root = Path(data_dir)
    return _OpenEarthMapPaths(
        data_dir=root,
        archive_path=root / "raw" / "OpenEarthMap.zip",
        extracted_dir=root / "extracted",
        prepared_dir=root / "prepared",
        manifest_path=root / "manifest.jsonl",
    )


def _build_click_command():
    import click

    @click.command("openearthmap", help="Download and prepare the OpenEarthMap manifest.")
    @click.option("--data-dir", type=click.Path(path_type=Path), default=Path("data/openearthmap"))
    @click.option("--skip-download", is_flag=True, help="Use an existing raw/OpenEarthMap.zip.")
    @click.option("--skip-extract", is_flag=True, help="Use an existing extracted directory.")
    @click.option("--force-download", is_flag=True, help="Replace the existing archive.")
    @click.option("--force-extract", is_flag=True, help="Replace the extracted directory.")
    @click.option("--force-prepared", is_flag=True, help="Rewrite prepared sample data files.")
    def command(
        data_dir: Path,
        skip_download: bool,
        skip_extract: bool,
        force_download: bool,
        force_extract: bool,
        force_prepared: bool,
    ) -> None:
        try:
            result = download_and_prepare_openearthmap(
                data_dir,
                download=not skip_download,
                extract=not skip_extract,
                force_download=force_download,
                force_extract=force_extract,
                force_prepared=force_prepared,
                progress=click.echo,
            )
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"manifest={result.manifest_path}")
        click.echo(f"prepared={result.prepared_dir}")
        click.echo(f"samples={result.samples}")
        click.echo(f"prepared_samples={result.prepared_samples}")
        click.echo(f"scanned_images={result.scanned_images}")

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
