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
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class OpenEarthMapResult:
    data_dir: Path
    archive_path: Path
    extracted_dir: Path
    manifest_path: Path
    classes_path: Path
    samples: int
    scanned_images: int
    missing_masks: int
    image_dirs_without_labels: int


@dataclass(frozen=True)
class OpenEarthMapScan:
    samples: tuple[SatelliteSample, ...]
    scanned_images: int
    missing_masks: int
    image_dirs_without_labels: int


def download_and_prepare_openearthmap(
    data_dir: str | Path = "data/openearthmap",
    *,
    download: bool = True,
    extract: bool = True,
    force_download: bool = False,
    force_extract: bool = False,
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

    _report(progress, f"Writing manifest to {paths.manifest_path}")
    write_manifest(scan.samples, paths.manifest_path)
    _report(progress, f"Writing class metadata to {paths.classes_path}")
    write_classes(paths.classes_path)
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
        manifest_path=paths.manifest_path,
        classes_path=paths.classes_path,
        samples=len(scan.samples),
        scanned_images=scan.scanned_images,
        missing_masks=scan.missing_masks,
        image_dirs_without_labels=scan.image_dirs_without_labels,
    )


def find_openearthmap_samples(extracted_dir: str | Path) -> list[SatelliteSample]:
    return list(scan_openearthmap(extracted_dir).samples)


def scan_openearthmap(
    extracted_dir: str | Path,
    *,
    progress: ProgressReporter | None = None,
) -> OpenEarthMapScan:
    root = Path(extracted_dir)
    pairs: list[SatelliteSample] = []
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
                SatelliteSample(
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


def write_classes(classes_path: str | Path) -> Path:
    path = Path(classes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "openearthmap",
        "classes": [
            {
                "id": OPENEARTHMAP_CLASS_IDS[name],
                "name": name,
                "rgb": list(OPENEARTHMAP_RGB[name]),
            }
            for name in OPENEARTHMAP_CLASSES
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


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
    if extracted_dir.exists() and any(extracted_dir.iterdir()) and not force:
        _report(progress, f"Using cached extracted data {extracted_dir}")
        return
    if force and extracted_dir.exists():
        _report(progress, f"Removing existing extracted data {extracted_dir}")
        shutil.rmtree(extracted_dir)

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, extracted_dir, progress=progress)


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
    manifest_path: Path
    classes_path: Path


def _paths(data_dir: str | Path) -> _OpenEarthMapPaths:
    root = Path(data_dir)
    return _OpenEarthMapPaths(
        data_dir=root,
        archive_path=root / "raw" / "OpenEarthMap.zip",
        extracted_dir=root / "extracted",
        manifest_path=root / "manifest.jsonl",
        classes_path=root / "classes.json",
    )


def _build_click_command():
    import click

    @click.command("openearthmap", help="Download and prepare the OpenEarthMap manifest.")
    @click.option("--data-dir", type=click.Path(path_type=Path), default=Path("data/openearthmap"))
    @click.option("--skip-download", is_flag=True, help="Use an existing raw/OpenEarthMap.zip.")
    @click.option("--skip-extract", is_flag=True, help="Use an existing extracted directory.")
    @click.option("--force-download", is_flag=True, help="Replace the existing archive.")
    @click.option("--force-extract", is_flag=True, help="Replace the extracted directory.")
    def command(
        data_dir: Path,
        skip_download: bool,
        skip_extract: bool,
        force_download: bool,
        force_extract: bool,
    ) -> None:
        result = download_and_prepare_openearthmap(
            data_dir,
            download=not skip_download,
            extract=not skip_extract,
            force_download=force_download,
            force_extract=force_extract,
            progress=click.echo,
        )
        click.echo(f"manifest={result.manifest_path}")
        click.echo(f"classes={result.classes_path}")
        click.echo(f"samples={result.samples}")
        click.echo(f"scanned_images={result.scanned_images}")

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
