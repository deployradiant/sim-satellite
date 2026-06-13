"""Download and prepare OpenEarthMap for the first training runs."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from sim_satellite.data.loader import SatelliteSample, write_manifest

OPENEARTHMAP_URL = "https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".tif", ".tiff"}
OPENEARTHMAP_CLASSES = (
    "unknown",
    "bareland",
    "grass",
    "pavement",
    "road",
    "tree",
    "water",
    "cropland",
    "building",
)
OPENEARTHMAP_CLASS_IDS = {name: index for index, name in enumerate(OPENEARTHMAP_CLASSES)}
OPENEARTHMAP_RGB = {
    "unknown": (0, 0, 0),
    "bareland": (128, 0, 0),
    "grass": (0, 255, 36),
    "pavement": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "cropland": (75, 181, 73),
    "building": (222, 31, 7),
}


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
) -> OpenEarthMapResult:
    paths = _paths(data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    if download:
        _download(paths.archive_path, force=force_download)
    elif not paths.archive_path.exists() and extract:
        raise FileNotFoundError(f"OpenEarthMap archive not found: {paths.archive_path}")

    if extract:
        _extract(paths.archive_path, paths.extracted_dir, force=force_extract)
    elif not paths.extracted_dir.exists():
        raise FileNotFoundError(
            f"OpenEarthMap extracted directory not found: {paths.extracted_dir}"
        )

    scan = scan_openearthmap(paths.extracted_dir)
    if scan.image_dirs_without_labels:
        raise RuntimeError(
            "OpenEarthMap scan found image directories without matching labels: "
            f"{scan.image_dirs_without_labels}"
        )
    if scan.missing_masks:
        raise RuntimeError(f"OpenEarthMap scan found images without masks: {scan.missing_masks}")
    if not scan.samples:
        raise RuntimeError(f"No OpenEarthMap image/mask pairs found under {paths.extracted_dir}")

    write_manifest(scan.samples, paths.manifest_path)
    write_classes(paths.classes_path)

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


def scan_openearthmap(extracted_dir: str | Path) -> OpenEarthMapScan:
    root = Path(extracted_dir)
    pairs: list[SatelliteSample] = []
    scanned_images = 0
    missing_masks = 0
    image_dirs_without_labels = 0

    for image_dir in sorted(path for path in root.rglob("*") if _is_image_dir(path)):
        label_dir = _matching_label_dir(image_dir)
        if label_dir is None:
            image_dirs_without_labels += 1
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


def _download(archive_path: Path, *, force: bool) -> None:
    if archive_path.exists() and not force:
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")

    with urllib.request.urlopen(OPENEARTHMAP_URL) as response:
        with partial_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial_path.replace(archive_path)


def _extract(archive_path: Path, extracted_dir: Path, *, force: bool) -> None:
    if extracted_dir.exists() and any(extracted_dir.iterdir()) and not force:
        return
    if force and extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, extracted_dir)


def _safe_extract(archive: zipfile.ZipFile, extracted_dir: Path) -> None:
    destination = extracted_dir.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
    archive.extractall(destination)


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
        )
        click.echo(f"manifest={result.manifest_path}")
        click.echo(f"classes={result.classes_path}")
        click.echo(f"samples={result.samples}")
        click.echo(f"scanned_images={result.scanned_images}")

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
