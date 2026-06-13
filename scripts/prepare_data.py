"""Prepare a JSONL sample manifest from image and mask directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from sim_satellite.data import (
        DEFAULT_MANIFEST_NAME,
        SatelliteDataCache,
        SatelliteSample,
        write_manifest,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sim_satellite.data import (
        DEFAULT_MANIFEST_NAME,
        SatelliteDataCache,
        SatelliteSample,
        write_manifest,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def build_samples(
    image_dir: str | Path,
    *,
    source: str,
    split: str,
    mask_dir: str | Path | None = None,
    classes: tuple[str, ...] = (),
) -> list[SatelliteSample]:
    """Build normalized samples from a directory of images."""

    images = _discover_images(Path(image_dir))
    masks = _discover_masks(Path(mask_dir)) if mask_dir else {}

    return [
        SatelliteSample(
            image_path=image,
            mask_path=masks.get(image.stem),
            source=source,
            split=split,
            classes=classes,
            metadata={"filename": image.name},
        )
        for image in images
    ]


def prepare_manifest(
    image_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    source: str,
    split: str,
    mask_dir: str | Path | None = None,
    classes: tuple[str, ...] = (),
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> Path:
    """Write a deterministic manifest for an image directory."""

    cache = SatelliteDataCache(cache_dir).ensure()
    samples = build_samples(
        image_dir,
        source=source,
        split=split,
        mask_dir=mask_dir,
        classes=classes,
    )
    return write_manifest(samples, cache.manifest_path(manifest_name), root=cache.root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a normalized satellite sample manifest.")
    parser.add_argument("image_dir", help="Directory containing source images.")
    parser.add_argument("--cache-dir", default=None, help="Cache root. Defaults to ./data.")
    parser.add_argument("--source", required=True, help="Dataset/source name.")
    parser.add_argument("--split", default="train", help="Split assigned to every discovered sample.")
    parser.add_argument("--mask-dir", default=None, help="Optional directory containing masks by matching stem.")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Class label present in masks.")
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME, help="Output manifest filename.")
    args = parser.parse_args(argv)

    manifest_path = prepare_manifest(
        args.image_dir,
        cache_dir=args.cache_dir,
        source=args.source,
        split=args.split,
        mask_dir=args.mask_dir,
        classes=tuple(args.classes),
        manifest_name=args.manifest_name,
    )
    print(manifest_path)
    return 0


def _discover_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _discover_masks(mask_dir: Path) -> dict[str, Path]:
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask directory does not exist: {mask_dir}")

    masks: dict[str, Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            masks[path.stem] = path
    return masks


if __name__ == "__main__":
    raise SystemExit(main())
