"""Local data cache and deterministic sample loading for satellite imagery."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path("data")
DEFAULT_MANIFEST_NAME = "samples.jsonl"


def cache_root(cache_dir: str | Path | None = None) -> Path:
    """Return the configured local cache root.

    `cache_dir` wins, then SIM_SATELLITE_DATA_DIR, then ./data.
    """

    return Path(cache_dir or os.environ.get("SIM_SATELLITE_DATA_DIR", DEFAULT_CACHE_DIR))


@dataclass(frozen=True)
class SatelliteSample:
    """Normalized path-level sample record.

    Image decoding is intentionally left to downstream training code. If that
    code needs to read images locally, Pillow is the likely lightweight choice.
    """

    image_path: Path
    source: str
    split: str
    mask_path: Path | None = None
    classes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], root: str | Path | None = None) -> "SatelliteSample":
        """Normalize a manifest record into a sample."""

        image_path = _required_path(record, "image_path", root)
        mask_value = record.get("mask_path")
        mask_path = _resolve_path(mask_value, root) if mask_value else None

        source = _required_text(record, "source")
        split = _required_text(record, "split")
        classes = tuple(str(item) for item in record.get("classes", ()))
        metadata = dict(record.get("metadata") or {})

        return cls(
            image_path=image_path,
            mask_path=mask_path,
            source=source,
            split=split,
            classes=classes,
            metadata=metadata,
        )

    def to_record(self, root: str | Path | None = None) -> dict[str, Any]:
        """Convert a sample into a portable JSON manifest record."""

        return {
            "image_path": _portable_path(self.image_path, root),
            "mask_path": _portable_path(self.mask_path, root) if self.mask_path else None,
            "source": self.source,
            "split": self.split,
            "classes": list(self.classes),
            "metadata": dict(self.metadata),
        }


class SatelliteDataCache:
    """Small helper for the local data cache layout."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = cache_root(root)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def prepared_dir(self) -> Path:
        return self.root / "prepared"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    def manifest_path(self, name: str = DEFAULT_MANIFEST_NAME) -> Path:
        return self.manifests_dir / name

    def ensure(self) -> "SatelliteDataCache":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        return self


class SatelliteDataset(Sequence[SatelliteSample]):
    """Deterministic sequence of normalized satellite samples."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        split: str | None = None,
        source: str | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        cache = SatelliteDataCache(cache_dir)
        path = Path(manifest_path) if manifest_path else cache.manifest_path()
        samples = read_manifest(path, root=cache.root)

        if split is not None:
            samples = [sample for sample in samples if sample.split == split]
        if source is not None:
            samples = [sample for sample in samples if sample.source == source]

        samples = sorted(samples, key=_sample_key)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | slice) -> SatelliteSample | tuple[SatelliteSample, ...]:
        return self.samples[index]

    def __iter__(self) -> Iterator[SatelliteSample]:
        return iter(self.samples)


class SatelliteDataLoader:
    """Simple deterministic batch iterator over a dataset or sample iterable."""

    def __init__(self, samples: Iterable[SatelliteSample], batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.samples = tuple(samples)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[SatelliteSample, ...]]:
        for start in range(0, len(self.samples), self.batch_size):
            yield self.samples[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size


def read_manifest(path: str | Path, root: str | Path | None = None) -> list[SatelliteSample]:
    """Read a JSONL sample manifest."""

    manifest_path = Path(path)
    samples: list[SatelliteSample] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from exc
            samples.append(SatelliteSample.from_record(record, root=root))

    return samples


def write_manifest(
    samples: Iterable[SatelliteSample],
    path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """Write samples as JSONL in deterministic order."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_samples = sorted(samples, key=_sample_key)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in ordered_samples:
            json.dump(sample.to_record(root=root), handle, sort_keys=True)
            handle.write("\n")

    return manifest_path


def load_dataset(
    manifest_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    split: str | None = None,
    source: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> SatelliteDataset:
    """Convenience wrapper for creating a `SatelliteDataset`."""

    return SatelliteDataset(
        manifest_path,
        cache_dir=cache_dir,
        split=split,
        source=source,
        shuffle=shuffle,
        seed=seed,
    )


def _sample_key(sample: SatelliteSample) -> tuple[str, str, str, str]:
    mask = str(sample.mask_path) if sample.mask_path else ""
    return (sample.split, sample.source, str(sample.image_path), mask)


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"manifest record is missing required field: {key}")
    return str(value)


def _required_path(record: Mapping[str, Any], key: str, root: str | Path | None) -> Path:
    value = record.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"manifest record is missing required field: {key}")
    return _resolve_path(value, root)


def _resolve_path(value: Any, root: str | Path | None) -> Path:
    path = Path(str(value))
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def _portable_path(path: Path | None, root: str | Path | None) -> str | None:
    if path is None:
        return None

    resolved_path = Path(path)
    if root is None:
        return str(resolved_path)

    try:
        return str(resolved_path.relative_to(Path(root)))
    except ValueError:
        return str(resolved_path)
