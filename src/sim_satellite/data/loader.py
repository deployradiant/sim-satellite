"""Small manifest loader for prepared satellite image/mask pairs."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SatelliteSample:
    image_path: Path
    mask_path: Path
    prepared_path: Path
    split: str
    source: str = "openearthmap"

    @classmethod
    def from_json(cls, value: dict[str, str], root: Path) -> SatelliteSample:
        if not value.get("prepared"):
            raise ValueError("manifest sample is missing required prepared path")
        return cls(
            image_path=_resolve(root, value["image"]),
            mask_path=_resolve(root, value["mask"]),
            prepared_path=_resolve(root, value["prepared"]),
            split=value["split"],
            source=value.get("source", "openearthmap"),
        )

    def to_json(self, root: Path) -> dict[str, str]:
        value = {
            "image": _relative_or_absolute(self.image_path, root),
            "mask": _relative_or_absolute(self.mask_path, root),
            "split": self.split,
            "source": self.source,
        }
        value["prepared"] = _relative_or_absolute(self.prepared_path, root)
        return value


class SatelliteDataset(Sequence[SatelliteSample]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        samples = read_manifest(manifest_path)
        if split is not None:
            samples = [sample for sample in samples if sample.split == split]

        samples = sorted(samples, key=lambda sample: str(sample.image_path))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(samples)

        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | slice) -> SatelliteSample | tuple[SatelliteSample, ...]:
        return self.samples[index]


def read_manifest(manifest_path: str | Path) -> list[SatelliteSample]:
    path = Path(manifest_path)
    root = path.parent
    samples: list[SatelliteSample] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            samples.append(SatelliteSample.from_json(value, root))

    return samples


def write_manifest(samples: Sequence[SatelliteSample], manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent

    with path.open("w", encoding="utf-8") as handle:
        for sample in sorted(samples, key=lambda item: str(item.image_path)):
            json.dump(sample.to_json(root), handle, sort_keys=True)
            handle.write("\n")

    return path


def iter_batches(
    samples: Sequence[SatelliteSample],
    batch_size: int,
) -> Iterator[tuple[SatelliteSample, ...]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(samples), batch_size):
        yield tuple(samples[start : start + batch_size])


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
