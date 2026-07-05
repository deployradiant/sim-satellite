"""Deterministic benchmark sample selection."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim_satellite.data.loader import SatelliteSample, read_manifest
from sim_satellite.research.spec import BenchmarkProtocol, sample_identity_digest


@dataclass(frozen=True)
class BenchmarkSelection:
    eval_samples: tuple[SatelliteSample, ...]
    eval_groups: tuple[str, ...]
    validation_holdout_patterns: str
    sample_digest: str
    manifest_path: str
    manifest_samples: int
    manifest_samples_by_source: dict[str, int]
    manifest_samples_by_split: dict[str, int]
    eval_sample_identities: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_samples": self.manifest_samples,
            "manifest_samples_by_source": self.manifest_samples_by_source,
            "manifest_samples_by_split": self.manifest_samples_by_split,
            "eval_samples": len(self.eval_samples),
            "eval_groups": list(self.eval_groups),
            "validation_holdout_patterns": self.validation_holdout_patterns,
            "sample_digest": self.sample_digest,
            "eval_sample_identities": list(self.eval_sample_identities),
        }


def select_benchmark_samples(
    protocol: BenchmarkProtocol,
    *,
    manifest_path: str = "",
) -> BenchmarkSelection:
    path = Path(manifest_path or protocol.manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark manifest not found: {path}. Run `sim-satellite openearthmap` first."
        )
    all_samples = read_manifest(path)
    if not all_samples:
        raise RuntimeError(f"Benchmark manifest has no samples: {path}")
    split_samples = [sample for sample in all_samples if sample.split == protocol.split]
    if not split_samples:
        raise RuntimeError(f"No samples for split `{protocol.split}` in {path}")

    if protocol.eval_sample_identities:
        eval_samples = _pinned_eval_samples(split_samples, protocol.eval_sample_identities)
    else:
        eval_samples = _seeded_eval_samples(split_samples, protocol)

    eval_groups = tuple(sorted({sample_group(sample) for sample in eval_samples}))
    grouped = {sample_group(sample) for sample in split_samples}
    if len(eval_groups) >= len(grouped):
        raise RuntimeError("Benchmark holdout would consume every sample group")

    identities = tuple(sample_identity(sample) for sample in eval_samples)
    holdout_patterns = sorted({group.split(":", 1)[-1] for group in eval_groups})
    by_source: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for sample in all_samples:
        by_source[sample.source] = by_source.get(sample.source, 0) + 1
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
    return BenchmarkSelection(
        eval_samples=eval_samples,
        eval_groups=eval_groups,
        validation_holdout_patterns=",".join(holdout_patterns),
        sample_digest=sample_identity_digest(identities),
        manifest_path=str(path),
        manifest_samples=len(all_samples),
        manifest_samples_by_source=dict(sorted(by_source.items())),
        manifest_samples_by_split=dict(sorted(by_split.items())),
        eval_sample_identities=identities,
    )


def sample_identity(sample: SatelliteSample) -> str:
    tail = "/".join(sample.image_path.parts[-3:])
    return f"{sample.source}:{sample.split}:{tail}"


def sample_group(sample: SatelliteSample) -> str:
    parts = sample.image_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"image", "images", "img", "imgs"}:
            tail = parts[index + 1 :]
            if len(tail) > 1:
                return f"{sample.source}:{tail[0]}"
            if index > 0:
                return f"{sample.source}:{parts[index - 1]}"
            if tail:
                return f"{sample.source}:{_stem_group(Path(tail[0]).stem)}"
    parent = sample.image_path.parent.name
    if parent:
        return f"{sample.source}:{parent}"
    return f"{sample.source}:{_stem_group(sample.image_path.stem)}"


def _seeded_eval_samples(
    split_samples: list[SatelliteSample],
    protocol: BenchmarkProtocol,
) -> tuple[SatelliteSample, ...]:
    grouped: dict[str, list[SatelliteSample]] = {}
    for sample in split_samples:
        grouped.setdefault(sample_group(sample), []).append(sample)
    if len(grouped) < 2:
        raise RuntimeError("Benchmark holdout requires at least two sample groups")

    ranked = sorted(grouped, key=lambda group: _group_rank(protocol.seed, group))
    pool: list[SatelliteSample] = []
    eval_groups: list[str] = []
    for group in ranked:
        eval_groups.append(group)
        pool.extend(grouped[group])
        if len(pool) >= protocol.samples:
            break
    if len(pool) < protocol.samples:
        raise RuntimeError(
            f"Benchmark holdout found only {len(pool)} samples for "
            f"{protocol.samples} requested eval samples"
        )
    if len(eval_groups) >= len(grouped):
        raise RuntimeError("Benchmark holdout would consume every sample group")

    pool.sort(key=sample_identity)
    random.Random(protocol.seed).shuffle(pool)
    return tuple(pool[: protocol.samples])


def _pinned_eval_samples(
    split_samples: list[SatelliteSample],
    identities: tuple[str, ...],
) -> tuple[SatelliteSample, ...]:
    by_identity: dict[str, SatelliteSample] = {}
    duplicates: set[str] = set()
    for sample in split_samples:
        identity = sample_identity(sample)
        if identity in by_identity:
            duplicates.add(identity)
        by_identity[identity] = sample
    if duplicates:
        raise RuntimeError(
            "Benchmark manifest has duplicate pinned sample identities: "
            f"{', '.join(sorted(duplicates))}"
        )

    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise RuntimeError(
            "Benchmark manifest is missing pinned eval samples. "
            f"Expected identities: {preview}{suffix}"
        )
    return tuple(by_identity[identity] for identity in identities)


def _group_rank(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def _stem_group(stem: str) -> str:
    for separator in ("_", "-"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem
