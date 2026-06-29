"""Balanced OpenEarthMap research loops for VAE experiments."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import click
import torch

from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH
from sim_satellite.vae.training import VAETrainingConfig, run_vae_training

ProgressReporter = Callable[[str], None]
DEFAULT_VAE_RESEARCH_ARTIFACT_DIR = "artifacts/vae-research"
DEFAULT_VAE_RESEARCH_MANIFEST_DIR = "data/openearthmap/research"
CLASS_INDEX_VERSION = 2
SUPPORTED_CLASS_INDEX_VERSIONS = {1, CLASS_INDEX_VERSION}
LOCATION_BUCKETS_PER_GROUP = 4


@dataclass(frozen=True)
class VAEResearchManifestConfig:
    base_manifest_path: str = DEFAULT_MANIFEST_PATH
    class_index_path: str = ""
    output_manifest_path: str = (
        f"{DEFAULT_VAE_RESEARCH_MANIFEST_DIR}/vae-balanced-openearthmap.jsonl"
    )
    report_path: str = f"{DEFAULT_VAE_RESEARCH_ARTIFACT_DIR}/vae-balanced-openearthmap.json"
    split: str = "train"
    output_split: str = "train"
    samples: int = 512
    seed: int = 20260629
    min_class_pixels: int = 64
    max_group_fraction: float = 0.15
    max_location_bucket_fraction: float = 0.15
    include_unlabeled: bool = False

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VAEResearchManifestConfig:
        return _dataclass_from_dict(cls, values)


@dataclass(frozen=True)
class VAEResearchManifestResult:
    base_manifest_path: str
    output_manifest_path: str
    report_path: str
    split: str
    output_split: str
    seed: int
    candidate_samples: int
    selected_samples: int
    selected_groups: int
    class_hits: dict[str, int]
    class_pixel_fraction: dict[str, float]
    group_counts: dict[str, int]
    location_bucket_counts: dict[str, int]
    missing_classes: list[str]


@dataclass(frozen=True)
class VAEResearchSweepConfig:
    name: str = "balanced-vae"
    base_manifest_path: str = DEFAULT_MANIFEST_PATH
    manifest_dir: str = DEFAULT_VAE_RESEARCH_MANIFEST_DIR
    artifact_dir: str = DEFAULT_VAE_RESEARCH_ARTIFACT_DIR
    class_index_path: str = ""
    split: str = "train"
    output_split: str = "train"
    trials: int = 4
    samples: int = 512
    dataset_seed: int = 20260629
    training_seed: int = 90210
    min_class_pixels: int = 64
    max_group_fraction: float = 0.15
    max_location_bucket_fraction: float = 0.15
    include_unlabeled: bool = False
    train_fraction: float = 0.82
    validation_split: str = "city"
    validation_samples: int = 80
    image_size: int = 256
    eval_size: int = 256
    batch_size: int = 4
    steps: int = 1500
    learning_rate: float = 1.5e-4
    latent_channels: int = 24
    hidden_channels: int = 32
    downsample_factor: int = 4
    kl_weight: float = 1e-6
    mse_weight: float = 0.2
    reconstruction_exponential_weight: float = 0.5
    reconstruction_exponential_scale: float = 12.0
    edge_weight: float = 0.0
    class_balanced_l1_weight: float = 0.0
    class_balanced_edge_weight: float = 0.0
    include_unlabeled_in_class_loss: bool = False
    class_balanced_crop_probability: float = 0.0
    include_unlabeled_in_class_crop: bool = False
    cache_images: int = 256
    log_every: int = 250
    grid_samples: int = 8

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VAEResearchSweepConfig:
        return _dataclass_from_dict(cls, values)


@dataclass(frozen=True)
class VAEResearchSweepResult:
    name: str
    trials: int
    best_trial_index: int
    best_metric: str
    trial_results: list[dict[str, Any]]


@dataclass(frozen=True)
class _SampleClassInfo:
    sample: SatelliteSample
    group: str
    location_bucket: str
    class_counts: tuple[int, ...]
    total_pixels: int


def build_vae_research_manifest(
    config: VAEResearchManifestConfig,
    *,
    progress: ProgressReporter | None = None,
) -> VAEResearchManifestResult:
    _validate_manifest_config(config)
    base_manifest_path = Path(config.base_manifest_path)
    if config.class_index_path:
        infos = _load_or_build_sample_index(
            base_manifest_path,
            split=config.split,
            class_index_path=Path(config.class_index_path),
            progress=progress,
        )
    else:
        candidates = _samples_for_split(base_manifest_path, config.split)
        if not candidates:
            raise RuntimeError(f"No samples for split `{config.split}` in {base_manifest_path}")

        _report(progress, f"Analyzing {len(candidates)} prepared masks for class balance")
        infos = _analyze_samples(
            candidates,
            progress=progress,
        )
    _require_class_coverage(
        infos,
        min_class_pixels=config.min_class_pixels,
        include_unlabeled=config.include_unlabeled,
    )
    return _build_vae_research_manifest_from_infos(config, infos, progress=progress)


def load_vae_research_manifest_config_file(config_path: Path) -> VAEResearchManifestConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("VAE research manifest config JSON must be an object")
    return load_vae_research_manifest_config_values(values)


def load_vae_research_manifest_config_values(
    values: Mapping[str, Any],
) -> VAEResearchManifestConfig:
    config = VAEResearchManifestConfig.from_dict(values)
    _validate_manifest_config(config)
    return config


def _build_vae_research_manifest_from_infos(
    config: VAEResearchManifestConfig,
    infos: Sequence[_SampleClassInfo],
    *,
    progress: ProgressReporter | None = None,
) -> VAEResearchManifestResult:
    selected = _select_balanced_samples(infos, config)
    if not selected:
        raise RuntimeError("Balanced VAE manifest selection produced no samples")

    selected_samples = tuple(replace(info.sample, split=config.output_split) for info in selected)
    output_manifest_path = Path(config.output_manifest_path)
    write_manifest(selected_samples, output_manifest_path)
    result = _manifest_result(
        config,
        candidate_samples=len(infos),
        selected=selected,
    )
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "result": asdict(result),
                "group_source": "source plus region inferred from image path",
                "location_bucket_source": (
                    "region plus filename tile id modulo "
                    f"{LOCATION_BUCKETS_PER_GROUP}; OpenEarthMap manifests do not "
                    "include true latitude/longitude coordinates"
                ),
                "classes": list(OPENEARTHMAP_CLASSES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(
        progress,
        f"Wrote balanced VAE manifest with {result.selected_samples} samples "
        f"across {result.selected_groups} groups and "
        f"{len(result.location_bucket_counts)} location buckets",
    )
    return result


def run_vae_research_sweep(
    config: VAEResearchSweepConfig,
    *,
    progress: ProgressReporter | None = None,
    after_index: Callable[[], None] | None = None,
    after_trial: Callable[[], None] | None = None,
) -> VAEResearchSweepResult:
    _validate_sweep_config(config)
    trial_results: list[dict[str, Any]] = []
    class_index_path = _sweep_class_index_path(config)
    infos = _load_or_build_sample_index(
        Path(config.base_manifest_path),
        split=config.split,
        class_index_path=class_index_path,
        progress=progress,
    )
    _require_class_coverage(
        infos,
        min_class_pixels=config.min_class_pixels,
        include_unlabeled=config.include_unlabeled,
    )
    if after_index is not None:
        after_index()

    for trial_index in range(config.trials):
        dataset_seed = config.dataset_seed + trial_index * 1009
        training_seed = config.training_seed + trial_index * 9176
        stem = _trial_stem(config.name, trial_index, dataset_seed, training_seed)
        manifest_config = VAEResearchManifestConfig(
            base_manifest_path=config.base_manifest_path,
            output_manifest_path=str(Path(config.manifest_dir) / f"{stem}.jsonl"),
            report_path=str(Path(config.artifact_dir) / f"{stem}-manifest.json"),
            split=config.split,
            output_split=config.output_split,
            samples=config.samples,
            seed=dataset_seed,
            min_class_pixels=config.min_class_pixels,
            max_group_fraction=config.max_group_fraction,
            max_location_bucket_fraction=config.max_location_bucket_fraction,
            include_unlabeled=config.include_unlabeled,
        )
        _report(
            progress,
            f"Starting VAE research trial {trial_index + 1}/{config.trials}: {stem}",
        )
        manifest_result = _build_vae_research_manifest_from_infos(
            manifest_config,
            infos,
            progress=progress,
        )
        training_config = _trial_training_config(
            config,
            manifest_path=manifest_config.output_manifest_path,
            training_seed=training_seed,
            stem=stem,
        )
        training_result = run_vae_training(training_config)
        trial_payload = {
            "trial_index": trial_index,
            "dataset_seed": dataset_seed,
            "training_seed": training_seed,
            "manifest": asdict(manifest_result),
            "training": asdict(training_result),
        }
        trial_results.append(trial_payload)
        if after_trial is not None:
            after_trial()

    best_trial_index = _best_trial_index(trial_results)
    sweep_result = VAEResearchSweepResult(
        name=config.name,
        trials=config.trials,
        best_trial_index=best_trial_index,
        best_metric="custom_vae.ssim",
        trial_results=trial_results,
    )
    summary_path = Path(config.artifact_dir) / f"{_safe_name(config.name)}-sweep-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "result": asdict(sweep_result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sweep_result


def load_vae_research_sweep_config_file(config_path: Path) -> VAEResearchSweepConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("VAE research sweep config JSON must be an object")
    return load_vae_research_sweep_config_values(values)


def load_vae_research_sweep_config_values(
    values: Mapping[str, Any],
) -> VAEResearchSweepConfig:
    config = VAEResearchSweepConfig.from_dict(values)
    _validate_sweep_config(config)
    return config


def _trial_training_config(
    config: VAEResearchSweepConfig,
    *,
    manifest_path: str,
    training_seed: int,
    stem: str,
) -> VAETrainingConfig:
    artifact_dir = Path(config.artifact_dir)
    return VAETrainingConfig(
        seed=training_seed,
        manifest_path=manifest_path,
        split=config.output_split,
        train_fraction=config.train_fraction,
        validation_split=config.validation_split,
        validation_samples=config.validation_samples,
        image_size=config.image_size,
        eval_size=config.eval_size,
        batch_size=config.batch_size,
        steps=config.steps,
        learning_rate=config.learning_rate,
        latent_channels=config.latent_channels,
        hidden_channels=config.hidden_channels,
        downsample_factor=config.downsample_factor,
        kl_weight=config.kl_weight,
        mse_weight=config.mse_weight,
        reconstruction_exponential_weight=config.reconstruction_exponential_weight,
        reconstruction_exponential_scale=config.reconstruction_exponential_scale,
        edge_weight=config.edge_weight,
        class_balanced_l1_weight=config.class_balanced_l1_weight,
        class_balanced_edge_weight=config.class_balanced_edge_weight,
        include_unlabeled_in_class_loss=config.include_unlabeled_in_class_loss,
        class_balanced_crop_probability=config.class_balanced_crop_probability,
        include_unlabeled_in_class_crop=config.include_unlabeled_in_class_crop,
        device="cuda",
        cache_images=config.cache_images,
        log_every=config.log_every,
        grid_samples=config.grid_samples,
        checkpoint_path=str(artifact_dir / f"{stem}.pt"),
        metrics_path=str(artifact_dir / f"{stem}-metrics.json"),
        grid_path=str(artifact_dir / f"{stem}-grid.png"),
    )


def _load_or_build_sample_index(
    base_manifest_path: Path,
    *,
    split: str,
    class_index_path: Path,
    progress: ProgressReporter | None,
) -> tuple[_SampleClassInfo, ...]:
    cached = _load_sample_index(
        class_index_path,
        base_manifest_path=base_manifest_path,
        split=split,
    )
    if cached is not None:
        _report(progress, f"Loaded VAE class index with {len(cached)} samples")
        return cached

    candidates = _samples_for_split(base_manifest_path, split)
    if not candidates:
        raise RuntimeError(f"No samples for split `{split}` in {base_manifest_path}")
    _report(progress, f"Building VAE class index from {len(candidates)} prepared masks")
    infos = _analyze_samples(candidates, progress=progress)
    _write_sample_index(
        class_index_path,
        base_manifest_path=base_manifest_path,
        split=split,
        infos=infos,
    )
    _report(progress, f"Wrote VAE class index: {class_index_path}")
    return infos


def _load_sample_index(
    class_index_path: Path,
    *,
    base_manifest_path: Path,
    split: str,
) -> tuple[_SampleClassInfo, ...] | None:
    if not class_index_path.exists():
        return None
    try:
        payload = json.loads(class_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("version") not in SUPPORTED_CLASS_INDEX_VERSIONS:
        return None
    if payload.get("base_manifest_path") != str(base_manifest_path):
        return None
    if payload.get("split") != split:
        return None
    if payload.get("classes") != list(OPENEARTHMAP_CLASSES):
        return None
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None
    try:
        return tuple(_sample_info_from_index_record(record) for record in records)
    except (TypeError, ValueError):
        return None


def _write_sample_index(
    class_index_path: Path,
    *,
    base_manifest_path: Path,
    split: str,
    infos: Sequence[_SampleClassInfo],
) -> None:
    class_index_path.parent.mkdir(parents=True, exist_ok=True)
    class_index_path.write_text(
        json.dumps(
            {
                "version": CLASS_INDEX_VERSION,
                "base_manifest_path": str(base_manifest_path),
                "split": split,
                "classes": list(OPENEARTHMAP_CLASSES),
                "records": [_sample_info_index_record(info) for info in infos],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sample_info_index_record(info: _SampleClassInfo) -> dict[str, Any]:
    return {
        "image_path": str(info.sample.image_path),
        "mask_path": str(info.sample.mask_path),
        "prepared_path": str(info.sample.prepared_path),
        "split": info.sample.split,
        "source": info.sample.source,
        "group": info.group,
        "location_bucket": info.location_bucket,
        "class_counts": list(info.class_counts),
        "total_pixels": info.total_pixels,
    }


def _sample_info_from_index_record(record: object) -> _SampleClassInfo:
    if not isinstance(record, Mapping):
        raise TypeError("class index record must be a mapping")
    class_counts = record["class_counts"]
    if not isinstance(class_counts, list):
        raise TypeError("class_counts must be a list")
    sample = SatelliteSample(
        image_path=Path(str(record["image_path"])),
        mask_path=Path(str(record["mask_path"])),
        prepared_path=Path(str(record["prepared_path"])),
        split=str(record["split"]),
        source=str(record["source"]),
    )
    group = _sample_group(sample)
    return _SampleClassInfo(
        sample=sample,
        group=group,
        location_bucket=_sample_location_bucket(sample, group=group),
        class_counts=tuple(int(value) for value in class_counts),
        total_pixels=int(record["total_pixels"]),
    )


def _analyze_samples(
    samples: Sequence[SatelliteSample],
    *,
    progress: ProgressReporter | None,
) -> tuple[_SampleClassInfo, ...]:
    infos: list[_SampleClassInfo] = []
    for index, sample in enumerate(samples, start=1):
        prepared = load_prepared_sample(sample.prepared_path, torch=torch)
        counts = torch.bincount(
            prepared.mask.long().reshape(-1),
            minlength=len(OPENEARTHMAP_CLASSES),
        )[: len(OPENEARTHMAP_CLASSES)]
        group = _sample_group(sample)
        infos.append(
            _SampleClassInfo(
                sample=sample,
                group=group,
                location_bucket=_sample_location_bucket(sample, group=group),
                class_counts=tuple(int(value) for value in counts.tolist()),
                total_pixels=int(prepared.mask.numel()),
            )
        )
        if _should_report(index, len(samples)):
            _report(progress, f"Analyzed {index}/{len(samples)} prepared masks")

    return tuple(infos)


def _require_class_coverage(
    infos: Sequence[_SampleClassInfo],
    *,
    min_class_pixels: int,
    include_unlabeled: bool,
) -> None:
    active_counts = _class_hits(infos, min_class_pixels=min_class_pixels)
    missing = [
        OPENEARTHMAP_CLASSES[class_id]
        for class_id in _useful_class_ids(include_unlabeled=include_unlabeled)
        if active_counts[class_id] == 0
    ]
    if missing:
        raise RuntimeError(f"Prepared data has no usable samples for classes: {', '.join(missing)}")


def _select_balanced_samples(
    infos: Sequence[_SampleClassInfo],
    config: VAEResearchManifestConfig,
) -> tuple[_SampleClassInfo, ...]:
    target = min(config.samples, len(infos))
    useful_class_ids = _useful_class_ids(include_unlabeled=config.include_unlabeled)
    class_availability = _class_hits(infos, min_class_pixels=config.min_class_pixels)
    active_class_ids = tuple(
        class_id for class_id in useful_class_ids if class_availability[class_id] > 0
    )
    if not active_class_ids:
        raise RuntimeError("No usable land-cover classes are available for VAE balancing")

    class_quota = {
        class_id: min(
            class_availability[class_id],
            max(1, math.ceil(target / len(active_class_ids))),
        )
        for class_id in active_class_ids
    }
    groups = sorted({info.group for info in infos})
    group_quota = max(1, math.ceil(target / max(1, min(len(groups), target))))
    max_group_samples = max(1, math.ceil(target * config.max_group_fraction))
    location_buckets = sorted({info.location_bucket for info in infos})
    location_bucket_quota = max(
        1,
        math.ceil(target / max(1, min(len(location_buckets), target))),
    )
    max_location_bucket_samples = max(
        1,
        math.ceil(target * config.max_location_bucket_fraction),
    )
    rng = random.Random(config.seed)
    tie_breakers = {id(info): rng.random() for info in infos}

    selected: list[_SampleClassInfo] = []
    selected_ids: set[int] = set()
    selected_class_hits = [0 for _ in OPENEARTHMAP_CLASSES]
    selected_group_counts: dict[str, int] = {}
    selected_location_bucket_counts: dict[str, int] = {}

    while len(selected) < target:
        allow_balance_overflow = _remaining_without_balance_overflow(
            infos,
            selected_ids,
            selected_group_counts,
            max_group_samples,
            selected_location_bucket_counts,
            max_location_bucket_samples,
        ) < target - len(selected)
        best = max(
            (
                info
                for info in infos
                if id(info) not in selected_ids
                and (
                    allow_balance_overflow
                    or (
                        selected_group_counts.get(info.group, 0) < max_group_samples
                        and selected_location_bucket_counts.get(info.location_bucket, 0)
                        < max_location_bucket_samples
                    )
                )
            ),
            key=lambda info: _selection_score(
                info,
                selected_class_hits=selected_class_hits,
                selected_group_counts=selected_group_counts,
                selected_location_bucket_counts=selected_location_bucket_counts,
                class_quota=class_quota,
                group_quota=group_quota,
                location_bucket_quota=location_bucket_quota,
                min_class_pixels=config.min_class_pixels,
                class_availability=class_availability,
                tie_breaker=tie_breakers[id(info)],
            ),
        )
        selected.append(best)
        selected_ids.add(id(best))
        selected_group_counts[best.group] = selected_group_counts.get(best.group, 0) + 1
        selected_location_bucket_counts[best.location_bucket] = (
            selected_location_bucket_counts.get(best.location_bucket, 0) + 1
        )
        for class_id in active_class_ids:
            if best.class_counts[class_id] >= config.min_class_pixels:
                selected_class_hits[class_id] += 1

    return tuple(selected)


def _selection_score(
    info: _SampleClassInfo,
    *,
    selected_class_hits: Sequence[int],
    selected_group_counts: Mapping[str, int],
    selected_location_bucket_counts: Mapping[str, int],
    class_quota: Mapping[int, int],
    group_quota: int,
    location_bucket_quota: int,
    min_class_pixels: int,
    class_availability: Sequence[int],
    tie_breaker: float,
) -> float:
    class_score = 0.0
    for class_id, quota in class_quota.items():
        if info.class_counts[class_id] < min_class_pixels:
            continue
        deficit = max(0, quota - selected_class_hits[class_id]) / quota
        rarity = 1.0 / math.sqrt(max(1, class_availability[class_id]))
        class_score += deficit * (1.0 + rarity)

    group_count = selected_group_counts.get(info.group, 0)
    group_score = max(0, group_quota - group_count) / group_quota
    group_penalty = group_count / max(1, group_quota)
    location_bucket_count = selected_location_bucket_counts.get(info.location_bucket, 0)
    location_bucket_score = (
        max(0, location_bucket_quota - location_bucket_count) / location_bucket_quota
    )
    location_bucket_penalty = location_bucket_count / max(1, location_bucket_quota)
    return (
        4.0 * class_score
        + group_score
        + 0.5 * location_bucket_score
        - 0.05 * group_penalty
        - 0.03 * location_bucket_penalty
        + tie_breaker * 1e-6
    )


def _remaining_without_balance_overflow(
    infos: Sequence[_SampleClassInfo],
    selected_ids: set[int],
    selected_group_counts: Mapping[str, int],
    max_group_samples: int,
    selected_location_bucket_counts: Mapping[str, int],
    max_location_bucket_samples: int,
) -> int:
    return sum(
        1
        for info in infos
        if id(info) not in selected_ids
        and selected_group_counts.get(info.group, 0) < max_group_samples
        and selected_location_bucket_counts.get(info.location_bucket, 0)
        < max_location_bucket_samples
    )


def _manifest_result(
    config: VAEResearchManifestConfig,
    *,
    candidate_samples: int,
    selected: Sequence[_SampleClassInfo],
) -> VAEResearchManifestResult:
    class_hits = _class_hits(selected, min_class_pixels=config.min_class_pixels)
    class_pixels = [0 for _ in OPENEARTHMAP_CLASSES]
    total_pixels = 0
    group_counts: dict[str, int] = {}
    location_bucket_counts: dict[str, int] = {}
    for info in selected:
        total_pixels += info.total_pixels
        group_counts[info.group] = group_counts.get(info.group, 0) + 1
        location_bucket_counts[info.location_bucket] = (
            location_bucket_counts.get(info.location_bucket, 0) + 1
        )
        for class_id, count in enumerate(info.class_counts):
            class_pixels[class_id] += count

    useful_class_ids = _useful_class_ids(include_unlabeled=config.include_unlabeled)
    missing_classes = [
        OPENEARTHMAP_CLASSES[class_id] for class_id in useful_class_ids if class_hits[class_id] == 0
    ]
    return VAEResearchManifestResult(
        base_manifest_path=config.base_manifest_path,
        output_manifest_path=config.output_manifest_path,
        report_path=config.report_path,
        split=config.split,
        output_split=config.output_split,
        seed=config.seed,
        candidate_samples=candidate_samples,
        selected_samples=len(selected),
        selected_groups=len(group_counts),
        class_hits={
            OPENEARTHMAP_CLASSES[class_id]: class_hits[class_id]
            for class_id in range(len(OPENEARTHMAP_CLASSES))
        },
        class_pixel_fraction={
            OPENEARTHMAP_CLASSES[class_id]: (
                float(class_pixels[class_id] / total_pixels) if total_pixels else 0.0
            )
            for class_id in range(len(OPENEARTHMAP_CLASSES))
        },
        group_counts=dict(sorted(group_counts.items())),
        location_bucket_counts=dict(sorted(location_bucket_counts.items())),
        missing_classes=missing_classes,
    )


def _class_hits(
    infos: Sequence[_SampleClassInfo],
    *,
    min_class_pixels: int,
) -> list[int]:
    hits = [0 for _ in OPENEARTHMAP_CLASSES]
    for info in infos:
        for class_id, count in enumerate(info.class_counts):
            if count >= min_class_pixels:
                hits[class_id] += 1
    return hits


def _samples_for_split(
    manifest_path: Path,
    split: str,
) -> tuple[SatelliteSample, ...]:
    split_names = _split_names(split)
    samples = tuple(read_manifest(manifest_path))
    if split_names == ("*",):
        return samples
    return tuple(sample for sample in samples if sample.split in split_names)


def _split_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if len(names) == 1 and names[0].lower() in {"*", "all"}:
        return ("*",)
    return names


def _sample_group(sample: SatelliteSample) -> str:
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


def _sample_location_bucket(sample: SatelliteSample, *, group: str) -> str:
    tile_id = _trailing_tile_id(sample.image_path.stem)
    if tile_id is None:
        return f"{group}:tile-unknown"
    return f"{group}:tile-{tile_id % LOCATION_BUCKETS_PER_GROUP}"


def _trailing_tile_id(stem: str) -> int | None:
    index = len(stem)
    while index > 0 and stem[index - 1].isdigit():
        index -= 1
    if index == len(stem):
        return None
    return int(stem[index:])


def _stem_group(stem: str) -> str:
    for separator in ("_", "-"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem


def _useful_class_ids(*, include_unlabeled: bool) -> tuple[int, ...]:
    start = 0 if include_unlabeled else 1
    return tuple(range(start, len(OPENEARTHMAP_CLASSES)))


def _best_trial_index(trial_results: Sequence[Mapping[str, Any]]) -> int:
    if not trial_results:
        return -1
    return max(
        range(len(trial_results)),
        key=lambda index: float(
            trial_results[index]["training"]["custom_vae"].get("ssim", float("-inf"))
        ),
    )


def _trial_stem(name: str, trial_index: int, dataset_seed: int, training_seed: int) -> str:
    return f"{_safe_name(name)}-t{trial_index:02d}-data{dataset_seed}-train{training_seed}"


def _sweep_class_index_path(config: VAEResearchSweepConfig) -> Path:
    if config.class_index_path:
        return Path(config.class_index_path)
    return Path(config.manifest_dir) / f"openearthmap-class-index-v{CLASS_INDEX_VERSION}.json"


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )
    return safe.strip(".-") or "vae-research"


def _validate_manifest_config(config: VAEResearchManifestConfig) -> None:
    for name in ("samples", "seed", "min_class_pixels"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.max_group_fraction <= 0.0 or config.max_group_fraction > 1.0:
        raise ValueError("max_group_fraction must be greater than 0 and at most 1")
    if config.max_location_bucket_fraction <= 0.0 or config.max_location_bucket_fraction > 1.0:
        raise ValueError("max_location_bucket_fraction must be greater than 0 and at most 1")
    for name in ("base_manifest_path", "output_manifest_path", "report_path", "split"):
        if not getattr(config, name):
            raise ValueError(f"{name} is required")


def _validate_sweep_config(config: VAEResearchSweepConfig) -> None:
    for name in (
        "trials",
        "samples",
        "validation_samples",
        "image_size",
        "eval_size",
        "batch_size",
        "steps",
        "learning_rate",
        "latent_channels",
        "hidden_channels",
        "downsample_factor",
        "cache_images",
        "log_every",
        "grid_samples",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.train_fraction <= 0.0 or config.train_fraction >= 1.0:
        raise ValueError("train_fraction must be greater than 0 and less than 1")
    for name in (
        "kl_weight",
        "mse_weight",
        "reconstruction_exponential_weight",
        "edge_weight",
        "class_balanced_l1_weight",
        "class_balanced_edge_weight",
    ):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if config.reconstruction_exponential_scale <= 0.0:
        raise ValueError("reconstruction_exponential_scale must be positive")
    if config.class_balanced_crop_probability < 0.0 or config.class_balanced_crop_probability > 1.0:
        raise ValueError("class_balanced_crop_probability must be between 0 and 1")
    if config.samples <= config.validation_samples:
        raise ValueError("samples must be greater than validation_samples")
    remaining_after_train = config.samples - int(config.samples * config.train_fraction)
    if remaining_after_train < config.validation_samples:
        raise ValueError(
            "samples and train_fraction must leave at least validation_samples held-out samples"
        )
    if config.max_group_fraction <= 0.0 or config.max_group_fraction > 1.0:
        raise ValueError("max_group_fraction must be greater than 0 and at most 1")
    if config.max_location_bucket_fraction <= 0.0 or config.max_location_bucket_fraction > 1.0:
        raise ValueError("max_location_bucket_fraction must be greater than 0 and at most 1")
    if config.validation_split not in {"sample", "city"}:
        raise ValueError("validation_split must be one of: sample, city")
    if config.downsample_factor != 4:
        raise ValueError("downsample_factor must be 4")
    if config.hidden_channels % 8 != 0:
        raise ValueError("hidden_channels must be divisible by 8")
    for name in ("name", "base_manifest_path", "manifest_dir", "artifact_dir", "split"):
        if not getattr(config, name):
            raise ValueError(f"{name} is required")


def _dataclass_from_dict(cls: type[Any], values: Mapping[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    return cls(**{name: values[name] for name in names if name in values})


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == total or total <= 10:
        return True
    return current % max(1, total // 10) == 0


@click.command(
    "vae-research-manifest",
    help="Build one balanced OpenEarthMap VAE research manifest without training.",
)
@click.option(
    "--target",
    type=click.Choice(["local", "modal"]),
    default="modal",
    show_default=True,
    help="Execution target.",
)
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def manifest_cli(target: str, config_path: Path) -> None:
    try:
        config = load_vae_research_manifest_config_file(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    if target == "modal":
        from sim_satellite.modal_app import run_vae_research_manifest_target

        result = run_vae_research_manifest_target(config)
    else:
        result = asdict(build_vae_research_manifest(config, progress=click.echo))
    click.echo(json.dumps(result, indent=2))


@click.command(
    "vae-research-sweep",
    help="Build balanced OpenEarthMap VAE research manifests and train trial seeds.",
)
@click.option(
    "--target",
    type=click.Choice(["local", "modal"]),
    default="modal",
    show_default=True,
    help="Execution target.",
)
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def sweep_cli(target: str, config_path: Path) -> None:
    try:
        config = load_vae_research_sweep_config_file(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    if target == "modal":
        from sim_satellite.modal_app import run_vae_research_sweep_target

        result = run_vae_research_sweep_target(config)
    else:
        result = asdict(run_vae_research_sweep(config, progress=click.echo))
    click.echo(json.dumps(result, indent=2))
