"""Frozen benchmark harness for mask-conditioned flow-matching checkpoints.

The benchmark exists so many parallel research agents can score independent
experiments against one shared, committed protocol and compare the results.
A benchmark spec pins the evaluation protocol (data holdout, seeds, sampler
settings, frozen segmentation judge) plus explicit competitiveness targets.
Every scored checkpoint produces one small JSON run record that is written
under `benchmarks/runs/` and committed, so worktree agents never conflict.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch
import torch.nn.functional as F

from sim_satellite import evaluation_metrics, training_data
from sim_satellite.data.loader import SatelliteSample, iter_batches, read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.generation import generate_conditioned_images
from sim_satellite.infer import LoadedGenerator, _load_checkpoint, load_generator_from_checkpoint
from sim_satellite.segmentation.checkpoint import (
    SegmentationCheckpoint,
    _load_segmentation_checkpoint,
)
from sim_satellite.segmentation.metrics import _confusion_matrix, _segmentation_metrics
from sim_satellite.segmentation.models import _forward_segmentation_model
from sim_satellite.segmentation.vae import _encode_vae_features
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH
from sim_satellite.vae.research import _sample_group

ProgressReporter = Callable[[str], None]
DEFAULT_RUNS_DIR = "benchmarks/runs"
DEFAULT_CACHE_DIR = "benchmarks/cache"
RUN_RECORD_VERSION = 1
REAL_ADHERENCE_CACHE_VERSION = 1
REAL_ADHERENCE_CACHE_KIND = "real-mask-adherence"
TARGET_DIRECTIONS = ("higher", "lower")


@dataclass(frozen=True)
class BenchmarkProtocol:
    """The frozen evaluation contract shared by every comparable run.

    Never edit a committed protocol in place; copy the spec file to a new
    version instead. Runs are only comparable within one protocol hash.
    """

    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    seed: int = 20260703
    samples: int = 64
    batch_size: int = 8
    image_size: int = 256
    generation_steps: int = 32
    guidance_scale: float = 1.0
    precision: str = "float32"
    segmentation_checkpoint_path: str = ""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BenchmarkProtocol:
        return _dataclass_from_dict(cls, values)


@dataclass(frozen=True)
class BenchmarkTarget:
    metric: str
    target: float
    direction: str
    label: str = ""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BenchmarkTarget:
        return _dataclass_from_dict(cls, values)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    protocol: BenchmarkProtocol
    primary_metric: str
    targets: tuple[BenchmarkTarget, ...]
    description: str = ""


@dataclass(frozen=True)
class BenchmarkRunOptions:
    checkpoint_path: str
    experiment: str
    notes: str = ""
    device: str = "cpu"
    git_commit: str = ""
    manifest_path: str = ""
    segmentation_checkpoint_path: str = ""
    mask_adherence: bool = True

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BenchmarkRunOptions:
        return _dataclass_from_dict(cls, values)


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
        }


def load_benchmark_spec_file(spec_path: Path) -> BenchmarkSpec:
    values = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Benchmark spec JSON must be an object")
    return load_benchmark_spec_values(values)


def load_benchmark_spec_values(values: Mapping[str, Any]) -> BenchmarkSpec:
    known = {"name", "description", "protocol", "primary_metric", "targets"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown benchmark spec fields: {', '.join(unknown)}")
    protocol_values = values.get("protocol")
    if not isinstance(protocol_values, Mapping):
        raise ValueError("Benchmark spec protocol must be an object")
    target_values = values.get("targets")
    if not isinstance(target_values, Sequence) or isinstance(target_values, str):
        raise ValueError("Benchmark spec targets must be a list")
    spec = BenchmarkSpec(
        name=str(values.get("name", "")),
        description=str(values.get("description", "")),
        protocol=BenchmarkProtocol.from_dict(protocol_values),
        primary_metric=str(values.get("primary_metric", "")),
        targets=tuple(
            BenchmarkTarget.from_dict(target)
            if isinstance(target, Mapping)
            else _reject_target(target)
            for target in target_values
        ),
    )
    _validate_spec(spec)
    return spec


def benchmark_protocol_hash(protocol: BenchmarkProtocol) -> str:
    canonical = json.dumps(asdict(protocol), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    grouped: dict[str, list[SatelliteSample]] = {}
    for sample in split_samples:
        grouped.setdefault(_sample_group(sample), []).append(sample)
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

    pool.sort(key=lambda sample: _sample_identity(sample))
    random.Random(protocol.seed).shuffle(pool)
    eval_samples = tuple(pool[: protocol.samples])
    holdout_patterns = sorted({group.split(":", 1)[-1] for group in eval_groups})
    digest_source = "\n".join(sorted(_sample_identity(sample) for sample in eval_samples))
    by_source: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for sample in all_samples:
        by_source[sample.source] = by_source.get(sample.source, 0) + 1
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
    return BenchmarkSelection(
        eval_samples=eval_samples,
        eval_groups=tuple(sorted(eval_groups)),
        validation_holdout_patterns=",".join(holdout_patterns),
        sample_digest=hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
        manifest_path=str(path),
        manifest_samples=len(all_samples),
        manifest_samples_by_source=dict(sorted(by_source.items())),
        manifest_samples_by_split=dict(sorted(by_split.items())),
    )


def run_benchmark(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    real_adherence_cache: Mapping[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    _validate_run_options(options)
    started_at_epoch_utc = time.time()
    started_at_monotonic = time.monotonic()
    protocol = spec.protocol
    protocol_hash = benchmark_protocol_hash(protocol)
    selection = select_benchmark_samples(protocol, manifest_path=options.manifest_path)
    device = torch.device(options.device)

    checkpoint_path = Path(options.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Benchmark checkpoint not found: {checkpoint_path}")
    raw_checkpoint = _load_checkpoint(checkpoint_path, torch=torch)
    generator = load_generator_from_checkpoint(
        raw_checkpoint,
        checkpoint_path=checkpoint_path,
        device=options.device,
    )
    _require_protocol_compatible_model(generator, protocol)
    scorer = _load_mask_adherence_scorer(
        spec,
        options,
        protocol_hash=protocol_hash,
        sample_digest=selection.sample_digest,
        real_adherence_cache=real_adherence_cache,
        progress=progress,
    )

    accumulator = evaluation_metrics.GenerationMetricAccumulator(
        class_count=generator.model_config.mask_channels,
        image_channels=generator.model_config.image_channels,
        torch=torch,
    )
    preloaded = training_data.preload_samples(selection.eval_samples)
    batches = tuple(iter_batches(selection.eval_samples, protocol.batch_size))
    for batch_index, batch in enumerate(batches):
        prepared = tuple(preloaded[sample] for sample in batch)
        crop_boxes = [
            training_data.center_square_crop_box(sample, crop_size=protocol.image_size)
            for sample in prepared
        ]
        clean, semantic_mask = training_data.load_batch(
            prepared,
            crop_boxes,
            torch=torch,
            model_config=generator.model_config,
            device=device,
        )
        generated = generate_conditioned_images(
            generator.model,
            semantic_mask,
            torch=torch,
            model_config=generator.model_config,
            device=device,
            seed=protocol.seed + 20_000 + batch_index,
            steps=protocol.generation_steps,
            guidance_scale=protocol.guidance_scale,
            precision=protocol.precision,
        )
        real = (clean.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        accumulator.update(generated, real, semantic_mask=semantic_mask, torch=torch)
        if scorer is not None:
            scorer.update(generated, real, labels=semantic_mask.argmax(dim=1))
        _report(
            progress,
            f"Scored benchmark batch {batch_index + 1}/{len(batches)} "
            f"({accumulator.sample_count}/{protocol.samples} samples)",
        )

    generation_payload = accumulator.payload(
        requested_sample_count=protocol.samples,
        generation_steps=protocol.generation_steps,
        guidance_scale=protocol.guidance_scale,
    )
    mask_adherence_payload = (
        scorer.payload() if scorer is not None else _disabled_mask_adherence_payload(spec, options)
    )
    record: dict[str, Any] = {
        "record_version": RUN_RECORD_VERSION,
        "benchmark": spec.name,
        "protocol_hash": protocol_hash,
        "protocol": asdict(protocol),
        "experiment": options.experiment,
        "notes": options.notes,
        "git_commit": options.git_commit,
        "created_at_epoch_utc": started_at_epoch_utc,
        "device": str(device),
        "checkpoint": _checkpoint_payload(generator, raw_checkpoint),
        "data": selection.payload(),
        "generation": generation_payload,
        "mask_adherence": mask_adherence_payload,
        "wall_seconds": max(0.0, time.monotonic() - started_at_monotonic),
    }
    record["run_id"] = _run_id(options.experiment, started_at_epoch_utc, record)
    if scorer is not None:
        record["real_adherence_cache"] = scorer.real_cache_payload(
            protocol_hash=protocol_hash,
            sample_digest=selection.sample_digest,
        )
    return record


def write_run_record(record: Mapping[str, Any], runs_dir: Path) -> Path:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Benchmark run record is missing run_id")
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.json"
    if path.exists():
        raise RuntimeError(f"Benchmark run record already exists: {path}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class _MaskAdherenceScorer:
    """Score generated and real images against the conditioning masks.

    A frozen segmentation checkpoint acts as the judge: generated images are
    segmented and compared to the mask the generator was conditioned on. The
    same score on real images gives the judge's own ceiling, so `iou_ratio`
    stays meaningful even while the segmentation model itself improves.

    The real-image side is identical for every run under one protocol (same
    frozen eval samples, same judge), so it can be seeded from a cache and
    skipped entirely; only the generated images are then segmented.
    """

    def __init__(
        self,
        checkpoint: SegmentationCheckpoint,
        *,
        checkpoint_path: str,
        checkpoint_sha256: str,
        cached_real_confusion: torch.Tensor | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.model = checkpoint.model
        self.vae_encoder = checkpoint.vae_encoder
        self.image_size = int(checkpoint.config["image_size"])
        self.use_context = int(checkpoint.config.get("context_size", 0)) > self.image_size
        class_count = len(OPENEARTHMAP_CLASSES)
        self.generated_confusion = torch.zeros((class_count, class_count), dtype=torch.long)
        self.real_scores_from_cache = cached_real_confusion is not None
        self.real_confusion = (
            cached_real_confusion
            if cached_real_confusion is not None
            else torch.zeros((class_count, class_count), dtype=torch.long)
        )

    def update(self, generated: torch.Tensor, real: torch.Tensor, *, labels: torch.Tensor) -> None:
        labels = labels.detach().cpu()
        self.generated_confusion += _confusion_matrix(
            self._predict(generated),
            labels,
            num_classes=len(OPENEARTHMAP_CLASSES),
        )
        if not self.real_scores_from_cache:
            self.real_confusion += _confusion_matrix(
                self._predict(real),
                labels,
                num_classes=len(OPENEARTHMAP_CLASSES),
            )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        target_size = (int(images.shape[-2]), int(images.shape[-1]))
        batch = images.detach().float().clamp(0.0, 1.0)
        if target_size != (self.image_size, self.image_size):
            batch = F.interpolate(
                batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_context:
            batch = torch.cat((batch, batch), dim=1)
        with torch.no_grad():
            predictions = _forward_segmentation_model(
                self.model,
                batch,
                vae_latents=_encode_vae_features(self.vae_encoder, batch),
            ).argmax(dim=1)
        if tuple(predictions.shape[-2:]) != target_size:
            predictions = (
                F.interpolate(
                    predictions.float().unsqueeze(1),
                    size=target_size,
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        return predictions.detach().cpu()

    def payload(self) -> dict[str, Any]:
        generated = _segmentation_metrics(self.generated_confusion, include_unlabeled_iou=False)
        real = _segmentation_metrics(self.real_confusion, include_unlabeled_iou=False)
        iou_ratio = (
            float(generated["mean_iou"]) / float(real["mean_iou"])
            if float(real["mean_iou"]) > 0.0
            else None
        )
        return {
            "enabled": True,
            "segmentation_checkpoint_path": self.checkpoint_path,
            "generated_mean_iou": float(generated["mean_iou"]),
            "generated_pixel_accuracy": float(generated["pixel_accuracy"]),
            "generated_class_iou": generated["class_iou"],
            "real_mean_iou": float(real["mean_iou"]),
            "real_pixel_accuracy": float(real["pixel_accuracy"]),
            "real_class_iou": real["class_iou"],
            "iou_ratio": iou_ratio,
            "real_scores_from_cache": self.real_scores_from_cache,
        }

    def real_cache_payload(self, *, protocol_hash: str, sample_digest: str) -> dict[str, Any]:
        return {
            "cache_version": REAL_ADHERENCE_CACHE_VERSION,
            "kind": REAL_ADHERENCE_CACHE_KIND,
            "protocol_hash": protocol_hash,
            "sample_digest": sample_digest,
            "segmentation_checkpoint_sha256": self.checkpoint_sha256,
            "confusion": self.real_confusion.tolist(),
        }


def real_adherence_cache_path(cache_dir: Path, protocol_hash: str) -> Path:
    return cache_dir / f"real-adherence-{protocol_hash[:16]}.json"


def load_real_adherence_cache(cache_dir: Path, protocol_hash: str) -> dict[str, Any] | None:
    path = real_adherence_cache_path(cache_dir, protocol_hash)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def write_real_adherence_cache(cache_dir: Path, payload: Mapping[str, Any]) -> Path:
    protocol_hash = payload.get("protocol_hash")
    if not isinstance(protocol_hash, str) or not protocol_hash:
        raise ValueError("Real-adherence cache payload is missing protocol_hash")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = real_adherence_cache_path(cache_dir, protocol_hash)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _cached_real_confusion(
    cache: Mapping[str, Any] | None,
    *,
    protocol_hash: str,
    sample_digest: str,
    checkpoint_sha256: str,
) -> torch.Tensor | None:
    if not isinstance(cache, Mapping):
        return None
    if cache.get("cache_version") != REAL_ADHERENCE_CACHE_VERSION:
        return None
    if cache.get("kind") != REAL_ADHERENCE_CACHE_KIND:
        return None
    if cache.get("protocol_hash") != protocol_hash:
        return None
    if cache.get("sample_digest") != sample_digest:
        return None
    if cache.get("segmentation_checkpoint_sha256") != checkpoint_sha256:
        return None
    confusion = cache.get("confusion")
    class_count = len(OPENEARTHMAP_CLASSES)
    if not isinstance(confusion, list) or len(confusion) != class_count:
        return None
    try:
        tensor = torch.tensor(confusion, dtype=torch.long)
    except (TypeError, ValueError):
        return None
    if tuple(tensor.shape) != (class_count, class_count):
        return None
    return tensor


def _load_mask_adherence_scorer(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    protocol_hash: str,
    sample_digest: str,
    real_adherence_cache: Mapping[str, Any] | None,
    progress: ProgressReporter | None,
) -> _MaskAdherenceScorer | None:
    segmentation_checkpoint_path = (
        options.segmentation_checkpoint_path or spec.protocol.segmentation_checkpoint_path
    )
    if not options.mask_adherence or not segmentation_checkpoint_path:
        return None
    checkpoint = _load_segmentation_checkpoint(
        Path(segmentation_checkpoint_path),
        device=options.device,
    )
    checkpoint_sha256 = _file_sha256(Path(segmentation_checkpoint_path))
    cached_real_confusion = _cached_real_confusion(
        real_adherence_cache,
        protocol_hash=protocol_hash,
        sample_digest=sample_digest,
        checkpoint_sha256=checkpoint_sha256,
    )
    if cached_real_confusion is not None:
        _report(progress, "Reusing cached real-image mask-adherence scores")
    return _MaskAdherenceScorer(
        checkpoint,
        checkpoint_path=segmentation_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        cached_real_confusion=cached_real_confusion,
    )


def _disabled_mask_adherence_payload(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
) -> dict[str, Any]:
    if not spec.protocol.segmentation_checkpoint_path:
        status = "disabled: protocol has no segmentation_checkpoint_path"
    else:
        status = "disabled: run used --no-mask-adherence"
    return {"enabled": False, "status": status}


def _require_protocol_compatible_model(
    generator: LoadedGenerator,
    protocol: BenchmarkProtocol,
) -> None:
    model_config = generator.model_config
    if model_config.image_size != protocol.image_size:
        raise ValueError(
            f"Checkpoint image_size {model_config.image_size} does not match "
            f"benchmark protocol image_size {protocol.image_size}: "
            f"{generator.checkpoint_path}"
        )
    if model_config.mask_channels != len(OPENEARTHMAP_CLASSES):
        raise ValueError(
            f"Checkpoint mask_channels {model_config.mask_channels} does not match "
            f"the {len(OPENEARTHMAP_CLASSES)} OpenEarthMap classes: "
            f"{generator.checkpoint_path}"
        )


def _checkpoint_payload(
    generator: LoadedGenerator,
    raw_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    training_config = raw_checkpoint.get("training_config")
    step = raw_checkpoint.get("step")
    return {
        "path": str(generator.checkpoint_path),
        "model_config": asdict(generator.model_config),
        "model_parameters": sum(parameter.numel() for parameter in generator.model.parameters()),
        "step": int(step) if isinstance(step, int) else None,
        "training_config": dict(training_config) if isinstance(training_config, Mapping) else None,
    }


def _group_rank(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_identity(sample: SatelliteSample) -> str:
    return "/".join(sample.image_path.parts[-3:])


def _run_id(experiment: str, created_at_epoch_utc: float, record: Mapping[str, Any]) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(created_at_epoch_utc))
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    return f"{timestamp}-{_safe_name(experiment)}-{digest}"


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )
    return safe.strip(".-") or "experiment"


def _validate_spec(spec: BenchmarkSpec) -> None:
    if not spec.name:
        raise ValueError("Benchmark spec name is required")
    if not spec.targets:
        raise ValueError("Benchmark spec requires at least one target")
    metrics = [target.metric for target in spec.targets]
    duplicates = sorted({metric for metric in metrics if metrics.count(metric) > 1})
    if duplicates:
        raise ValueError(f"Benchmark targets repeat metrics: {', '.join(duplicates)}")
    for target in spec.targets:
        if not target.metric:
            raise ValueError("Benchmark target metric is required")
        if target.direction not in TARGET_DIRECTIONS:
            raise ValueError(
                f"Benchmark target direction must be one of: {', '.join(TARGET_DIRECTIONS)}"
            )
    if not spec.primary_metric:
        raise ValueError("Benchmark spec primary_metric is required")
    if spec.primary_metric not in metrics:
        raise ValueError(
            f"Benchmark primary_metric {spec.primary_metric!r} must be one of the target metrics"
        )
    _validate_protocol(spec.protocol)


def _validate_protocol(protocol: BenchmarkProtocol) -> None:
    for name in ("samples", "batch_size", "generation_steps"):
        if getattr(protocol, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if protocol.seed < 0:
        raise ValueError("seed must be non-negative")
    if protocol.guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if protocol.image_size < 32 or protocol.image_size % 8 != 0:
        raise ValueError("image_size must be at least 32 and divisible by 8")
    if protocol.precision not in {"float32", "bf16"}:
        raise ValueError("precision must be one of: bf16, float32")
    for name in ("manifest_path", "split"):
        if not getattr(protocol, name):
            raise ValueError(f"{name} is required")


def _validate_run_options(options: BenchmarkRunOptions) -> None:
    if not options.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not options.experiment:
        raise ValueError("experiment is required")


def _reject_target(target: object) -> BenchmarkTarget:
    raise ValueError(f"Benchmark target must be an object: {target!r}")


def _dataclass_from_dict(cls: type[Any], values: Mapping[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    return cls(**{name: values[name] for name in names if name in values})


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _run_with_target(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    target: str,
    holdout_only: bool,
    real_adherence_cache: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if target == "local":
        if holdout_only:
            result = select_benchmark_samples(
                spec.protocol,
                manifest_path=options.manifest_path,
            ).payload()
            result["protocol_hash"] = benchmark_protocol_hash(spec.protocol)
        else:
            result = run_benchmark(
                spec,
                options,
                real_adherence_cache=real_adherence_cache,
                progress=click.echo,
            )
        result["execution_target"] = "local"
        return result

    if target != "modal":
        raise ValueError(f"Unsupported benchmark target: {target}")

    from sim_satellite.modal_app import run_benchmark_target

    return run_benchmark_target(
        spec,
        options,
        holdout_only=holdout_only,
        real_adherence_cache=real_adherence_cache,
    )


@click.command(
    "benchmark",
    help="Score a diffusion checkpoint against a frozen benchmark spec.",
)
@click.option(
    "--target",
    type=click.Choice(["modal", "local"]),
    default="modal",
    show_default=True,
)
@click.option("--checkpoint", "checkpoint_path", default="", help="Diffusion checkpoint to score.")
@click.option("--experiment", default="", help="Short experiment name recorded in the run id.")
@click.option("--notes", default="", help="Free-form notes stored in the run record.")
@click.option("--device", default="cpu", show_default=True, help="Device for --target local.")
@click.option(
    "--runs-dir",
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    help="Directory where the run record JSON is written.",
)
@click.option(
    "--mask-adherence/--no-mask-adherence",
    default=True,
    show_default=True,
    help="Score generated images against conditioning masks with the frozen segmenter.",
)
@click.option(
    "--cache-dir",
    default=DEFAULT_CACHE_DIR,
    show_default=True,
    help="Directory for committed per-protocol caches such as real-image judge scores.",
)
@click.option(
    "--holdout-only",
    is_flag=True,
    help="Print the frozen eval selection and holdout patterns without scoring.",
)
@click.argument(
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def benchmark_cli(
    target: str,
    checkpoint_path: str,
    experiment: str,
    notes: str,
    device: str,
    runs_dir: str,
    mask_adherence: bool,
    cache_dir: str,
    holdout_only: bool,
    spec_path: Path,
) -> None:
    try:
        spec = load_benchmark_spec_file(spec_path)
        options = BenchmarkRunOptions(
            checkpoint_path=checkpoint_path,
            experiment=experiment,
            notes=notes,
            device=device,
            git_commit=_current_git_commit(),
            mask_adherence=mask_adherence,
        )
        real_adherence_cache = None
        if not holdout_only:
            _validate_run_options(options)
            real_adherence_cache = load_real_adherence_cache(
                Path(cache_dir),
                benchmark_protocol_hash(spec.protocol),
            )
        result = _run_with_target(
            spec,
            options,
            target=target,
            holdout_only=holdout_only,
            real_adherence_cache=real_adherence_cache,
        )
        if not holdout_only:
            cache_payload = result.pop("real_adherence_cache", None)
            if isinstance(cache_payload, Mapping):
                result["real_adherence_cache_path"] = str(
                    write_real_adherence_cache(Path(cache_dir), cache_payload)
                )
            record_path = write_run_record(result, Path(runs_dir))
            result["record_path"] = str(record_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    benchmark_cli()
