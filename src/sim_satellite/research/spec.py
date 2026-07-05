"""Benchmark spec models, validation, and identity hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

DEFAULT_RUNS_DIR = "benchmarks/runs"
DEFAULT_CACHE_DIR = "benchmarks/cache"
RUN_RECORD_VERSION = 1
TARGET_DIRECTIONS = ("higher", "lower")
EVAL_PROTOCOL_FIELDS = (
    "manifest_path",
    "split",
    "seed",
    "samples",
    "image_size",
    "segmentation_checkpoint_path",
    "eval_sample_identities",
)
SAMPLER_PROTOCOL_FIELDS = ("generation_steps", "guidance_scale", "precision", "batch_size")


@dataclass(frozen=True)
class BenchmarkProtocol:
    """The frozen evaluation contract shared by comparable benchmark runs."""

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
    eval_sample_identities: tuple[str, ...] = ()

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
    generation_steps: int = 0
    guidance_scale: float | None = None
    precision: str = ""
    batch_size: int = 0

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BenchmarkRunOptions:
        return _dataclass_from_dict(cls, values)


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


def benchmark_eval_hash(protocol: BenchmarkProtocol) -> str:
    """Hash the evaluation identity, including the pinned sample list."""

    eval_fields = {name: getattr(protocol, name) for name in EVAL_PROTOCOL_FIELDS}
    canonical = json.dumps(eval_fields, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def benchmark_run_protocol(
    protocol: BenchmarkProtocol,
    options: BenchmarkRunOptions,
) -> BenchmarkProtocol:
    updates: dict[str, Any] = {}
    if options.generation_steps > 0:
        updates["generation_steps"] = options.generation_steps
    if options.guidance_scale is not None:
        updates["guidance_scale"] = options.guidance_scale
    if options.precision:
        updates["precision"] = options.precision
    if options.batch_size > 0:
        updates["batch_size"] = options.batch_size
    if not updates:
        return protocol
    run_protocol = replace(protocol, **updates)
    _validate_protocol(run_protocol)
    return run_protocol


def benchmark_protocol_sample_digest(protocol: BenchmarkProtocol) -> str:
    if not protocol.eval_sample_identities:
        return ""
    return sample_identity_digest(protocol.eval_sample_identities)


def sample_identity_digest(identities: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


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
    if protocol.eval_sample_identities and len(protocol.eval_sample_identities) != protocol.samples:
        raise ValueError(
            "eval_sample_identities length must match samples "
            f"({len(protocol.eval_sample_identities)} != {protocol.samples})"
        )
    duplicates = sorted(
        {
            identity
            for identity in protocol.eval_sample_identities
            if protocol.eval_sample_identities.count(identity) > 1
        }
    )
    if duplicates:
        raise ValueError(f"eval_sample_identities contains duplicates: {', '.join(duplicates)}")


def validate_run_options(options: BenchmarkRunOptions) -> None:
    if not options.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not options.experiment:
        raise ValueError("experiment is required")
    if options.generation_steps < 0:
        raise ValueError("generation_steps override must be non-negative")
    if options.guidance_scale is not None and options.guidance_scale < 0.0:
        raise ValueError("guidance_scale override must be non-negative")
    if options.precision and options.precision not in {"float32", "bf16"}:
        raise ValueError("precision override must be one of: bf16, float32")
    if options.batch_size < 0:
        raise ValueError("batch_size override must be non-negative")


def _reject_target(target: object) -> BenchmarkTarget:
    raise ValueError(f"Benchmark target must be an object: {target!r}")


def _dataclass_from_dict(cls: type[Any], values: Mapping[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    parsed = {name: values[name] for name in names if name in values}
    if cls is BenchmarkProtocol and "eval_sample_identities" in parsed:
        parsed["eval_sample_identities"] = _string_tuple(
            parsed["eval_sample_identities"],
            field_name="eval_sample_identities",
        )
    return cls(**parsed)


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings")
    items = tuple(value)
    if not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return items
