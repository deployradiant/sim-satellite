"""Benchmark execution for mask-conditioned flow-matching checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sim_satellite import evaluation_metrics, training_data
from sim_satellite.data.loader import iter_batches
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.generation import generate_conditioned_images
from sim_satellite.infer import LoadedGenerator, _load_checkpoint, load_generator_from_checkpoint
from sim_satellite.research.mask_adherence import (
    disabled_mask_adherence_payload,
    load_mask_adherence_scorer,
)
from sim_satellite.research.selection import select_benchmark_samples
from sim_satellite.research.spec import (
    RUN_RECORD_VERSION,
    BenchmarkProtocol,
    BenchmarkRunOptions,
    BenchmarkSpec,
    benchmark_eval_hash,
    benchmark_run_protocol,
    validate_run_options,
)

ProgressReporter = Callable[[str], None]


def run_benchmark(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    real_adherence_cache: Mapping[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    validate_run_options(options)
    started_at_epoch_utc = time.time()
    started_at_monotonic = time.monotonic()
    protocol = benchmark_run_protocol(spec.protocol, options)
    eval_hash = benchmark_eval_hash(protocol)
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
    scorer = load_mask_adherence_scorer(
        spec,
        options,
        eval_hash=eval_hash,
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
    sample_offset = 0
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
            noise=_per_sample_noise(
                protocol,
                generator.model_config,
                sample_offset=sample_offset,
                count=len(batch),
            ),
        )
        sample_offset += len(batch)
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
        scorer.payload() if scorer is not None else disabled_mask_adherence_payload(spec, options)
    )
    record: dict[str, Any] = {
        "record_version": RUN_RECORD_VERSION,
        "benchmark": spec.name,
        "eval_hash": eval_hash,
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
            eval_hash=eval_hash,
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


def _per_sample_noise(
    protocol: BenchmarkProtocol,
    model_config: Any,
    *,
    sample_offset: int,
    count: int,
) -> torch.Tensor:
    """Seed generation noise per eval sample, not per batch."""

    shape = (model_config.image_channels, model_config.image_size, model_config.image_size)
    noise = [
        torch.randn(
            shape,
            generator=torch.Generator(device="cpu").manual_seed(
                protocol.seed + 20_000 + sample_offset + index
            ),
            device="cpu",
        )
        for index in range(count)
    ]
    return torch.stack(noise)


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


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)
