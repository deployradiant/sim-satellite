from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.model import ModelConfig, build_model
from sim_satellite.research.benchmark import (
    BenchmarkProtocol,
    BenchmarkRunOptions,
    BenchmarkSpec,
    BenchmarkTarget,
    benchmark_eval_hash,
    load_benchmark_spec_values,
    load_real_adherence_cache,
    real_adherence_cache_path,
    run_benchmark,
    sample_identity_digest,
    select_benchmark_samples,
    write_real_adherence_cache,
    write_run_record,
)
from sim_satellite.segmentation.checkpoint import _save_checkpoint
from sim_satellite.segmentation.config import SegmentationTrainingConfig
from sim_satellite.segmentation.models import build_segmentation_model
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig

EVAL_IMAGE_SIZE = 32


def _spec(protocol: BenchmarkProtocol) -> BenchmarkSpec:
    return BenchmarkSpec(
        name="unit-benchmark",
        protocol=protocol,
        primary_metric="generation.ssim",
        targets=(BenchmarkTarget(metric="generation.ssim", target=0.9, direction="higher"),),
    )


def _write_dataset(
    tmp_path: Path,
    *,
    regions: tuple[str, ...] = ("alpha", "beta", "gamma"),
    per_region: int = 2,
) -> Path:
    config = ModelConfig(image_size=EVAL_IMAGE_SIZE, mask_channels=len(OPENEARTHMAP_CLASSES))
    samples = []
    for region_index, region in enumerate(regions):
        for index in range(per_region):
            image_path = tmp_path / "openearthmap" / region / "images" / f"{region}_{index}.png"
            mask_path = tmp_path / "openearthmap" / region / "labels" / f"{region}_{index}.png"
            prepared_path = tmp_path / "openearthmap" / "prepared" / region / f"{region}_{index}.pt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 40), (40 * region_index + 20, 80, 120)).save(image_path)
            mask = Image.new("L", (40, 40), (region_index + index) % 8 + 1)
            mask.save(mask_path)
            sample = SatelliteSample(
                image_path=image_path,
                mask_path=mask_path,
                prepared_path=prepared_path,
                split="train",
                source="openearthmap",
            )
            write_prepared_sample(sample, torch=torch, config=config)
            samples.append(sample)
    manifest_path = tmp_path / "openearthmap" / "manifest.jsonl"
    write_manifest(samples, manifest_path)
    return manifest_path


def _write_diffusion_checkpoint(path: Path, *, image_size: int = EVAL_IMAGE_SIZE) -> ModelConfig:
    config = ModelConfig(
        image_size=image_size,
        mask_channels=len(OPENEARTHMAP_CLASSES),
        hidden_channels=16,
    )
    model = build_model(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": asdict(config), "model_state": model.state_dict()}, path)
    return config


def _write_segmentation_checkpoint(tmp_path: Path) -> Path:
    vae_path = tmp_path / "vae.pt"
    vae_config = SatelliteVAEConfig(latent_channels=4, hidden_channels=8, downsample_factor=4)
    vae = SatelliteVAE(vae_config)
    torch.save(
        {
            "vae_config": asdict(vae_config),
            "model_state": vae.state_dict(),
            "latent_stats": {"mean": [0.0] * 4, "std": [1.0] * 4},
        },
        vae_path,
    )
    checkpoint_path = tmp_path / "segmenter.pt"
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=4,
    )
    _save_checkpoint(
        checkpoint_path,
        model,
        config=asdict(
            SegmentationTrainingConfig(
                image_size=16,
                vae_checkpoint_path=str(vae_path),
            )
        ),
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )
    return checkpoint_path


def test_benchmark_spec_validation_rejects_bad_specs() -> None:
    protocol = {"samples": 4}
    base = {
        "name": "spec",
        "protocol": protocol,
        "primary_metric": "generation.ssim",
        "targets": [{"metric": "generation.ssim", "target": 0.5, "direction": "higher"}],
    }

    load_benchmark_spec_values(base)

    with pytest.raises(ValueError, match="Unknown benchmark spec fields"):
        load_benchmark_spec_values({**base, "extra": 1})
    with pytest.raises(ValueError, match="direction"):
        load_benchmark_spec_values(
            {
                **base,
                "targets": [{"metric": "generation.ssim", "target": 0.5, "direction": "up"}],
            }
        )
    with pytest.raises(ValueError, match="primary_metric"):
        load_benchmark_spec_values({**base, "primary_metric": "generation.mse"})
    with pytest.raises(ValueError, match="repeat metrics"):
        load_benchmark_spec_values(
            {
                **base,
                "targets": [
                    {"metric": "generation.ssim", "target": 0.5, "direction": "higher"},
                    {"metric": "generation.ssim", "target": 0.6, "direction": "higher"},
                ],
            }
        )
    with pytest.raises(ValueError, match="precision"):
        load_benchmark_spec_values({**base, "protocol": {"precision": "fp16"}})
    with pytest.raises(ValueError, match="eval_sample_identities length"):
        load_benchmark_spec_values(
            {
                **base,
                "protocol": {
                    "samples": 2,
                    "eval_sample_identities": ["openearthmap:train:alpha/images/a.png"],
                },
            }
        )


def test_benchmark_eval_hash_pins_eval_fields_not_sampler_config() -> None:
    protocol = BenchmarkProtocol(samples=4)

    assert benchmark_eval_hash(protocol) == benchmark_eval_hash(BenchmarkProtocol(samples=4))
    for sampler_change in (
        {"generation_steps": 64},
        {"guidance_scale": 2.0},
        {"precision": "bf16"},
        {"batch_size": 1},
    ):
        assert benchmark_eval_hash(protocol) == benchmark_eval_hash(
            replace(protocol, **sampler_change)
        )
    for eval_change in (
        {"seed": 1},
        {"samples": 8},
        {"image_size": 64},
        {"split": "val"},
        {"manifest_path": "elsewhere.jsonl"},
        {"segmentation_checkpoint_path": "judge.pt"},
        {"eval_sample_identities": ("openearthmap:train:a/images/0.png",) * 4},
    ):
        assert benchmark_eval_hash(protocol) != benchmark_eval_hash(
            replace(protocol, **eval_change)
        )


def test_select_benchmark_samples_holds_out_whole_groups(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, per_region=3)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=3,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
    )

    selection = select_benchmark_samples(protocol)
    repeat = select_benchmark_samples(protocol)

    assert [str(s.image_path) for s in selection.eval_samples] == [
        str(s.image_path) for s in repeat.eval_samples
    ]
    assert selection.sample_digest == repeat.sample_digest
    assert len(selection.eval_samples) == 3
    eval_regions = {group.split(":", 1)[-1] for group in selection.eval_groups}
    assert eval_regions < {"alpha", "beta", "gamma"}
    assert selection.validation_holdout_patterns == ",".join(sorted(eval_regions))
    for sample in selection.eval_samples:
        assert any(region in str(sample.image_path) for region in eval_regions)
    assert selection.manifest_samples == 9
    assert selection.manifest_samples_by_source == {"openearthmap": 9}
    assert selection.manifest_samples_by_split == {"train": 9}


def test_select_benchmark_samples_can_pin_exact_eval_identities(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, regions=("alpha", "beta", "gamma"), per_region=3)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=3,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
    )
    initial = select_benchmark_samples(protocol)
    pinned_protocol = replace(
        protocol,
        eval_sample_identities=initial.eval_sample_identities,
    )
    pinned = select_benchmark_samples(pinned_protocol)

    _write_dataset(tmp_path, regions=("alpha", "beta", "gamma", "delta"), per_region=3)
    repinned = select_benchmark_samples(pinned_protocol)

    assert pinned.eval_sample_identities == initial.eval_sample_identities
    assert repinned.eval_sample_identities == initial.eval_sample_identities
    assert pinned.sample_digest == sample_identity_digest(initial.eval_sample_identities)
    assert repinned.sample_digest == pinned.sample_digest
    assert [str(sample.image_path) for sample in repinned.eval_samples] == [
        str(sample.image_path) for sample in pinned.eval_samples
    ]

    missing_protocol = replace(
        pinned_protocol,
        eval_sample_identities=("openearthmap:train:missing/images/nope.png",) * 3,
    )
    with pytest.raises(RuntimeError, match="missing pinned eval samples"):
        select_benchmark_samples(missing_protocol)


def test_select_benchmark_samples_rejects_undersized_pools(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, per_region=1)

    with pytest.raises(RuntimeError, match="every sample group"):
        select_benchmark_samples(BenchmarkProtocol(manifest_path=str(manifest_path), samples=3))
    with pytest.raises(RuntimeError, match="only"):
        select_benchmark_samples(BenchmarkProtocol(manifest_path=str(manifest_path), samples=99))


def test_run_benchmark_scores_checkpoint_and_writes_record(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    _write_diffusion_checkpoint(checkpoint_path)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=2,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
        generation_steps=2,
        seed=7,
    )
    options = BenchmarkRunOptions(
        checkpoint_path=str(checkpoint_path),
        experiment="unit test",
        notes="fixture run",
        device="cpu",
        git_commit="abc1234",
    )

    record = run_benchmark(_spec(protocol), options)

    assert record["record_version"] == 1
    assert record["benchmark"] == "unit-benchmark"
    assert record["eval_hash"] == benchmark_eval_hash(protocol)
    assert record["experiment"] == "unit test"
    assert record["git_commit"] == "abc1234"
    assert record["generation"]["sample_count"] == 2
    assert record["generation"]["status"] == "computed"
    assert 0.0 <= record["generation"]["ssim"] <= 1.0
    assert record["mask_adherence"] == {
        "enabled": False,
        "status": "disabled: protocol has no segmentation_checkpoint_path",
    }
    assert "real_adherence_cache" not in record
    assert record["checkpoint"]["model_parameters"] > 0
    assert record["checkpoint"]["training_config"] is None
    assert record["data"]["eval_samples"] == 2
    assert record["data"]["sample_digest"]
    assert "unit-test" in record["run_id"]

    runs_dir = tmp_path / "runs"
    record_path = write_run_record(record, runs_dir)
    assert json.loads(record_path.read_text(encoding="utf-8"))["run_id"] == record["run_id"]
    with pytest.raises(RuntimeError, match="already exists"):
        write_run_record(record, runs_dir)


def test_run_benchmark_scores_mask_adherence_with_frozen_segmenter(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    _write_diffusion_checkpoint(checkpoint_path)
    segmentation_checkpoint_path = _write_segmentation_checkpoint(tmp_path)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=2,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
        generation_steps=2,
        segmentation_checkpoint_path=str(segmentation_checkpoint_path),
    )
    options = BenchmarkRunOptions(
        checkpoint_path=str(checkpoint_path),
        experiment="seg-judge",
        device="cpu",
    )

    record = run_benchmark(_spec(protocol), options)

    adherence = record["mask_adherence"]
    assert adherence["enabled"] is True
    assert adherence["real_scores_from_cache"] is False
    assert 0.0 <= adherence["generated_mean_iou"] <= 1.0
    assert 0.0 <= adherence["real_mean_iou"] <= 1.0
    assert 0.0 <= adherence["generated_pixel_accuracy"] <= 1.0
    assert set(adherence["generated_class_iou"]) == set(OPENEARTHMAP_CLASSES)
    cache = record["real_adherence_cache"]
    assert cache["cache_version"] == 1
    assert cache["eval_hash"] == record["eval_hash"]
    assert cache["sample_digest"] == record["data"]["sample_digest"]
    assert cache["segmentation_checkpoint_sha256"]
    assert len(cache["confusion"]) == len(OPENEARTHMAP_CLASSES)

    cached_record = run_benchmark(_spec(protocol), options, real_adherence_cache=cache)
    cached_adherence = cached_record["mask_adherence"]
    assert cached_adherence["real_scores_from_cache"] is True
    assert cached_adherence["real_mean_iou"] == pytest.approx(adherence["real_mean_iou"])
    assert cached_adherence["generated_mean_iou"] == pytest.approx(adherence["generated_mean_iou"])
    assert cached_record["real_adherence_cache"] == cache

    stale_record = run_benchmark(
        _spec(protocol),
        options,
        real_adherence_cache={**cache, "sample_digest": "stale"},
    )
    assert stale_record["mask_adherence"]["real_scores_from_cache"] is False

    disabled = run_benchmark(
        _spec(protocol),
        BenchmarkRunOptions(
            checkpoint_path=str(checkpoint_path),
            experiment="seg-judge-off",
            device="cpu",
            mask_adherence=False,
        ),
    )
    assert disabled["mask_adherence"]["enabled"] is False
    assert "real_adherence_cache" not in disabled


def test_run_benchmark_results_do_not_depend_on_batch_size(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    _write_diffusion_checkpoint(checkpoint_path)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=2,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
        generation_steps=2,
        seed=7,
    )
    options = BenchmarkRunOptions(
        checkpoint_path=str(checkpoint_path),
        experiment="batch-invariance",
        device="cpu",
    )

    batched = run_benchmark(_spec(protocol), options)
    single = run_benchmark(_spec(replace(protocol, batch_size=1)), options)

    assert batched["eval_hash"] == single["eval_hash"]
    assert single["generation"]["mse"] == pytest.approx(batched["generation"]["mse"])
    assert single["generation"]["ssim"] == pytest.approx(batched["generation"]["ssim"])


def test_real_adherence_cache_roundtrips_and_ignores_invalid_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    payload = {
        "cache_version": 1,
        "kind": "real-mask-adherence",
        "eval_hash": "a" * 64,
        "sample_digest": "b" * 64,
        "segmentation_checkpoint_sha256": "c" * 64,
        "confusion": [[0] * len(OPENEARTHMAP_CLASSES)] * len(OPENEARTHMAP_CLASSES),
    }

    path = write_real_adherence_cache(cache_dir, payload)
    assert path == real_adherence_cache_path(cache_dir, "a" * 64)
    assert load_real_adherence_cache(cache_dir, "a" * 64) == payload
    assert load_real_adherence_cache(cache_dir, "d" * 64) is None

    path.write_text("{broken", encoding="utf-8")
    assert load_real_adherence_cache(cache_dir, "a" * 64) is None

    with pytest.raises(ValueError, match="eval_hash"):
        write_real_adherence_cache(cache_dir, {"confusion": []})


def test_run_benchmark_rejects_protocol_incompatible_checkpoints(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    _write_diffusion_checkpoint(checkpoint_path, image_size=40)
    protocol = BenchmarkProtocol(
        manifest_path=str(manifest_path),
        samples=2,
        batch_size=2,
        image_size=EVAL_IMAGE_SIZE,
        generation_steps=2,
    )

    with pytest.raises(ValueError, match="image_size"):
        run_benchmark(
            _spec(protocol),
            BenchmarkRunOptions(
                checkpoint_path=str(checkpoint_path),
                experiment="mismatch",
                device="cpu",
            ),
        )
