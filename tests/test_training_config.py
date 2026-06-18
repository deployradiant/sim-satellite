from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import sim_satellite.train as train_module
from sim_satellite import training_log
from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.train import (
    TIMESTEP_LOGIT_NORMAL_MEAN,
    TIMESTEP_LOGIT_NORMAL_STD,
    TIMESTEP_UNIFORM_MIX_PROBABILITY,
    _autocast_context,
    _load_training_samples,
    _load_validation_batches,
    _sample_flow_timesteps,
    _training_device,
)
from sim_satellite.training_config import TrainingConfig
from sim_satellite.training_data import preload_samples, random_square_crop_box


def test_training_cli_loads_json_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps({"steps": 2, "batch_size": 3, "manifest_path": "manifest.jsonl"}),
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    def fake_run_with_target(config: TrainingConfig, target: str) -> dict[str, object]:
        received["config"] = config
        received["target"] = target
        return {"execution_target": target, "steps": config.steps}

    monkeypatch.setattr(train_module, "_run_with_target", fake_run_with_target)

    result = CliRunner().invoke(train_module.cli, ["--target", "local", str(config_path)])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"execution_target": "local", "steps": 2}
    assert received["target"] == "local"
    assert received["config"] == TrainingConfig(
        steps=2,
        batch_size=3,
        manifest_path="manifest.jsonl",
    )


def test_training_cli_has_no_per_field_training_options() -> None:
    result = CliRunner().invoke(train_module.cli, ["--help"])

    assert result.exit_code == 0
    assert "CONFIG_PATH" in result.output
    assert "--target" in result.output
    assert "--steps" not in result.output
    assert "--batch-size" not in result.output


def test_training_config_rejects_unknown_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    with pytest.raises(ValueError, match="Unknown training config fields: future_field"):
        TrainingConfig.from_dict({**asdict(config), "future_field": "rejected"})


def test_training_device_requires_available_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
        _training_device(TrainingConfig(device="cuda"), torch=_FakeTorch(cuda_available=False))


def test_bf16_autocast_is_cuda_only() -> None:
    fake_torch = _FakeTorch(cuda_available=True)

    context = _autocast_context(fake_torch, "bf16")

    assert context == "autocast-context"
    assert fake_torch.autocast_calls == [("cuda", fake_torch.bfloat16)]


def test_run_timing_payload_estimates_remaining_runtime() -> None:
    payload = training_log.run_timing_payload(
        started_at_epoch_utc=1_000.0,
        started_at_monotonic=10.0,
        start_step=100,
        step=220,
        steps=300,
        batch_size=10,
        now_epoch_utc=1_060.0,
        now_monotonic=70.0,
    )

    assert payload["elapsed_seconds"] == 60.0
    assert payload["completed_steps_this_run"] == 120
    assert payload["completed_images_this_run"] == 1_200
    assert payload["remaining_steps"] == 80
    assert payload["remaining_images"] == 800
    assert payload["steps_per_second"] == 2.0
    assert payload["images_per_second"] == 20.0
    assert payload["estimated_remaining_seconds"] == 40.0
    assert payload["estimated_total_seconds"] == 100.0
    assert payload["estimated_finish_epoch_utc"] == 1_100.0


def test_flow_timesteps_use_seeded_uniform_logit_mixture_by_default() -> None:
    actual_generator = torch.Generator(device="cpu").manual_seed(321)
    expected_generator = torch.Generator(device="cpu").manual_seed(321)

    actual = _sample_flow_timesteps(
        16,
        torch=torch,
        generator=actual_generator,
        device=torch.device("cpu"),
    )
    logits = torch.randn(16, generator=expected_generator, device="cpu")
    logit_normal = torch.sigmoid(logits * TIMESTEP_LOGIT_NORMAL_STD + TIMESTEP_LOGIT_NORMAL_MEAN)
    uniform = torch.rand(16, generator=expected_generator, device="cpu")
    use_uniform = (
        torch.rand(16, generator=expected_generator, device="cpu")
        < TIMESTEP_UNIFORM_MIX_PROBABILITY
    )
    expected = torch.where(use_uniform, uniform, logit_normal)

    assert torch.allclose(actual, expected)
    assert torch.any(use_uniform)
    assert torch.any(~use_uniform)
    assert torch.all(actual >= 0.0)
    assert torch.all(actual <= 1.0)


def test_uniform_timestep_mix_increases_endpoint_tail_mass() -> None:
    logit_generator = torch.Generator(device="cpu").manual_seed(456)
    mixed_generator = torch.Generator(device="cpu").manual_seed(456)

    logits = torch.randn(4096, generator=logit_generator, device="cpu")
    logit_normal = torch.sigmoid(logits * TIMESTEP_LOGIT_NORMAL_STD + TIMESTEP_LOGIT_NORMAL_MEAN)
    mixed = _sample_flow_timesteps(
        4096,
        torch=torch,
        generator=mixed_generator,
        device=torch.device("cpu"),
    )

    logit_tail_mass = ((logit_normal < 0.1) | (logit_normal > 0.9)).float().mean()
    mixed_tail_mass = ((mixed < 0.1) | (mixed > 0.9)).float().mean()

    assert logit_tail_mass < 0.05
    assert mixed_tail_mass > 0.08
    assert mixed_tail_mass > logit_tail_mass * 2.0


def test_validation_holdout_patterns_are_excluded_from_training(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"prepared": "prepared/abancay/a.pt", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"prepared": "prepared/tokyo/b.pt", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/maputo/images/c.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/maputo/labels/c.tif", '
                '"prepared": "prepared/maputo/c.pt", '
                '"split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = _load_training_samples(
        TrainingConfig(
            manifest_path=str(manifest_path),
            validation_holdout_patterns="abancay, maputo",
        )
    )

    assert [sample.image_path.name for sample in samples] == ["b.tif"]


def test_validation_batches_use_holdout_patterns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"prepared": "prepared/abancay/a.pt", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"prepared": "prepared/tokyo/b.pt", '
                '"split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    batches = _load_validation_batches(
        TrainingConfig(
            manifest_path=str(manifest_path),
            validation_every=10,
            validation_holdout_patterns="abancay",
            batch_size=1,
        )
    )

    assert len(batches) == 1
    assert batches[0][0].image_path.name == "a.tif"


def test_validation_requires_split_or_holdout_patterns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"image": "images/a.tif", "mask": "labels/a.tif", '
        '"prepared": "prepared/a.pt", "split": "train"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation_split or validation_holdout_patterns"):
        _load_validation_batches(
            TrainingConfig(manifest_path=str(manifest_path), validation_every=1)
        )


def test_random_square_crop_box_is_aligned_and_deterministic(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    prepared_path = tmp_path / "prepared.pt"
    Image.new("RGB", (10, 8)).save(image_path)
    Image.new("L", (10, 8)).save(mask_path)
    torch.save(
        {
            "version": PREPARED_DATA_VERSION,
            "image": torch.zeros(8, 10, 3, dtype=torch.uint8),
            "mask": torch.zeros(8, 10, dtype=torch.uint8),
            "size": (10, 8),
            "coordinates": None,
        },
        prepared_path,
    )
    sample = SatelliteSample(
        image_path=image_path,
        mask_path=mask_path,
        prepared_path=prepared_path,
        split="train",
    )
    preloaded = preload_samples((sample,))

    crop_box = random_square_crop_box(
        sample,
        crop_size=4,
        torch=_FakeTorch([3, 2]),
        generator=object(),
        preloaded_samples=preloaded,
    )

    assert crop_box == (3, 2, 7, 6)


class _FakeTorch:
    bfloat16 = object()

    def __init__(
        self,
        values: list[int] | None = None,
        *,
        cuda_available: bool = True,
    ) -> None:
        self.values = values or []
        self.cuda = _FakeCuda(cuda_available)
        self.autocast_calls: list[tuple[str, object]] = []

    def device(self, value: str) -> torch.device:
        return torch.device(value)

    def autocast(self, *, device_type: str, dtype: object) -> str:
        self.autocast_calls.append((device_type, dtype))
        return "autocast-context"

    def randint(
        self,
        low: int,
        high: int,
        shape: tuple[int],
        *,
        generator: object,
        device: str,
    ) -> _FakeTensor:
        assert low == 0
        assert shape == (1,)
        assert device == "cpu"
        value = self.values.pop(0)
        assert value < high
        return _FakeTensor(value)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTensor:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value
