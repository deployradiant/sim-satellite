from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import sim_satellite.train as train_module
from sim_satellite.data.loader import SatelliteSample
from sim_satellite.train import (
    TIMESTEP_LOGIT_NORMAL_MEAN,
    TIMESTEP_LOGIT_NORMAL_STD,
    TrainingConfig,
    _load_training_samples,
    _load_validation_batches,
    _random_square_crop_box,
    _run_timing_payload,
    _sample_flow_timesteps,
)


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


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(steps=0)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        TrainingConfig(save_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(plot_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(validation_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(validation_batches=0)


def test_run_timing_payload_estimates_remaining_runtime() -> None:
    payload = _run_timing_payload(
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


def test_flow_timesteps_use_logit_normal_sampling() -> None:
    actual_generator = torch.Generator(device="cpu").manual_seed(123)
    expected_generator = torch.Generator(device="cpu").manual_seed(123)

    actual = _sample_flow_timesteps(
        8,
        torch=torch,
        generator=actual_generator,
        device=torch.device("cpu"),
    )
    logits = torch.randn(8, generator=expected_generator, device="cpu")
    expected = torch.sigmoid(logits * TIMESTEP_LOGIT_NORMAL_STD + TIMESTEP_LOGIT_NORMAL_MEAN)

    assert torch.allclose(actual, expected)
    assert torch.all(actual > 0.0)
    assert torch.all(actual < 1.0)


def test_validation_holdout_patterns_are_excluded_from_training(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/maputo/images/c.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/maputo/labels/c.tif", '
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
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
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
        '{"image": "images/a.tif", "mask": "labels/a.tif", "split": "train"}\n',
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
    Image.new("RGB", (10, 8)).save(image_path)
    Image.new("L", (10, 8)).save(mask_path)
    sample = SatelliteSample(image_path=image_path, mask_path=mask_path, split="train")

    crop_box = _random_square_crop_box(
        sample,
        crop_size=4,
        torch=_FakeTorch([3, 2]),
        generator=object(),
    )

    assert crop_box == (3, 2, 7, 6)


class _FakeTorch:
    def __init__(self, values: list[int]) -> None:
        self.values = values

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


class _FakeTensor:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value
