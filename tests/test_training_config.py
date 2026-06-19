from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import sim_satellite.train as train_module
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


def test_training_cli_uses_json_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    help_result = CliRunner().invoke(train_module.cli, ["--help"])
    assert help_result.exit_code == 0
    assert "CONFIG_PATH" in help_result.output
    assert "--target" in help_result.output
    assert "--steps" not in help_result.output
    assert "--batch-size" not in help_result.output


def test_training_config_rejects_unknown_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    with pytest.raises(ValueError, match="Unknown training config fields: future_field"):
        TrainingConfig.from_dict({**asdict(config), "future_field": "rejected"})


def test_training_cuda_runtime_helpers() -> None:
    with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
        _training_device(TrainingConfig(device="cuda"), torch=_FakeTorch(cuda_available=False))

    fake_torch = _FakeTorch(cuda_available=True)
    context = _autocast_context(fake_torch, "bf16")

    assert context == "autocast-context"
    assert fake_torch.autocast_calls == [("cuda", fake_torch.bfloat16)]


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


def test_validation_holdout_patterns_split_training_and_validation(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="validation_split or validation_holdout_patterns"):
        _load_validation_batches(
            TrainingConfig(manifest_path=str(manifest_path), validation_every=1)
        )

class _FakeTorch:
    bfloat16 = object()

    def __init__(
        self,
        *,
        cuda_available: bool = True,
    ) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.autocast_calls: list[tuple[str, object]] = []

    def device(self, value: str) -> torch.device:
        return torch.device(value)

    def autocast(self, *, device_type: str, dtype: object) -> str:
        self.autocast_calls.append((device_type, dtype))
        return "autocast-context"


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available
