from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from sim_satellite import infer
from sim_satellite.infer import GenerationConfig, InferenceConfig, InferenceResult
from sim_satellite.model import ModelConfig, build_model


def test_inference_config_rejects_old_model_shape_fields() -> None:
    with pytest.raises(ValueError, match="Unknown inference config fields: image_size"):
        InferenceConfig.from_dict({**asdict(InferenceConfig()), "image_size": 64})


def test_inference_defaults_use_more_sampler_steps() -> None:
    assert InferenceConfig().steps == 16
    assert GenerationConfig(mask_path="mask.png").steps == 16


def test_flow_inference_schedule_uses_midpoint_timesteps() -> None:
    schedule = infer._flow_inference_schedule(4)

    assert [timestep for timestep, _ in schedule] == pytest.approx(
        [0.875, 0.625, 0.375, 0.125]
    )
    assert [step_size for _, step_size in schedule] == pytest.approx([0.25] * 4)


def test_run_inference_requires_explicit_mask_path() -> None:
    with pytest.raises(ValueError, match="mask_path is required"):
        infer.run_inference(InferenceConfig())


def test_cli_target_resolves_default_manifest_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(b"mask")
    manifest_path.write_text(
        '{"image": "image.png", "mask": "mask.png", '
        '"prepared": "prepared/sample.pt", "split": "train"}\n',
        encoding="utf-8",
    )
    received: dict[str, InferenceConfig] = {}

    def fake_run_inference(config: InferenceConfig) -> InferenceResult:
        received["config"] = config
        return InferenceResult(
            output_path="",
            checkpoint_path="",
            mask_path="",
            device="",
        )

    monkeypatch.setattr(infer, "run_inference", fake_run_inference)

    result = infer._run_with_target(
        InferenceConfig(manifest_path=str(manifest_path)),
        target="local",
    )

    assert result == {
        "output_path": "",
        "checkpoint_path": "",
        "mask_path": "",
        "device": "",
        "execution_target": "local",
    }
    assert received["config"].mask_path == str(mask_path)


def test_inference_rejects_malformed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    config = InferenceConfig(
        checkpoint_path=str(checkpoint_path),
        output_path=str(tmp_path / "sample.pt"),
        mask_path=str(tmp_path / "mask.png"),
    )
    cases = (
        ({"model_state": {}}, "missing model_config"),
        ({"model_config": {}}, "missing model_state"),
        (["not", "a", "mapping"], "checkpoint must be a mapping"),
    )

    for checkpoint, message in cases:

        def fake_load(*_: Any, checkpoint: Any = checkpoint, **__: Any) -> Any:
            return checkpoint

        monkeypatch.setattr(infer.torch, "load", fake_load)
        with pytest.raises(RuntimeError, match=message):
            infer.run_inference(config)


def test_load_generator_accepts_flow_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(tmp_path, torch=torch)

    loaded = infer.load_generator(checkpoint_path, device="cpu")

    assert loaded.model_config == ModelConfig(
        image_size=32,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
    )


def test_load_generator_rejects_incompatible_flow_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(tmp_path, torch=torch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state"]["input_conv.weight"] = torch.zeros(4, 99, 3, 3)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="incompatible with flow U-Net"):
        infer.load_generator(checkpoint_path, device="cpu")


def _write_flow_checkpoint(
    tmp_path: Path,
    *,
    torch: Any,
) -> Path:
    checkpoint_path = tmp_path / "flow.pt"
    config = ModelConfig(
        image_size=32,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
    )
    model = build_model(config)
    checkpoint = {
        "model_config": asdict(config),
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
