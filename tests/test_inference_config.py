from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from sim_satellite import infer
from sim_satellite.infer import InferenceConfig, InferenceResult
from sim_satellite.model import ModelConfig, build_model


def test_inference_config_rejects_old_model_shape_fields() -> None:
    with pytest.raises(ValueError, match="Unknown inference config fields: image_size"):
        InferenceConfig.from_dict({**asdict(InferenceConfig()), "image_size": 64})


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


def test_load_generator_accepts_supported_checkpoint_formats(tmp_path: Path) -> None:
    expected_coordinate_config = ModelConfig(
        image_size=32,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
    )
    expected_no_coordinate_config = ModelConfig(
        image_size=32,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
        use_coordinates=False,
    )
    cases = (
        (
            _write_flow_checkpoint(tmp_path, torch=torch, filename="flow.pt"),
            expected_coordinate_config,
        ),
        (
            _write_flow_checkpoint(
                tmp_path,
                torch=torch,
                filename="flow-no-coordinates.pt",
                use_coordinates=False,
            ),
            expected_no_coordinate_config,
        ),
    )

    for checkpoint_path, expected_config in cases:
        loaded = infer.load_generator(checkpoint_path, device="cpu")

        assert loaded.model_config == expected_config


def test_load_generator_rejects_incompatible_flow_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(tmp_path, torch=torch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state"]["input_conv.weight"] = torch.zeros(4, 99, 3, 3)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="incompatible with flow U-Net"):
        infer.load_generator(checkpoint_path, device="cpu")


def test_load_generator_rejects_missing_coordinate_projection(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(tmp_path, torch=torch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state"] = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if not key.startswith("coordinate_projection.")
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing coordinate_projection"):
        infer.load_generator(checkpoint_path, device="cpu")


def test_load_generator_rejects_unexpected_coordinate_projection(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(
        tmp_path,
        torch=torch,
        use_coordinates=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state"]["coordinate_projection.0.weight"] = torch.zeros(4, 3)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="contains coordinate_projection"):
        infer.load_generator(checkpoint_path, device="cpu")


def test_generate_image_supports_inpaint_checkpoint_condition(tmp_path: Path) -> None:
    checkpoint_path = _write_flow_checkpoint(
        tmp_path,
        torch=torch,
        filename="inpaint-flow.pt",
        use_coordinates=False,
        use_inpaint_mask=True,
    )
    mask_path = tmp_path / "mask.png"
    Image.new("L", (32, 32), color=1).save(mask_path)

    loaded = infer.load_generator(checkpoint_path, device="cpu")
    result = infer.generate_image_with_model(
        loaded,
        infer.GenerationConfig(
            mask_path=str(mask_path),
            steps=1,
            guidance_scale=2.0,
        ),
    )

    assert result.image.shape == (1, 3, 32, 32)
    assert result.semantic_mask.shape == (1, 2, 32, 32)
    assert torch.isfinite(result.image).all()


def test_empty_guidance_condition_preserves_inpaint_channel() -> None:
    config = ModelConfig(mask_channels=2, use_inpaint_mask=True)
    model_condition = torch.zeros(1, 3, 2, 2)
    model_condition[:, 0] = 1.0
    model_condition[:, -1] = 1.0

    empty_condition = infer._empty_guidance_condition(
        model_condition,
        config,
        torch=torch,
    )

    assert torch.equal(empty_condition[:, : config.mask_channels], torch.zeros(1, 2, 2, 2))
    assert torch.equal(empty_condition[:, config.mask_channels :], torch.ones(1, 1, 2, 2))


def test_flow_inference_tensors_match_schedule() -> None:
    timesteps, step_size = infer._flow_inference_tensors(4, torch=torch, device="cpu")

    assert timesteps.shape == (4, 1)
    assert step_size == 0.25
    assert torch.allclose(
        timesteps.flatten(),
        torch.tensor([0.875, 0.625, 0.375, 0.125]),
    )
    assert infer._flow_inference_schedule(4) == (
        (0.875, 0.25),
        (0.625, 0.25),
        (0.375, 0.25),
        (0.125, 0.25),
    )


def _write_flow_checkpoint(
    tmp_path: Path,
    *,
    torch: Any,
    filename: str = "flow.pt",
    use_coordinates: bool = True,
    use_inpaint_mask: bool = False,
) -> Path:
    checkpoint_path = tmp_path / filename
    config = ModelConfig(
        image_size=32,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
        use_coordinates=use_coordinates,
        use_inpaint_mask=use_inpaint_mask,
    )
    model = build_model(config)
    checkpoint = {
        "model_config": asdict(config),
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
