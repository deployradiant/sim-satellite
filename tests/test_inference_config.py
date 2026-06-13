from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite import infer  # noqa: E402
from sim_satellite.infer import DEFAULT_TARGET, GenerationConfig, InferenceConfig  # noqa: E402
from sim_satellite.model import ModelConfig  # noqa: E402


def test_inference_config_defaults_are_deterministic() -> None:
    config = InferenceConfig()

    assert DEFAULT_TARGET == "modal"
    assert config.seed == 1337
    assert config.checkpoint_path == "artifacts/tiny-sim-satellite.pt"
    assert config.output_path == "artifacts/sample.pt"
    assert config.manifest_path == "data/openearthmap/manifest.jsonl"
    assert config.split == "train"
    assert config.mask_path is None
    assert config.device == "cpu"
    assert config.steps == 8


def test_inference_config_round_trips_known_fields() -> None:
    config = InferenceConfig(steps=3, mask_path="data/mask.png")

    assert InferenceConfig.from_dict(config.to_dict()) == config


def test_inference_config_rejects_unknown_fields() -> None:
    config = InferenceConfig(steps=3)

    with pytest.raises(ValueError, match="Unknown inference config fields: future_field"):
        InferenceConfig.from_dict({**config.to_dict(), "future_field": "rejected"})


def test_inference_config_rejects_old_model_shape_fields() -> None:
    with pytest.raises(ValueError, match="Unknown inference config fields: image_size"):
        InferenceConfig.from_dict({**InferenceConfig().to_dict(), "image_size": 64})


def test_generation_config_rejects_output_path() -> None:
    with pytest.raises(ValueError, match="Unknown generation config fields: output_path"):
        GenerationConfig.from_dict(
            {
                "checkpoint_path": "artifacts/model.pt",
                "mask_path": "data/mask.png",
                "output_path": "artifacts/sample.pt",
            }
        )


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
        '{"image": "image.png", "mask": "mask.png", "split": "train"}\n',
        encoding="utf-8",
    )
    received: dict[str, InferenceConfig] = {}

    def fake_run_inference(config: InferenceConfig) -> SimpleNamespace:
        received["config"] = config
        return SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(infer, "run_inference", fake_run_inference)

    result = infer._run_with_target(
        InferenceConfig(manifest_path=str(manifest_path)),
        target="local",
    )

    assert result == {"execution_target": "local"}
    assert received["config"].mask_path == str(mask_path)


def test_inference_requires_model_config_in_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    fake_torch = _FakeTorch({"model_state": {}})
    monkeypatch.setattr(infer, "require_torch", lambda: (fake_torch, object()))

    config = InferenceConfig(
        checkpoint_path=str(checkpoint_path),
        output_path=str(tmp_path / "sample.pt"),
        mask_path=str(tmp_path / "mask.png"),
    )

    with pytest.raises(RuntimeError, match="missing model_config"):
        infer.run_inference(config)


def test_inference_requires_model_state_in_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    fake_torch = _FakeTorch({"model_config": {}})
    monkeypatch.setattr(infer, "require_torch", lambda: (fake_torch, object()))

    config = InferenceConfig(
        checkpoint_path=str(checkpoint_path),
        output_path=str(tmp_path / "sample.pt"),
        mask_path=str(tmp_path / "mask.png"),
    )

    with pytest.raises(RuntimeError, match="missing model_state"):
        infer.run_inference(config)


def test_inference_requires_checkpoint_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    fake_torch = _FakeTorch(["not", "a", "mapping"])
    monkeypatch.setattr(infer, "require_torch", lambda: (fake_torch, object()))

    config = InferenceConfig(
        checkpoint_path=str(checkpoint_path),
        output_path=str(tmp_path / "sample.pt"),
        mask_path=str(tmp_path / "mask.png"),
    )

    with pytest.raises(RuntimeError, match="checkpoint must be a mapping"):
        infer.run_inference(config)


def test_load_generator_accepts_legacy_tiny_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = _write_legacy_tiny_checkpoint(tmp_path, torch=torch)

    loaded = infer.load_generator(checkpoint_path, device="cpu")

    assert loaded.architecture == "legacy tiny denoiser"
    assert loaded.legacy_timesteps == 16
    assert loaded.model_config == ModelConfig(
        image_size=8,
        image_channels=3,
        mask_channels=2,
        hidden_channels=4,
        seed=99,
    )


def test_generate_image_supports_legacy_tiny_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = _write_legacy_tiny_checkpoint(tmp_path, torch=torch)
    mask_path = tmp_path / "mask.png"
    Image.new("L", (8, 8), 0).save(mask_path)

    result = infer.generate_image(
        GenerationConfig(
            checkpoint_path=str(checkpoint_path),
            mask_path=str(mask_path),
            steps=1,
        )
    )

    assert result.image.shape == (1, 3, 8, 8)
    assert result.model_config.mask_channels == 2


def test_load_generator_rejects_incompatible_legacy_tiny_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = _write_legacy_tiny_checkpoint(
        tmp_path,
        torch=torch,
        first_input_channels=5,
    )

    with pytest.raises(RuntimeError, match="incompatible with legacy tiny denoiser"):
        infer.load_generator(checkpoint_path, device="cpu")


class _FakeTorch:
    def __init__(self, checkpoint: Any) -> None:
        self.checkpoint = checkpoint

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def device(self, name: str) -> str:
        return name

    def load(self, path: Path, map_location: str) -> dict[str, Any]:
        self.loaded_path = path
        self.map_location = map_location
        return self.checkpoint


def _write_legacy_tiny_checkpoint(
    tmp_path: Path,
    *,
    torch: Any,
    first_input_channels: int = 6,
) -> Path:
    checkpoint_path = tmp_path / "legacy.pt"
    hidden_channels = 4
    checkpoint = {
        "model_config": {
            "image_size": 8,
            "image_channels": 3,
            "mask_channels": 2,
            "hidden_channels": hidden_channels,
            "timesteps": 16,
            "seed": 99,
        },
        "model_state": {
            "net.0.weight": torch.zeros(hidden_channels, first_input_channels, 3, 3),
            "net.0.bias": torch.zeros(hidden_channels),
            "net.2.weight": torch.zeros(hidden_channels, hidden_channels, 3, 3),
            "net.2.bias": torch.zeros(hidden_channels),
            "net.4.weight": torch.zeros(3, hidden_channels, 3, 3),
            "net.4.bias": torch.zeros(3),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
