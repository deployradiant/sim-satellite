from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite import infer  # noqa: E402
from sim_satellite.infer import DEFAULT_TARGET, InferenceConfig  # noqa: E402


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


class _FakeTorch:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint = checkpoint

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def device(self, name: str) -> str:
        return name

    def load(self, path: Path, map_location: str) -> dict[str, Any]:
        self.loaded_path = path
        self.map_location = map_location
        return self.checkpoint
