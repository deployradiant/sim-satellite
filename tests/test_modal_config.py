from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite import modal_app  # noqa: E402
from sim_satellite.infer import InferenceConfig  # noqa: E402
from sim_satellite.modal_app import (  # noqa: E402
    _modal_inference_config,
    _modal_training_config,
)
from sim_satellite.train import TrainingConfig  # noqa: E402


def test_modal_training_config_maps_default_paths_to_remote_mounts() -> None:
    values = _modal_training_config(TrainingConfig())

    assert values["checkpoint_path"] == "/artifacts/tiny-sim-satellite.pt"
    assert values["manifest_path"] == "/data/openearthmap/manifest.jsonl"


def test_modal_inference_config_maps_default_paths_to_remote_mounts() -> None:
    values = _modal_inference_config(
        InferenceConfig(mask_path="data/openearthmap/extracted/train/labels/tile.png")
    )

    assert values["checkpoint_path"] == "/artifacts/tiny-sim-satellite.pt"
    assert values["output_path"] == "/artifacts/sample.pt"
    assert values["manifest_path"] == "/data/openearthmap/manifest.jsonl"
    assert values["mask_path"] == "/data/openearthmap/extracted/train/labels/tile.png"


def test_modal_training_target_runs_inside_app_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})

    monkeypatch.setattr(modal_app, "app", fake_app)
    monkeypatch.setattr(modal_app, "train_remote", fake_train_remote)
    monkeypatch.setattr(modal_app, "infer_remote", object())

    result = modal_app.run_training_target(TrainingConfig(), target="modal")

    assert result == {"execution_target": "modal"}
    assert fake_app.exited
    assert fake_train_remote.received["checkpoint_path"] == "/artifacts/tiny-sim-satellite.pt"


def test_modal_inference_target_runs_inside_app_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    fake_infer_remote = _FakeRemote(fake_app, {"execution_target": "modal"})

    monkeypatch.setattr(modal_app, "app", fake_app)
    monkeypatch.setattr(modal_app, "train_remote", object())
    monkeypatch.setattr(modal_app, "infer_remote", fake_infer_remote)

    result = modal_app.run_inference_target(InferenceConfig(), target="modal")

    assert result == {"execution_target": "modal"}
    assert fake_app.exited
    assert fake_infer_remote.received["output_path"] == "/artifacts/sample.pt"


class _FakeApp:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def run(self):
        return self

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.exited = True


class _FakeRemote:
    def __init__(self, app: _FakeApp, result: dict[str, str]) -> None:
        self.app = app
        self.result = result
        self.received = None

    def remote(self, config: dict[str, object]) -> dict[str, str]:
        assert self.app.entered
        self.received = config
        return self.result
