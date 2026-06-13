from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite import modal_app  # noqa: E402
from sim_satellite.infer import InferenceConfig  # noqa: E402
from sim_satellite.train import TrainingConfig  # noqa: E402


def test_modal_training_rejects_relative_manifest_outside_data_before_app_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    fake_app = _FakeApp()
    fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

    with pytest.raises(ValueError, match="manifest_path.*local data/ cache"):
        modal_app.run_training_target(
            TrainingConfig(manifest_path="manifest.jsonl"),
            target="modal",
        )

    assert fake_app.run_calls == 0
    assert fake_app.enter_calls == 0
    assert fake_train_remote.received is None


def test_modal_training_rejects_missing_manifest_before_app_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "openearthmap").mkdir(parents=True)
    fake_app = _FakeApp()
    fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

    with pytest.raises(FileNotFoundError, match="manifest_path not found locally"):
        modal_app.run_training_target(
            TrainingConfig(manifest_path="data/openearthmap/missing.jsonl"),
            target="modal",
        )

    assert fake_app.run_calls == 0
    assert fake_app.enter_calls == 0
    assert fake_train_remote.received is None


def test_modal_inference_rejects_absolute_mask_outside_data_before_app_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_data_file(tmp_path, "openearthmap/manifest.jsonl")
    outside_mask_path = tmp_path / "mask.png"
    outside_mask_path.write_bytes(b"mask")
    fake_app = _FakeApp()
    fake_infer_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, infer_remote=fake_infer_remote)

    with pytest.raises(ValueError, match="mask_path.*local data/ cache"):
        modal_app.run_inference_target(
            InferenceConfig(
                manifest_path=str(manifest_path),
                mask_path=str(outside_mask_path),
            ),
            target="modal",
        )

    assert fake_app.run_calls == 0
    assert fake_app.enter_calls == 0
    assert fake_infer_remote.received is None


def test_modal_training_enters_app_run_with_valid_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_data_file(tmp_path, "openearthmap/manifest.jsonl")
    fake_app = _FakeApp()
    fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

    result = modal_app.run_training_target(
        TrainingConfig(
            manifest_path="data/openearthmap/manifest.jsonl",
            checkpoint_path="artifacts/checkpoints/model.pt",
        ),
        target="modal",
    )

    assert result == {"execution_target": "modal"}
    assert fake_app.run_calls == 1
    assert fake_app.enter_calls == 1
    assert fake_app.exit_calls == 1
    assert fake_train_remote.received == {
        **TrainingConfig(
            manifest_path="data/openearthmap/manifest.jsonl",
            checkpoint_path="artifacts/checkpoints/model.pt",
        ).to_dict(),
        "device": "cuda",
        "checkpoint_path": "/artifacts/checkpoints/model.pt",
        "manifest_path": "/data/openearthmap/manifest.jsonl",
    }


def test_modal_inference_enters_app_run_with_valid_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_data_file(tmp_path, "openearthmap/manifest.jsonl")
    _write_data_file(tmp_path, "openearthmap/extracted/train/labels/tile.png", content=b"mask")
    fake_app = _FakeApp()
    fake_infer_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, infer_remote=fake_infer_remote)

    result = modal_app.run_inference_target(
        InferenceConfig(
            checkpoint_path="artifacts/checkpoints/model.pt",
            output_path="artifacts/generated/sample.pt",
            manifest_path=str(manifest_path),
            mask_path="data/openearthmap/extracted/train/labels/tile.png",
        ),
        target="modal",
    )

    assert result == {"execution_target": "modal"}
    assert fake_app.run_calls == 1
    assert fake_app.enter_calls == 1
    assert fake_app.exit_calls == 1
    assert fake_infer_remote.received == {
        **InferenceConfig(
            checkpoint_path="artifacts/checkpoints/model.pt",
            output_path="artifacts/generated/sample.pt",
            manifest_path=str(manifest_path),
            mask_path="data/openearthmap/extracted/train/labels/tile.png",
        ).to_dict(),
        "device": "cuda",
        "checkpoint_path": "/artifacts/checkpoints/model.pt",
        "output_path": "/artifacts/generated/sample.pt",
        "manifest_path": "/data/openearthmap/manifest.jsonl",
        "mask_path": "/data/openearthmap/extracted/train/labels/tile.png",
    }


def test_modal_data_mount_ignores_raw_archives() -> None:
    assert modal_app._ignore_modal_data_path(Path("data/openearthmap/raw/OpenEarthMap.zip"))
    assert not modal_app._ignore_modal_data_path(
        Path("data/openearthmap/extracted/train/images/tile.tif")
    )


def _install_modal_fakes(
    monkeypatch: pytest.MonkeyPatch,
    fake_app: _FakeApp,
    *,
    train_remote: object | None = None,
    infer_remote: object | None = None,
) -> None:
    train_remote = train_remote if train_remote is not None else object()
    infer_remote = infer_remote if infer_remote is not None else object()
    monkeypatch.setattr(modal_app, "app", fake_app)
    monkeypatch.setattr(modal_app, "train_remote", train_remote)
    monkeypatch.setattr(modal_app, "infer_remote", infer_remote)


def _write_data_file(tmp_path: Path, relative_path: str, *, content: bytes = b"") -> Path:
    path = tmp_path / "data" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class _FakeApp:
    def __init__(self) -> None:
        self.run_calls = 0
        self.enter_calls = 0
        self.exit_calls = 0

    def run(self) -> _FakeApp:
        self.run_calls += 1
        return self

    def __enter__(self) -> _FakeApp:
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.exit_calls += 1


class _FakeRemote:
    def __init__(self, app: _FakeApp, result: dict[str, str]) -> None:
        self.app = app
        self.result = result
        self.received: dict[str, object] | None = None

    def remote(self, config: dict[str, object]) -> dict[str, str]:
        assert self.app.enter_calls == 1
        assert self.app.exit_calls == 0
        self.received = config
        return self.result
