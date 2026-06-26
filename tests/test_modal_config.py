from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from sim_satellite import modal_app
from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.infer import InferenceConfig
from sim_satellite.training_config import TrainingConfig
from sim_satellite.web import WebConfig


def test_modal_gpu_defaults_keep_inference_and_web_on_cheapest_gpu() -> None:
    assert modal_app.DEFAULT_MODAL_TRAINING_GPU == "H100!"
    assert modal_app.DEFAULT_MODAL_INFERENCE_GPU == "T4"
    if "SIM_SATELLITE_MODAL_INFERENCE_GPU" not in os.environ:
        assert modal_app.MODAL_INFERENCE_GPU == "T4"
    if not {
        "SIM_SATELLITE_MODAL_INFERENCE_GPU",
        "SIM_SATELLITE_MODAL_WEB_GPU",
    } & set(os.environ):
        assert modal_app.MODAL_WEB_GPU == "T4"


def test_modal_training_maps_relative_manifest_to_remote_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    cases = (
        ("manifest.jsonl", "/data/manifest.jsonl"),
        ("data/openearthmap/manifest.jsonl", "/data/openearthmap/manifest.jsonl"),
    )

    for manifest_path, remote_manifest_path in cases:
        fake_app = _FakeApp()
        fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
        _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

        modal_app.run_training_target(
            TrainingConfig(manifest_path=manifest_path),
        )

        assert fake_app.run_calls == 1
        assert fake_train_remote.received is not None
        assert fake_train_remote.received["manifest_path"] == remote_manifest_path


def test_modal_training_maps_landcoverai_mix_manifest_to_remote_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_app = _FakeApp()
    fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
    _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

    modal_app.run_training_target(
        TrainingConfig(manifest_path="data/landcoverai-mix/ratio-10/manifest.jsonl"),
    )

    assert fake_train_remote.received is not None
    assert fake_train_remote.received["manifest_path"] == (
        "/data/landcoverai-mix/ratio-10/manifest.jsonl"
    )


def test_modal_training_rejects_invalid_artifact_paths_before_app_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_data_file(tmp_path, "openearthmap/manifest.jsonl")
    cases = (
        (str(tmp_path / "model.pt"), "artifact paths.*relative or under /artifacts"),
        ("artifacts/../model.pt", "artifact paths.*stay under /artifacts"),
    )

    for checkpoint_path, message in cases:
        fake_app = _FakeApp()
        fake_train_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
        _install_modal_fakes(monkeypatch, fake_app, train_remote=fake_train_remote)

        with pytest.raises(ValueError, match=message):
            modal_app.run_training_target(
                TrainingConfig(
                    manifest_path="data/openearthmap/manifest.jsonl",
                    checkpoint_path=checkpoint_path,
                ),
            )

        assert fake_app.run_calls == 0
        assert fake_app.enter_calls == 0
        assert fake_train_remote.received is None


def test_modal_inference_rejects_invalid_masks_before_app_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_data_file(tmp_path, "openearthmap/manifest.jsonl")
    outside_mask_path = tmp_path / "mask.png"
    outside_mask_path.write_bytes(b"mask")
    cases = (
        (
            InferenceConfig(
                manifest_path=str(manifest_path),
                mask_path=str(outside_mask_path),
            ),
            "mask_path.*under local data/",
        ),
        (
            InferenceConfig(manifest_path=str(manifest_path), mask_path=None),
            "mask_path is required",
        ),
    )

    for config, message in cases:
        fake_app = _FakeApp()
        fake_infer_remote = _FakeRemote(fake_app, {"execution_target": "modal"})
        _install_modal_fakes(monkeypatch, fake_app, infer_remote=fake_infer_remote)

        with pytest.raises(ValueError, match=message):
            modal_app.run_inference_target(config)

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
            resume_from="artifacts/checkpoints/model-step-50000.pt",
        )
    )

    assert result == {"execution_target": "modal"}
    assert fake_app.run_calls == 1
    assert fake_app.enter_calls == 1
    assert fake_app.exit_calls == 1
    assert fake_train_remote.received == {
        **asdict(
            TrainingConfig(
                manifest_path="data/openearthmap/manifest.jsonl",
                checkpoint_path="artifacts/checkpoints/model.pt",
                resume_from="artifacts/checkpoints/model-step-50000.pt",
            )
        ),
        "device": "cuda",
        "checkpoint_path": "/artifacts/checkpoints/model.pt",
        "manifest_path": "/data/openearthmap/manifest.jsonl",
        "resume_from": "/artifacts/checkpoints/model-step-50000.pt",
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
        )
    )

    assert result == {"execution_target": "modal"}
    assert fake_app.run_calls == 1
    assert fake_app.enter_calls == 1
    assert fake_app.exit_calls == 1
    assert fake_infer_remote.received == {
        **asdict(
            InferenceConfig(
                checkpoint_path="artifacts/checkpoints/model.pt",
                output_path="artifacts/generated/sample.pt",
                manifest_path=str(manifest_path),
                mask_path="data/openearthmap/extracted/train/labels/tile.png",
            )
        ),
        "device": "cuda",
        "checkpoint_path": "/artifacts/checkpoints/model.pt",
        "output_path": "/artifacts/generated/sample.pt",
        "manifest_path": "/data/openearthmap/manifest.jsonl",
        "mask_path": "/data/openearthmap/extracted/train/labels/tile.png",
    }


def test_modal_web_config_uses_artifact_checkpoint_and_remote_local_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_data_file(tmp_path, "openearthmap/manifest.jsonl")

    config = modal_app._modal_web_config(
        WebConfig(
            checkpoint_path="artifacts/checkpoints/web.pt",
            checkpoint_glob="artifacts/checkpoints/web-step-*.pt",
            comparison_checkpoint_path="artifacts/checkpoints/main.pt",
            comparison_checkpoint_glob="artifacts/checkpoints/main-step-*.pt",
            manifest_path=str(manifest_path),
            device="cuda",
            steps=3,
            max_samples=7,
        )
    )

    assert config.checkpoint_path == "/artifacts/checkpoints/web.pt"
    assert config.checkpoint_glob == "/artifacts/checkpoints/web-step-*.pt"
    assert config.comparison_checkpoint_path == "/artifacts/checkpoints/main.pt"
    assert config.comparison_checkpoint_glob == "/artifacts/checkpoints/main-step-*.pt"
    assert config.manifest_path == "/data/openearthmap/manifest.jsonl"
    assert config.device == "cuda"
    assert config.steps == 3
    assert config.max_samples == 7


def test_modal_data_path_maps_inside_modal_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modal_data_root = tmp_path / "data"
    modal_data_root.mkdir()
    monkeypatch.setattr(modal_app, "MODAL_DATA_ROOT", modal_data_root)

    assert modal_app._modal_data_path(
        "data/openearthmap/manifest.jsonl", field_name="manifest_path"
    ) == str(modal_data_root / "openearthmap/manifest.jsonl")
    assert modal_app._modal_data_path(
        str(modal_data_root / "openearthmap/manifest.jsonl"),
        field_name="manifest_path",
    ) == str(modal_data_root / "openearthmap/manifest.jsonl")

    with pytest.raises(ValueError, match="under local data/"):
        modal_app._modal_data_path("/tmp/manifest.jsonl", field_name="manifest_path")


def test_modal_landcoverai_mix_ratio_parses_manifest_parent() -> None:
    assert (
        modal_app._landcoverai_mix_ratio_from_manifest(
            Path("/data/landcoverai-mix/baseline/manifest.jsonl")
        )
        == 0.0
    )
    assert (
        modal_app._landcoverai_mix_ratio_from_manifest(
            Path("/data/landcoverai-mix/ratio-33/manifest.jsonl")
        )
        == 0.33
    )
    with pytest.raises(ValueError, match="baseline.*ratio-N"):
        modal_app._landcoverai_mix_ratio_from_manifest(
            Path("/data/landcoverai-mix/nope/manifest.jsonl")
        )


def test_modal_writes_deterministic_landcoverai_mix_manifest(tmp_path: Path) -> None:
    openearthmap_manifest = tmp_path / "openearthmap" / "manifest.jsonl"
    landcoverai_manifest = tmp_path / "landcoverai" / "manifest.jsonl"
    output_manifest = tmp_path / "landcoverai-mix" / "ratio-50" / "manifest.jsonl"
    _write_manifest_samples(
        openearthmap_manifest,
        source="openearthmap",
        count=10,
        split="train",
    )
    _write_manifest_samples(
        landcoverai_manifest,
        source="landcoverai",
        count=20,
        split="train",
    )

    modal_app._write_landcoverai_mix_manifest(
        output_manifest,
        openearthmap_manifest=openearthmap_manifest,
        landcoverai_manifest=landcoverai_manifest,
        landcoverai_ratio=0.5,
        validation_samples=2,
        seed=2026,
    )

    samples = read_manifest(output_manifest)
    source_counts = _source_split_counts(samples)
    assert source_counts == {
        ("landcoverai", "train"): 4,
        ("openearthmap", "train"): 8,
        ("openearthmap", "val"): 2,
    }


def test_modal_writes_baseline_mix_manifest_without_landcoverai(tmp_path: Path) -> None:
    openearthmap_manifest = tmp_path / "openearthmap" / "manifest.jsonl"
    landcoverai_manifest = tmp_path / "landcoverai" / "manifest.jsonl"
    output_manifest = tmp_path / "landcoverai-mix" / "baseline" / "manifest.jsonl"
    _write_manifest_samples(
        openearthmap_manifest,
        source="openearthmap",
        count=10,
        split="train",
    )

    modal_app._write_landcoverai_mix_manifest(
        output_manifest,
        openearthmap_manifest=openearthmap_manifest,
        landcoverai_manifest=landcoverai_manifest,
        landcoverai_ratio=0.0,
        validation_samples=2,
        seed=2026,
    )

    samples = read_manifest(output_manifest)
    assert _source_split_counts(samples) == {
        ("openearthmap", "train"): 8,
        ("openearthmap", "val"): 2,
    }


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


def _write_manifest_samples(
    manifest_path: Path,
    *,
    source: str,
    count: int,
    split: str,
) -> None:
    samples = []
    for index in range(count):
        samples.append(
            SatelliteSample(
                image_path=manifest_path.parent / source / "images" / f"{index}.png",
                mask_path=manifest_path.parent / source / "masks" / f"{index}.png",
                prepared_path=manifest_path.parent / source / "prepared" / f"{index}.pt",
                split=split,
                source=source,
            )
        )
    write_manifest(samples, manifest_path)


def _source_split_counts(samples: list[SatelliteSample]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        key = (sample.source, sample.split)
        counts[key] = counts.get(key, 0) + 1
    return counts


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

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
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
