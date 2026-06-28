"""Modal entrypoints for training and inference.

The project defaults to Modal for execution, while local targets remain useful
for small PyTorch debugging runs.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from sim_satellite.data.landcoverai import download_and_prepare_landcoverai
from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import download_and_prepare_openearthmap
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.infer import (
    InferenceConfig,
    run_inference,
)
from sim_satellite.train import run_training
from sim_satellite.training_config import (
    TrainingConfig,
    load_training_config_values,
)
from sim_satellite.web import WebConfig, create_app

APP_NAME = os.environ.get("SIM_SATELLITE_MODAL_APP_NAME", "sim-satellite")
ARTIFACT_VOLUME_NAME = "sim-satellite-artifacts"
LOCAL_DATA_ROOT = Path("data")
LOCAL_MODAL_WEB_DATA_ROOT = Path(
    os.environ.get("SIM_SATELLITE_LOCAL_MODAL_WEB_DATA_ROOT", "data/modal-web-preview")
)
LOCAL_WEB_STATIC_ROOT = Path(__file__).with_name("web_static")
MODAL_ARTIFACT_ROOT = Path("/artifacts")
MODAL_DATA_ROOT = Path("/data")
MODAL_EPHEMERAL_DISK_MIB = int(
    os.environ.get("SIM_SATELLITE_MODAL_EPHEMERAL_DISK_MIB", str(512 * 1024))
)
MODAL_PACKAGE_ROOT = Path("/root/sim_satellite")
MODAL_WEB_DATA_ROOT = Path(
    os.environ.get(
        "SIM_SATELLITE_MODAL_WEB_DATA_ROOT",
        str(MODAL_PACKAGE_ROOT / "modal_web_data"),
    )
)
MODAL_WEB_MANIFEST_PATH = MODAL_WEB_DATA_ROOT / "manifest.jsonl"
MODAL_OPENEARTHMAP_DATA_DIR = MODAL_DATA_ROOT / "openearthmap"
MODAL_LANDCOVERAI_DATA_DIR = MODAL_DATA_ROOT / "landcoverai"
MODAL_LANDCOVERAI_MIX_ROOT = MODAL_DATA_ROOT / "landcoverai-mix"
LOCAL_LANDCOVERAI_ARCHIVE_PATH = os.environ.get("SIM_SATELLITE_LOCAL_LANDCOVERAI_ARCHIVE_PATH")
MODAL_PACKAGED_LANDCOVERAI_ARCHIVE = MODAL_PACKAGE_ROOT / "landcover.ai.v1.zip"
MODAL_LANDCOVERAI_MIX_SEED = int(os.environ.get("SIM_SATELLITE_LANDCOVERAI_MIX_SEED", "20260626"))
MODAL_LANDCOVERAI_MIX_VALIDATION_SAMPLES = int(
    os.environ.get("SIM_SATELLITE_LANDCOVERAI_MIX_VALIDATION_SAMPLES", "256")
)
MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION = float(
    os.environ.get("SIM_SATELLITE_LANDCOVERAI_MIX_MIN_LABELED_FRACTION", "0.10")
)
DEFAULT_MODAL_TRAINING_GPU = "H100!"
DEFAULT_MODAL_INFERENCE_GPU = "T4"
MODAL_TRAINING_GPU = os.environ.get(
    "SIM_SATELLITE_MODAL_TRAINING_GPU",
    DEFAULT_MODAL_TRAINING_GPU,
)
MODAL_INFERENCE_GPU = os.environ.get(
    "SIM_SATELLITE_MODAL_INFERENCE_GPU",
    DEFAULT_MODAL_INFERENCE_GPU,
)
MODAL_WEB_GPU = os.environ.get("SIM_SATELLITE_MODAL_WEB_GPU", MODAL_INFERENCE_GPU)
MODAL_WEB_CHECKPOINT_GLOB = os.environ.get(
    "SIM_SATELLITE_MODAL_WEB_CHECKPOINT_GLOB",
    "artifacts/openearthmap-flow-unet-25m-256-nodeep-lr2e4-h100-b10-20k-save1k-step-*.pt",
)
MODAL_WEB_COMPARISON_CHECKPOINT_GLOB = os.environ.get(
    "SIM_SATELLITE_MODAL_WEB_COMPARISON_CHECKPOINT_GLOB",
    "",
)
MODAL_IMAGE_ENV = {
    "SIM_SATELLITE_MODAL_WEB_CHECKPOINT_GLOB": MODAL_WEB_CHECKPOINT_GLOB,
    "SIM_SATELLITE_MODAL_WEB_COMPARISON_CHECKPOINT_GLOB": MODAL_WEB_COMPARISON_CHECKPOINT_GLOB,
}
if LOCAL_LANDCOVERAI_ARCHIVE_PATH:
    MODAL_IMAGE_ENV["SIM_SATELLITE_MODAL_LANDCOVERAI_ARCHIVE_PATH"] = str(
        MODAL_PACKAGED_LANDCOVERAI_ARCHIVE
    )
MODAL_WEB_ENV: dict[str, str | None] = {
    "SIM_SATELLITE_MODAL_WEB_CHECKPOINT_GLOB": MODAL_WEB_CHECKPOINT_GLOB,
    "SIM_SATELLITE_MODAL_WEB_COMPARISON_CHECKPOINT_GLOB": MODAL_WEB_COMPARISON_CHECKPOINT_GLOB,
}
MODAL_TRAINING_TIMEOUT = 24 * 60 * 60
MODAL_DATA_SETUP_TIMEOUT = 2 * 60 * 60


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(MODAL_IMAGE_ENV)
    .pip_install("click", "fastapi", "matplotlib", "pillow", "torch")
    .add_local_python_source("sim_satellite")
    .add_local_file(
        LOCAL_WEB_STATIC_ROOT / "index.html",
        str(MODAL_PACKAGE_ROOT / "web_static" / "index.html"),
    )
)
if LOCAL_MODAL_WEB_DATA_ROOT.exists():
    image = image.add_local_dir(
        LOCAL_MODAL_WEB_DATA_ROOT,
        str(MODAL_WEB_DATA_ROOT),
    )
if LOCAL_LANDCOVERAI_ARCHIVE_PATH:
    image = image.add_local_file(
        Path(LOCAL_LANDCOVERAI_ARCHIVE_PATH),
        str(MODAL_PACKAGED_LANDCOVERAI_ARCHIVE),
    )

artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=MODAL_TRAINING_TIMEOUT,
    gpu=MODAL_TRAINING_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def train_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_training_data(config["manifest_path"])
    result = run_training(
        load_training_config_values(config),
        after_save=artifact_volume.commit,
    )
    artifact_volume.commit()
    result = asdict(result)
    result["execution_target"] = "modal"
    return result


@app.function(
    image=image,
    timeout=MODAL_TRAINING_TIMEOUT,
    gpu=MODAL_TRAINING_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def train_many_remote(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for config in configs:
        _ensure_modal_training_data(config["manifest_path"])
        result = run_training(
            load_training_config_values(config),
            after_save=artifact_volume.commit,
        )
        artifact_volume.commit()
        result_payload = asdict(result)
        result_payload["execution_target"] = "modal"
        results.append(result_payload)
    return results


@app.function(
    image=image,
    timeout=MODAL_DATA_SETUP_TIMEOUT,
    gpu=MODAL_INFERENCE_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def infer_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_training_data(config["manifest_path"])
    artifact_volume.reload()
    result = asdict(run_inference(InferenceConfig.from_dict(config)))
    artifact_volume.commit()
    result["execution_target"] = "modal"
    return result


@app.function(
    image=image,
    env=MODAL_WEB_ENV,
    timeout=MODAL_DATA_SETUP_TIMEOUT,
    gpu=MODAL_WEB_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
@modal.asgi_app()
def web_remote() -> Any:
    web_config = WebConfig(
        checkpoint_glob=MODAL_WEB_CHECKPOINT_GLOB,
        comparison_checkpoint_glob=MODAL_WEB_COMPARISON_CHECKPOINT_GLOB,
        manifest_path=str(MODAL_WEB_MANIFEST_PATH),
    )
    if not MODAL_WEB_MANIFEST_PATH.exists():
        _ensure_modal_openearthmap_data(WebConfig.manifest_path)
        web_config = WebConfig(
            checkpoint_glob=MODAL_WEB_CHECKPOINT_GLOB,
            comparison_checkpoint_glob=MODAL_WEB_COMPARISON_CHECKPOINT_GLOB,
        )
    artifact_volume.reload()
    return create_app(
        _modal_web_config(web_config),
        before_checkpoint_resolve=artifact_volume.reload,
    )


def run_training_target(
    config: TrainingConfig,
) -> dict[str, Any]:
    modal_config = _modal_training_config(config)
    with modal.enable_output():
        with app.run():
            return train_remote.remote(modal_config)


def run_training_sweep_target(
    configs: list[TrainingConfig],
) -> list[dict[str, Any]]:
    modal_configs = [_modal_training_config(config) for config in configs]
    with modal.enable_output():
        with app.run():
            return train_many_remote.remote(modal_configs)


def run_inference_target(
    config: InferenceConfig,
) -> dict[str, Any]:
    modal_config = _modal_inference_config(config)
    with modal.enable_output():
        with app.run():
            return infer_remote.remote(modal_config)


def _modal_training_config(config: TrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    values["device"] = "cuda"
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path, field_name="manifest_path")
    if config.metrics_path:
        values["metrics_path"] = _modal_artifact_path(config.metrics_path)
    if config.loss_plot_path:
        values["loss_plot_path"] = _modal_artifact_path(config.loss_plot_path)
    if config.resume_from:
        values["resume_from"] = _modal_artifact_path(config.resume_from)
    return values


def _modal_inference_config(config: InferenceConfig) -> dict[str, Any]:
    values = asdict(config)
    values["device"] = "cuda"
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["output_path"] = _modal_artifact_path(config.output_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path, field_name="manifest_path")
    if config.mask_path is None:
        raise ValueError("Modal mask_path is required for inference")
    values["mask_path"] = _modal_data_path(config.mask_path, field_name="mask_path")
    return values


def _modal_web_config(config: WebConfig) -> WebConfig:
    manifest_path = Path(config.manifest_path)
    return WebConfig(
        checkpoint_path=_modal_artifact_path(config.checkpoint_path),
        checkpoint_glob=(
            _modal_artifact_path(config.checkpoint_glob) if config.checkpoint_glob else ""
        ),
        checkpoint_label=config.checkpoint_label,
        comparison_checkpoint_path=(
            _modal_artifact_path(config.comparison_checkpoint_path)
            if config.comparison_checkpoint_path
            else ""
        ),
        comparison_checkpoint_glob=(
            _modal_artifact_path(config.comparison_checkpoint_glob)
            if config.comparison_checkpoint_glob
            else ""
        ),
        comparison_checkpoint_label=config.comparison_checkpoint_label,
        manifest_path=(
            str(manifest_path)
            if _is_relative_to(manifest_path, MODAL_WEB_DATA_ROOT)
            else _modal_data_path(config.manifest_path, field_name="manifest_path")
        ),
        split=config.split,
        device="cuda",
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed,
        host=config.host,
        port=config.port,
        max_samples=config.max_samples,
        display_size=config.display_size,
    )


def _modal_artifact_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            relative_path = path.relative_to(MODAL_ARTIFACT_ROOT)
        except ValueError:
            raise ValueError(
                f"Modal artifact paths must be relative or under {MODAL_ARTIFACT_ROOT}: {value!r}"
            ) from None
        if ".." in relative_path.parts:
            raise ValueError(
                f"Modal artifact paths must stay under {MODAL_ARTIFACT_ROOT}: {value!r}"
            )
        return str(path)
    if path.parts and path.parts[0] == "artifacts":
        path = path.relative_to("artifacts")
    if ".." in path.parts:
        raise ValueError(f"Modal artifact paths must stay under {MODAL_ARTIFACT_ROOT}: {value!r}")
    return str(MODAL_ARTIFACT_ROOT / path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _modal_data_path(value: str, *, field_name: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path.relative_to(MODAL_DATA_ROOT)
        except ValueError:
            try:
                relative_path = path.relative_to((Path.cwd() / LOCAL_DATA_ROOT).resolve())
            except ValueError:
                raise ValueError(
                    f"Modal {field_name} must be relative, under local {LOCAL_DATA_ROOT}/, "
                    f"or under {MODAL_DATA_ROOT}: {value!r}"
                ) from None
            return str(MODAL_DATA_ROOT / relative_path)
        return str(path)

    if path.parts and path.parts[0] == str(LOCAL_DATA_ROOT):
        path = path.relative_to(LOCAL_DATA_ROOT)
    if ".." in path.parts:
        raise ValueError(f"Modal {field_name} must stay under {MODAL_DATA_ROOT}: {value!r}")
    return str(MODAL_DATA_ROOT / path)


def _ensure_modal_openearthmap_data(manifest_path: str) -> None:
    data_dir = _modal_data_dir_for_manifest(manifest_path)
    if _modal_openearthmap_data_ready(data_dir, Path(manifest_path)):
        _print_modal_data_event("modal_data_ready", data_dir=str(data_dir))
        return

    _print_modal_data_event(
        "modal_data_setup_start",
        data_dir=str(data_dir),
        ephemeral_disk_mib=MODAL_EPHEMERAL_DISK_MIB,
    )
    result = download_and_prepare_openearthmap(
        data_dir,
        force_prepared=True,
        progress=lambda message: _print_modal_data_event(
            "modal_data_setup_progress",
            data_dir=str(data_dir),
            message=message,
        ),
    )
    _print_modal_data_event(
        "modal_data_setup_done",
        data_dir=str(result.data_dir),
        manifest_path=str(result.manifest_path),
        prepared_dir=str(result.prepared_dir),
        samples=result.samples,
        prepared_samples=result.prepared_samples,
        scanned_images=result.scanned_images,
        missing_masks=result.missing_masks,
    )


def _ensure_modal_training_data(manifest_path: str) -> None:
    manifest = Path(manifest_path)
    if _is_modal_landcoverai_mix_manifest(manifest):
        _ensure_modal_landcoverai_mix_data(manifest)
        return
    _ensure_modal_openearthmap_data(manifest_path)


def _is_modal_landcoverai_mix_manifest(manifest_path: Path) -> bool:
    try:
        manifest_path.relative_to(MODAL_LANDCOVERAI_MIX_ROOT)
    except ValueError:
        return False
    return manifest_path.name == "manifest.jsonl"


def _ensure_modal_landcoverai_mix_data(manifest_path: Path) -> None:
    ratio = _landcoverai_mix_ratio_from_manifest(manifest_path)
    if _modal_landcoverai_mix_data_ready(manifest_path, landcoverai_ratio=ratio):
        _print_modal_data_event("modal_landcoverai_mix_ready", manifest_path=str(manifest_path))
        return

    ratio_label: str | float = "all" if ratio is None else ratio
    _print_modal_data_event(
        "modal_landcoverai_mix_setup_start",
        manifest_path=str(manifest_path),
        landcoverai_ratio=ratio_label,
        validation_samples=MODAL_LANDCOVERAI_MIX_VALIDATION_SAMPLES,
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
    )
    _ensure_modal_openearthmap_data(str(MODAL_OPENEARTHMAP_DATA_DIR / "manifest.jsonl"))
    if ratio is None or ratio > 0.0:
        _ensure_modal_landcoverai_data()
    _write_landcoverai_mix_manifest(
        manifest_path,
        openearthmap_manifest=MODAL_OPENEARTHMAP_DATA_DIR / "manifest.jsonl",
        landcoverai_manifest=MODAL_LANDCOVERAI_DATA_DIR / "manifest.jsonl",
        landcoverai_ratio=ratio,
        validation_samples=MODAL_LANDCOVERAI_MIX_VALIDATION_SAMPLES,
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
        seed=MODAL_LANDCOVERAI_MIX_SEED,
    )
    _print_modal_data_event(
        "modal_landcoverai_mix_setup_done",
        manifest_path=str(manifest_path),
        landcoverai_ratio=ratio_label,
    )


def _ensure_modal_landcoverai_data() -> None:
    manifest_path = MODAL_LANDCOVERAI_DATA_DIR / "manifest.jsonl"
    if _modal_landcoverai_data_ready(
        manifest_path,
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
    ):
        _print_modal_data_event(
            "modal_landcoverai_data_ready",
            data_dir=str(MODAL_LANDCOVERAI_DATA_DIR),
        )
        return

    _print_modal_data_event(
        "modal_landcoverai_data_setup_start",
        data_dir=str(MODAL_LANDCOVERAI_DATA_DIR),
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
    )
    _seed_modal_landcoverai_archive()
    result = download_and_prepare_landcoverai(
        MODAL_LANDCOVERAI_DATA_DIR,
        force_prepared=True,
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
        progress=lambda message: _print_modal_data_event(
            "modal_landcoverai_data_setup_progress",
            data_dir=str(MODAL_LANDCOVERAI_DATA_DIR),
            message=message,
        ),
    )
    _write_manifest_marker(
        result.manifest_path,
        _landcoverai_data_marker_payload(
            manifest_path=result.manifest_path,
            min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
            samples=result.samples,
            prepared_samples=result.prepared_samples,
            total_tiles=result.total_tiles,
            supervised_tiles=result.supervised_tiles,
            skipped_tiles=result.skipped_tiles,
            split_counts=result.split_counts,
        ),
    )
    _print_modal_data_event(
        "modal_landcoverai_data_setup_done",
        data_dir=str(result.data_dir),
        manifest_path=str(result.manifest_path),
        prepared_dir=str(result.prepared_dir),
        samples=result.samples,
        prepared_samples=result.prepared_samples,
        total_tiles=result.total_tiles,
        supervised_tiles=result.supervised_tiles,
        skipped_tiles=result.skipped_tiles,
        split_counts=result.split_counts,
    )


def _seed_modal_landcoverai_archive() -> None:
    packaged_archive = os.environ.get("SIM_SATELLITE_MODAL_LANDCOVERAI_ARCHIVE_PATH")
    if not packaged_archive:
        return
    source_path = Path(packaged_archive)
    if not source_path.exists():
        raise FileNotFoundError(f"Packaged LandCover.ai archive not found: {source_path}")
    archive_path = MODAL_LANDCOVERAI_DATA_DIR / "raw" / "landcover.ai.v1.zip"
    if archive_path.exists():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, archive_path)
    _print_modal_data_event(
        "modal_landcoverai_archive_seeded",
        data_dir=str(MODAL_LANDCOVERAI_DATA_DIR),
        source_path=str(source_path),
        archive_path=str(archive_path),
    )


def _landcoverai_mix_ratio_from_manifest(manifest_path: Path) -> float | None:
    name = manifest_path.parent.name
    if name == "all":
        return None
    if name == "baseline":
        return 0.0
    prefix = "ratio-"
    if not name.startswith(prefix):
        raise ValueError(
            "LandCover.ai mix manifest parent must be `baseline`, `all`, or `ratio-N`: "
            f"{manifest_path}"
        )
    try:
        percent = int(name.removeprefix(prefix))
    except ValueError:
        raise ValueError(
            f"Invalid LandCover.ai mix ratio in manifest path: {manifest_path}"
        ) from None
    if percent < 0:
        raise ValueError(f"LandCover.ai mix ratio must be non-negative: {manifest_path}")
    return percent / 100.0


def _write_landcoverai_mix_manifest(
    manifest_path: Path,
    *,
    openearthmap_manifest: Path,
    landcoverai_manifest: Path,
    landcoverai_ratio: float | None,
    validation_samples: int,
    min_labeled_fraction: float,
    seed: int,
) -> None:
    openearthmap_source_samples = read_manifest(openearthmap_manifest)
    openearthmap_samples = [
        sample for sample in openearthmap_source_samples if sample.split == "train"
    ]
    if not openearthmap_samples:
        raise RuntimeError(f"OpenEarthMap manifest has no train samples: {openearthmap_manifest}")
    if validation_samples <= 0 or validation_samples >= len(openearthmap_samples):
        raise RuntimeError(
            f"validation_samples must be between 1 and {len(openearthmap_samples) - 1}"
        )

    shuffled_openearthmap = list(openearthmap_samples)
    random.Random(seed).shuffle(shuffled_openearthmap)
    openearthmap_val = shuffled_openearthmap[:validation_samples]
    openearthmap_train = shuffled_openearthmap[validation_samples:]

    mixed_samples = [
        _with_split(sample, "train")
        for sample in sorted(openearthmap_train, key=lambda item: str(item.image_path))
    ]
    landcoverai_count = 0
    landcoverai_source_samples: list[SatelliteSample] = []
    landcoverai_samples: list[SatelliteSample] = []
    landcoverai_ratio_label: str | float = "all" if landcoverai_ratio is None else landcoverai_ratio
    if landcoverai_ratio is None or landcoverai_ratio > 0.0:
        landcoverai_source_samples = read_manifest(landcoverai_manifest)
        landcoverai_samples = [
            sample for sample in landcoverai_source_samples if sample.split == "train"
        ]
        landcoverai_count = (
            len(landcoverai_samples)
            if landcoverai_ratio is None
            else int(len(openearthmap_train) * landcoverai_ratio)
        )
        if len(landcoverai_samples) < landcoverai_count:
            raise RuntimeError(
                f"LandCover.ai manifest only has {len(landcoverai_samples)} train samples; "
                f"need {landcoverai_count}"
            )
        ratio_seed_offset = (
            1_000_000 if landcoverai_ratio is None else int(landcoverai_ratio * 10_000)
        )
        random.Random(seed + ratio_seed_offset).shuffle(landcoverai_samples)
        mixed_samples.extend(
            _with_split(sample, "train")
            for sample in sorted(
                landcoverai_samples[:landcoverai_count],
                key=lambda item: str(item.image_path),
            )
        )
    mixed_samples.extend(
        _with_split(sample, "val")
        for sample in sorted(openearthmap_val, key=lambda item: str(item.image_path))
    )
    write_manifest(mixed_samples, manifest_path)
    _write_manifest_marker(
        manifest_path,
        _landcoverai_mix_marker_payload(
            manifest_path=manifest_path,
            openearthmap_manifest=openearthmap_manifest,
            landcoverai_manifest=landcoverai_manifest,
            landcoverai_ratio=landcoverai_ratio,
            validation_samples=validation_samples,
            min_labeled_fraction=min_labeled_fraction,
            seed=seed,
            openearthmap_source_samples=len(openearthmap_source_samples),
            openearthmap_source_train_samples=len(openearthmap_samples),
            openearthmap_train_samples=len(openearthmap_train),
            openearthmap_val_samples=len(openearthmap_val),
            landcoverai_source_samples=len(landcoverai_source_samples),
            landcoverai_source_train_samples=len(landcoverai_samples),
            landcoverai_train_samples=landcoverai_count,
            total_samples=len(mixed_samples),
        ),
    )
    _print_modal_data_event(
        "modal_landcoverai_mix_manifest_written",
        manifest_path=str(manifest_path),
        openearthmap_train=len(openearthmap_train),
        openearthmap_val=len(openearthmap_val),
        landcoverai_ratio=landcoverai_ratio_label,
        landcoverai_train=landcoverai_count,
        train_samples=len(mixed_samples) - len(openearthmap_val),
        total_samples=len(mixed_samples),
    )


def _with_split(sample: SatelliteSample, split: str) -> SatelliteSample:
    return SatelliteSample(
        image_path=sample.image_path,
        mask_path=sample.mask_path,
        prepared_path=sample.prepared_path,
        split=split,
        source=sample.source,
    )


def _modal_data_dir_for_manifest(manifest_path: str) -> Path:
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = Path(_modal_data_path(manifest_path, field_name="manifest_path"))
    try:
        manifest.relative_to(MODAL_DATA_ROOT)
    except ValueError:
        raise ValueError(
            f"Modal manifest_path must be under {MODAL_DATA_ROOT}: {manifest_path!r}"
        ) from None
    return manifest.parent


def _modal_openearthmap_data_ready(data_dir: Path, manifest_path: Path) -> bool:
    prepared_dir = data_dir / "prepared"
    if not manifest_path.exists() or not prepared_dir.exists():
        return False
    first_prepared = next(prepared_dir.rglob("*.pt"), None)
    if first_prepared is None:
        return False
    try:
        import torch

        payload = torch.load(first_prepared, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("version") == PREPARED_DATA_VERSION


def _modal_manifest_data_ready(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        samples = read_manifest(manifest_path)
    except (OSError, ValueError):
        return False
    if not samples:
        return False
    first_prepared = samples[0].prepared_path
    if any(not sample.prepared_path.exists() for sample in samples):
        return False
    try:
        import torch

        payload = torch.load(first_prepared, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("version") == PREPARED_DATA_VERSION


def _modal_landcoverai_data_ready(
    manifest_path: Path,
    *,
    min_labeled_fraction: float,
) -> bool:
    if not _modal_manifest_data_ready(manifest_path):
        return False
    return _manifest_marker_matches(
        manifest_path,
        {
            "kind": "landcoverai-data",
            "manifest_path": str(manifest_path),
            "min_labeled_fraction": min_labeled_fraction,
            "prepared_data_version": PREPARED_DATA_VERSION,
        },
    )


def _modal_landcoverai_mix_data_ready(
    manifest_path: Path,
    *,
    landcoverai_ratio: float | None,
) -> bool:
    if not _modal_manifest_data_ready(manifest_path):
        return False
    marker_payload = _expected_landcoverai_mix_marker_payload(
        manifest_path,
        openearthmap_manifest=MODAL_OPENEARTHMAP_DATA_DIR / "manifest.jsonl",
        landcoverai_manifest=MODAL_LANDCOVERAI_DATA_DIR / "manifest.jsonl",
        landcoverai_ratio=landcoverai_ratio,
        validation_samples=MODAL_LANDCOVERAI_MIX_VALIDATION_SAMPLES,
        min_labeled_fraction=MODAL_LANDCOVERAI_MIX_MIN_LABELED_FRACTION,
        seed=MODAL_LANDCOVERAI_MIX_SEED,
    )
    if marker_payload is None:
        return False
    return _manifest_marker_matches(manifest_path, marker_payload)


def _expected_landcoverai_mix_marker_payload(
    manifest_path: Path,
    *,
    openearthmap_manifest: Path,
    landcoverai_manifest: Path,
    landcoverai_ratio: float | None,
    validation_samples: int,
    min_labeled_fraction: float,
    seed: int,
) -> dict[str, Any] | None:
    try:
        openearthmap_source_samples = read_manifest(openearthmap_manifest)
        openearthmap_samples = [
            sample for sample in openearthmap_source_samples if sample.split == "train"
        ]
        if validation_samples <= 0 or validation_samples >= len(openearthmap_samples):
            return None
        openearthmap_train_samples = len(openearthmap_samples) - validation_samples
        landcoverai_source_samples: list[SatelliteSample] = []
        landcoverai_samples: list[SatelliteSample] = []
        landcoverai_train_samples = 0
        if landcoverai_ratio is None or landcoverai_ratio > 0.0:
            landcoverai_source_samples = read_manifest(landcoverai_manifest)
            landcoverai_samples = [
                sample for sample in landcoverai_source_samples if sample.split == "train"
            ]
            landcoverai_train_samples = (
                len(landcoverai_samples)
                if landcoverai_ratio is None
                else int(openearthmap_train_samples * landcoverai_ratio)
            )
            if len(landcoverai_samples) < landcoverai_train_samples:
                return None
    except (OSError, RuntimeError, ValueError):
        return None

    return _landcoverai_mix_marker_payload(
        manifest_path=manifest_path,
        openearthmap_manifest=openearthmap_manifest,
        landcoverai_manifest=landcoverai_manifest,
        landcoverai_ratio=landcoverai_ratio,
        validation_samples=validation_samples,
        min_labeled_fraction=min_labeled_fraction,
        seed=seed,
        openearthmap_source_samples=len(openearthmap_source_samples),
        openearthmap_source_train_samples=len(openearthmap_samples),
        openearthmap_train_samples=openearthmap_train_samples,
        openearthmap_val_samples=validation_samples,
        landcoverai_source_samples=len(landcoverai_source_samples),
        landcoverai_source_train_samples=len(landcoverai_samples),
        landcoverai_train_samples=landcoverai_train_samples,
        total_samples=openearthmap_train_samples + validation_samples + landcoverai_train_samples,
    )


def _landcoverai_data_marker_payload(
    *,
    manifest_path: Path,
    min_labeled_fraction: float,
    samples: int,
    prepared_samples: int,
    total_tiles: int,
    supervised_tiles: int,
    skipped_tiles: int,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "kind": "landcoverai-data",
        "marker_version": 1,
        "manifest_path": str(manifest_path),
        "min_labeled_fraction": min_labeled_fraction,
        "prepared_data_version": PREPARED_DATA_VERSION,
        "samples": samples,
        "prepared_samples": prepared_samples,
        "total_tiles": total_tiles,
        "supervised_tiles": supervised_tiles,
        "skipped_tiles": skipped_tiles,
        "split_counts": split_counts,
    }


def _landcoverai_mix_marker_payload(
    *,
    manifest_path: Path,
    openearthmap_manifest: Path,
    landcoverai_manifest: Path,
    landcoverai_ratio: float | None,
    validation_samples: int,
    min_labeled_fraction: float,
    seed: int,
    openearthmap_source_samples: int,
    openearthmap_source_train_samples: int,
    openearthmap_train_samples: int,
    openearthmap_val_samples: int,
    landcoverai_source_samples: int,
    landcoverai_source_train_samples: int,
    landcoverai_train_samples: int,
    total_samples: int,
) -> dict[str, Any]:
    return {
        "kind": "landcoverai-mix",
        "marker_version": 1,
        "manifest_path": str(manifest_path),
        "prepared_data_version": PREPARED_DATA_VERSION,
        "openearthmap_manifest": str(openearthmap_manifest),
        "openearthmap_source_samples": openearthmap_source_samples,
        "openearthmap_source_train_samples": openearthmap_source_train_samples,
        "openearthmap_train_samples": openearthmap_train_samples,
        "openearthmap_val_samples": openearthmap_val_samples,
        "landcoverai_manifest": str(landcoverai_manifest),
        "landcoverai_ratio": landcoverai_ratio,
        "landcoverai_source_samples": landcoverai_source_samples,
        "landcoverai_source_train_samples": landcoverai_source_train_samples,
        "landcoverai_train_samples": landcoverai_train_samples,
        "min_labeled_fraction": min_labeled_fraction,
        "seed": seed,
        "validation_samples": validation_samples,
        "total_samples": total_samples,
    }


def _write_manifest_marker(manifest_path: Path, payload: dict[str, Any]) -> None:
    marker_path = _manifest_marker_path(manifest_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_marker_matches(manifest_path: Path, expected: dict[str, Any]) -> bool:
    marker_path = _manifest_marker_path(manifest_path)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def _manifest_marker_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.name}.meta.json")


def _print_modal_data_event(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "logged_at_epoch_utc": time.time(),
                **payload,
            },
            sort_keys=True,
        ),
        flush=True,
    )
