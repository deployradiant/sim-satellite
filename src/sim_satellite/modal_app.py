"""Modal entrypoints for training and inference.

The project defaults to Modal for execution, while local targets remain useful
for small PyTorch debugging runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from sim_satellite.data.openearthmap import download_and_prepare_openearthmap
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.infer import (
    InferenceConfig,
    run_inference,
)
from sim_satellite.train import (
    TrainingConfig,
    run_training,
)
from sim_satellite.web import WebConfig, create_app

APP_NAME = os.environ.get("SIM_SATELLITE_MODAL_APP_NAME", "sim-satellite")
ARTIFACT_VOLUME_NAME = "sim-satellite-artifacts"
LOCAL_DATA_ROOT = Path("data")
LOCAL_WEB_STATIC_ROOT = Path(__file__).with_name("web_static")
MODAL_ARTIFACT_ROOT = Path("/artifacts")
MODAL_DATA_ROOT = Path("/data")
MODAL_EPHEMERAL_DISK_MIB = int(
    os.environ.get("SIM_SATELLITE_MODAL_EPHEMERAL_DISK_MIB", str(512 * 1024))
)
MODAL_PACKAGE_ROOT = Path("/root/sim_satellite")
MODAL_GPU = os.environ.get("SIM_SATELLITE_MODAL_GPU", "H100!")
MODAL_WEB_CHECKPOINT_GLOB = os.environ.get(
    "SIM_SATELLITE_MODAL_WEB_CHECKPOINT_GLOB",
    "artifacts/openearthmap-flow-unet-93m-256-nodeep-lr2e4-h100-b10-20k-save1k-step-*.pt",
)
MODAL_TRAINING_TIMEOUT = 24 * 60 * 60
MODAL_DATA_SETUP_TIMEOUT = 2 * 60 * 60


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("click", "fastapi", "matplotlib", "pillow", "rasterio", "torch")
    .add_local_python_source("sim_satellite")
    .add_local_file(
        LOCAL_WEB_STATIC_ROOT / "index.html",
        str(MODAL_PACKAGE_ROOT / "web_static" / "index.html"),
    )
)

artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=MODAL_TRAINING_TIMEOUT,
    gpu=MODAL_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def train_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_openearthmap_data(config["manifest_path"])
    result = run_training(
        TrainingConfig.from_dict(config),
        after_save=artifact_volume.commit,
    )
    artifact_volume.commit()
    result = asdict(result)
    result["execution_target"] = "modal"
    return result


@app.function(
    image=image,
    timeout=MODAL_DATA_SETUP_TIMEOUT,
    gpu=MODAL_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def infer_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_openearthmap_data(config["manifest_path"])
    artifact_volume.reload()
    result = asdict(run_inference(InferenceConfig.from_dict(config)))
    artifact_volume.commit()
    result["execution_target"] = "modal"
    return result


@app.function(
    image=image,
    timeout=MODAL_DATA_SETUP_TIMEOUT,
    gpu=MODAL_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
@modal.asgi_app()
def web_remote() -> Any:
    _ensure_modal_openearthmap_data(WebConfig.manifest_path)
    artifact_volume.reload()
    return create_app(
        _modal_web_config(WebConfig(checkpoint_glob=MODAL_WEB_CHECKPOINT_GLOB)),
        before_checkpoint_resolve=artifact_volume.reload,
    )


def run_training_target(
    config: TrainingConfig,
) -> dict[str, Any]:
    modal_config = _modal_training_config(config)
    with modal.enable_output():
        with app.run():
            return train_remote.remote(modal_config)


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
    return WebConfig(
        checkpoint_path=_modal_artifact_path(config.checkpoint_path),
        checkpoint_glob=(
            _modal_artifact_path(config.checkpoint_glob) if config.checkpoint_glob else ""
        ),
        manifest_path=_modal_data_path(config.manifest_path, field_name="manifest_path"),
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
