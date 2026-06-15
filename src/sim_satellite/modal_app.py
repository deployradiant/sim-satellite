"""Modal entrypoints for training and inference.

The project defaults to Modal for execution, while local targets remain useful
for small PyTorch debugging runs.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

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
MODAL_PACKAGE_ROOT = Path("/root/sim_satellite")
MODAL_GPU = os.environ.get("SIM_SATELLITE_MODAL_GPU", "H100!")
MODAL_WEB_CHECKPOINT_GLOB = os.environ.get(
    "SIM_SATELLITE_MODAL_WEB_CHECKPOINT_GLOB",
    "artifacts/openearthmap-flow-unet-93m-256-nodeep-lr2e4-h100-b10-20k-save1k-step-*.pt",
)
MODAL_INCLUDE_EXTRACTED_DATA = os.environ.get("SIM_SATELLITE_MODAL_INCLUDE_EXTRACTED") == "1"
MODAL_TRAINING_TIMEOUT = 24 * 60 * 60
DATA_MOUNT_CONFIGURED = LOCAL_DATA_ROOT.exists()


def _ignore_modal_data_path(path: Path) -> bool:
    if "raw" in path.parts or path.suffix == ".zip":
        return True
    return not MODAL_INCLUDE_EXTRACTED_DATA and "extracted" in path.parts


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("click", "fastapi", "matplotlib", "pillow", "torch")
    .add_local_python_source("sim_satellite")
    .add_local_file(
        LOCAL_WEB_STATIC_ROOT / "index.html",
        str(MODAL_PACKAGE_ROOT / "web_static" / "index.html"),
    )
)
if DATA_MOUNT_CONFIGURED:
    image = image.add_local_dir(
        LOCAL_DATA_ROOT,
        str(MODAL_DATA_ROOT),
        copy=False,
        ignore=_ignore_modal_data_path,
    )

artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=MODAL_TRAINING_TIMEOUT,
    gpu=MODAL_GPU,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def train_remote(config: dict[str, Any]) -> dict[str, Any]:
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
    timeout=900,
    gpu=MODAL_GPU,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def infer_remote(config: dict[str, Any]) -> dict[str, Any]:
    artifact_volume.reload()
    result = asdict(run_inference(InferenceConfig.from_dict(config)))
    artifact_volume.commit()
    result["execution_target"] = "modal"
    return result


@app.function(
    image=image,
    timeout=900,
    gpu=MODAL_GPU,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
@modal.asgi_app()
def web_remote() -> Any:
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
    if MODAL_DATA_ROOT.exists():
        if path.is_absolute():
            try:
                path.relative_to(MODAL_DATA_ROOT)
            except ValueError:
                raise ValueError(
                    f"Modal {field_name} must be under {MODAL_DATA_ROOT}: {value!r}"
                ) from None
            return str(path)
        if path.parts and path.parts[0] == "data":
            path = path.relative_to("data")
        if ".." in path.parts:
            raise ValueError(f"Modal {field_name} must stay under {MODAL_DATA_ROOT}: {value!r}")
        return str(MODAL_DATA_ROOT / path)

    if not DATA_MOUNT_CONFIGURED:
        raise RuntimeError(
            f"Modal {field_name} cannot be used because {LOCAL_DATA_ROOT}/ did not exist when "
            "sim_satellite.modal_app was imported, so it was not mounted into the Modal image. "
            "Run `sim-satellite openearthmap` before starting Modal training or inference."
        )
    relative_path = _local_data_relative_path(value, field_name=field_name)
    return str(MODAL_DATA_ROOT / relative_path)


def _local_data_relative_path(value: str, *, field_name: str) -> Path:
    path = Path(value)
    local_path = path if path.is_absolute() else Path.cwd() / path
    data_root = LOCAL_DATA_ROOT.resolve()
    resolved_path = local_path.resolve()

    try:
        relative_path = resolved_path.relative_to(data_root)
    except ValueError:
        raise ValueError(
            f"Modal {field_name} must be under the local data/ cache: {value!r}. "
            f"Modal only mounts {LOCAL_DATA_ROOT}/ at {MODAL_DATA_ROOT}."
        ) from None

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Modal {field_name} not found locally: {value!r}. "
            f"Modal only mounts existing files from {LOCAL_DATA_ROOT}/ at {MODAL_DATA_ROOT}."
        )

    return relative_path
