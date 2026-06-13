"""Modal entrypoints for training and inference.

The project defaults to Modal for execution, while local targets remain useful
for small PyTorch debugging runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

from sim_satellite.infer import (
    InferenceConfig,
    coerce_inference_config,
    run_inference,
)
from sim_satellite.train import (
    TrainingConfig,
    coerce_training_config,
    run_training,
)

APP_NAME = "sim-satellite"
ARTIFACT_VOLUME_NAME = "sim-satellite-artifacts"
LOCAL_DATA_ROOT = Path("data")
MODAL_ARTIFACT_ROOT = Path("/artifacts")
MODAL_DATA_ROOT = Path("/data")
MODAL_GPU = "A10G"
DATA_MOUNT_CONFIGURED = LOCAL_DATA_ROOT.exists()


def _ignore_modal_data_path(path: Path) -> bool:
    return "raw" in path.parts or path.suffix == ".zip"


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("click", "matplotlib", "pillow", "torch")
    .add_local_python_source("sim_satellite")
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
    timeout=3600,
    gpu=MODAL_GPU,
    volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume},
)
def train_remote(config: dict[str, Any]) -> dict[str, Any]:
    result = run_training(TrainingConfig.from_dict(config)).to_dict()
    artifact_volume.commit()
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
    result = run_inference(InferenceConfig.from_dict(config)).to_dict()
    artifact_volume.commit()
    result["execution_target"] = "modal"
    return result


def run_training_target(
    config: TrainingConfig | Mapping[str, Any] | None = None,
    target: str = "modal",
) -> dict[str, Any]:
    config = coerce_training_config(config)
    if target == "local":
        result = run_training(config).to_dict()
        result["execution_target"] = "local"
        return result
    if target != "modal":
        raise ValueError(f"Unsupported training target: {target}")
    modal_config = _modal_training_config(config)
    with modal.enable_output():
        with app.run():
            return train_remote.remote(modal_config)


def run_inference_target(
    config: InferenceConfig | Mapping[str, Any] | None = None,
    target: str = "modal",
) -> dict[str, Any]:
    config = coerce_inference_config(config)
    if target == "local":
        result = run_inference(config).to_dict()
        result["execution_target"] = "local"
        return result
    if target != "modal":
        raise ValueError(f"Unsupported inference target: {target}")
    modal_config = _modal_inference_config(config)
    with modal.enable_output():
        with app.run():
            return infer_remote.remote(modal_config)


def _modal_training_config(config: TrainingConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["device"] = "cuda"
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path, field_name="manifest_path")
    if config.metrics_path:
        values["metrics_path"] = _modal_artifact_path(config.metrics_path)
    if config.loss_plot_path:
        values["loss_plot_path"] = _modal_artifact_path(config.loss_plot_path)
    return values


def _modal_inference_config(config: InferenceConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["device"] = "cuda"
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["output_path"] = _modal_artifact_path(config.output_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path, field_name="manifest_path")
    if config.mask_path is None:
        raise ValueError("Modal mask_path is required for inference")
    values["mask_path"] = _modal_data_path(config.mask_path, field_name="mask_path")
    return values


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
