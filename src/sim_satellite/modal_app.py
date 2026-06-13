"""Modal entrypoints for training and inference.

The project defaults to Modal for execution, while these helpers preserve a
local fallback so the scaffold remains useful before Modal is configured.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("click", "pillow", "torch")
        .add_local_python_source("sim_satellite")
    )
    if LOCAL_DATA_ROOT.exists():
        image = image.add_local_dir(LOCAL_DATA_ROOT, str(MODAL_DATA_ROOT), copy=False)

    artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

    @app.function(image=image, timeout=900, volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume})
    def train_remote(config: dict[str, Any]) -> dict[str, Any]:
        result = run_training(TrainingConfig.from_dict(config)).to_dict()
        artifact_volume.commit()
        result["execution_target"] = "modal"
        return result

    @app.function(image=image, timeout=300, volumes={str(MODAL_ARTIFACT_ROOT): artifact_volume})
    def infer_remote(config: dict[str, Any]) -> dict[str, Any]:
        artifact_volume.reload()
        result = run_inference(InferenceConfig.from_dict(config)).to_dict()
        artifact_volume.commit()
        result["execution_target"] = "modal"
        return result

else:
    app = None
    train_remote = None
    infer_remote = None


def require_modal() -> None:
    if app is None or train_remote is None or infer_remote is None:
        raise RuntimeError(
            "Modal is required for this command. Install dependencies with `uv sync`."
        )


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
    require_modal()
    with app.run():
        return train_remote.remote(_modal_training_config(config))


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
    require_modal()
    with app.run():
        return infer_remote.remote(_modal_inference_config(config))


def _modal_training_config(config: TrainingConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path)
    return values


def _modal_inference_config(config: InferenceConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["checkpoint_path"] = _modal_artifact_path(config.checkpoint_path)
    values["output_path"] = _modal_artifact_path(config.output_path)
    values["manifest_path"] = _modal_data_path(config.manifest_path)
    if config.mask_path is not None:
        values["mask_path"] = _modal_data_path(config.mask_path)
    return values


def _modal_artifact_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "artifacts":
        path = _without_first_part(path)
    return str(MODAL_ARTIFACT_ROOT / path)


def _modal_data_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "data":
        return str(MODAL_DATA_ROOT / _without_first_part(path))
    return str(path)


def _without_first_part(path: Path) -> Path:
    if len(path.parts) == 1:
        return Path()
    return Path(*path.parts[1:])
