"""FastAPI server for the local mask-conditioned generation website."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.infer import InferenceConfig, generate_image
from sim_satellite.model import ModelConfig
from sim_satellite.train import DEFAULT_MANIFEST_PATH

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DISPLAY_SIZE = 256
DEFAULT_MAX_SAMPLES = 32
DEFAULT_WEB_CHECKPOINT_PATH = "artifacts/tiny-sim-satellite-modal.pt"
STATIC_DIR = Path(__file__).with_name("web_static")
INDEX_PATH = STATIC_DIR / "index.html"


@dataclass(frozen=True)
class WebConfig:
    checkpoint_path: str = DEFAULT_WEB_CHECKPOINT_PATH
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    device: str = "cpu"
    steps: int = 8
    seed: int = 1337
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_samples: int = DEFAULT_MAX_SAMPLES
    display_size: int = DEFAULT_DISPLAY_SIZE

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.port < 1:
            raise ValueError("port must be positive")
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.display_size < 1:
            raise ValueError("display_size must be positive")


class GenerateRequest(BaseModel):
    index: int = 0
    seed: int | None = None
    steps: int | None = None


def create_app(config: WebConfig | None = None) -> FastAPI:
    config = config or WebConfig()
    samples = _load_samples(config)
    app = FastAPI(title="Sim Satellite", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_PATH)

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return _state_payload(config, samples)

    @app.get("/api/mask")
    def mask(index: int = Query(default=0, ge=0)) -> Response:
        sample = _sample_at(samples, index)
        return Response(
            content=_mask_preview_png(sample.mask_path, config),
            media_type="image/png",
        )

    @app.post("/api/generate")
    def generate(payload: GenerateRequest) -> dict[str, Any]:
        sample = _sample_at(samples, payload.index)
        seed = config.seed if payload.seed is None else payload.seed
        steps = config.steps if payload.steps is None else payload.steps
        if steps < 1:
            raise HTTPException(status_code=400, detail="steps must be positive")

        try:
            generated = generate_image(
                InferenceConfig(
                    seed=seed,
                    checkpoint_path=config.checkpoint_path,
                    output_path="artifacts/web-sample.pt",
                    manifest_path=config.manifest_path,
                    split=config.split,
                    mask_path=str(sample.mask_path),
                    device=config.device,
                    steps=steps,
                )
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        png = _image_tensor_png(generated.image, config.display_size)
        return {
            "image": _data_uri(png, "image/png"),
            "sample": _sample_payload(payload.index, sample),
            "seed": seed,
            "steps": steps,
        }

    return app


def serve(config: WebConfig | None = None) -> None:
    config = config or WebConfig()
    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port)


def _load_samples(config: WebConfig) -> tuple[SatelliteSample, ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Web app manifest not found: {manifest_path}. "
            "Run `sim-satellite openearthmap` first."
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Web app HTML not found: {INDEX_PATH}")

    samples = tuple(SatelliteDataset(manifest_path, split=config.split))
    if not samples:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")
    return samples[: config.max_samples]


def _state_payload(config: WebConfig, samples: Sequence[SatelliteSample]) -> dict[str, Any]:
    return {
        "checkpointPath": config.checkpoint_path,
        "manifestPath": config.manifest_path,
        "split": config.split,
        "device": config.device,
        "steps": config.steps,
        "seed": config.seed,
        "samples": [_sample_payload(index, sample) for index, sample in enumerate(samples)],
        "classes": [
            {
                "id": index,
                "name": name,
                "rgb": OPENEARTHMAP_RGB[name],
            }
            for index, name in enumerate(OPENEARTHMAP_CLASSES)
        ],
    }


def _sample_payload(index: int, sample: SatelliteSample) -> dict[str, Any]:
    return {
        "index": index,
        "name": sample.image_path.stem,
        "imagePath": str(sample.image_path),
        "maskPath": str(sample.mask_path),
        "source": sample.source,
        "split": sample.split,
    }


def _sample_at(samples: Sequence[SatelliteSample], index: int) -> SatelliteSample:
    if index < 0 or index >= len(samples):
        raise HTTPException(status_code=400, detail=f"Sample index out of range: {index}")
    return samples[index]


def _mask_preview_png(mask_path: Path, config: WebConfig) -> bytes:
    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    with Image.open(mask_path) as mask:
        resized = mask.resize(
            (config.display_size, config.display_size),
            Image.Resampling.NEAREST,
        )
        labels = mask_to_class_ids(resized, model_config)

    image = Image.new("RGB", (config.display_size, config.display_size))
    colors = [OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES]
    image.putdata([colors[label] for label in labels])
    return _pil_png(image)


def _image_tensor_png(image: Any, display_size: int) -> bytes:
    tensor = image.detach().cpu()[0].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0)
    height = int(tensor.shape[0])
    width = int(tensor.shape[1])
    pixels = [tuple(int(channel) for channel in pixel) for row in tensor.tolist() for pixel in row]
    pil_image = Image.new("RGB", (width, height))
    pil_image.putdata(pixels)
    if width != display_size or height != display_size:
        pil_image = pil_image.resize((display_size, display_size), Image.Resampling.NEAREST)
    return _pil_png(pil_image)


def _pil_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _data_uri(body: bytes, content_type: str) -> str:
    encoded = base64.b64encode(body).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _build_click_command() -> Any:
    try:
        import click
    except ModuleNotFoundError:
        return None

    @click.command("app", help="Launch the local mask-conditioned generation app.")
    @click.option("--checkpoint-path", default=WebConfig.checkpoint_path, show_default=True)
    @click.option("--manifest-path", default=WebConfig.manifest_path, show_default=True)
    @click.option("--split", default=WebConfig.split, show_default=True)
    @click.option("--device", default=WebConfig.device, show_default=True)
    @click.option("--steps", default=WebConfig.steps, show_default=True)
    @click.option("--seed", default=WebConfig.seed, show_default=True)
    @click.option("--host", default=WebConfig.host, show_default=True)
    @click.option("--port", default=WebConfig.port, show_default=True, type=int)
    @click.option("--max-samples", default=WebConfig.max_samples, show_default=True)
    def command(
        checkpoint_path: str,
        manifest_path: str,
        split: str,
        device: str,
        steps: int,
        seed: int,
        host: str,
        port: int,
        max_samples: int,
    ) -> None:
        serve(
            WebConfig(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                split=split,
                device=device,
                steps=steps,
                seed=seed,
                host=host,
                port=port,
                max_samples=max_samples,
            )
        )

    return command


cli = _build_click_command()


if __name__ == "__main__":
    if cli is None:
        raise SystemExit("Click is required for the web app CLI.")
    cli()
