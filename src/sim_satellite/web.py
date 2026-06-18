"""FastAPI server for the local mask-conditioned generation website."""

from __future__ import annotations

import base64
import binascii
import glob
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Any

import click
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.prepared import (
    PreparedSample,
    load_prepared_sample,
    prepared_coordinates_tuple,
)
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.infer import GenerationConfig, generate_image_with_model, load_generator
from sim_satellite.model import ModelConfig
from sim_satellite.train import DEFAULT_MANIFEST_PATH

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DISPLAY_SIZE = 256
DEFAULT_MAX_SAMPLES = 32
DEFAULT_WEB_CHECKPOINT_PATH = "artifacts/tiny-sim-satellite-modal.pt"
PRIMARY_CHECKPOINT_ID = "primary"
COMPARISON_CHECKPOINT_ID = "comparison"
STATIC_DIR = Path(__file__).with_name("web_static")
INDEX_PATH = STATIC_DIR / "index.html"
MASK_DATA_URI_PREFIX = "data:image/png;base64,"
MASK_COLORS = tuple(OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES)
STREAM_MEDIA_TYPE = "application/x-ndjson"
_STREAM_DONE = object()
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class WebConfig:
    checkpoint_path: str = DEFAULT_WEB_CHECKPOINT_PATH
    checkpoint_glob: str = ""
    checkpoint_label: str = "Coordinate checkpoint"
    comparison_checkpoint_path: str = ""
    comparison_checkpoint_glob: str = ""
    comparison_checkpoint_label: str = "No-coordinate checkpoint"
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    device: str = "cpu"
    steps: int = 8
    guidance_scale: float = 1.0
    seed: int = 2026
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_samples: int = DEFAULT_MAX_SAMPLES
    display_size: int = DEFAULT_DISPLAY_SIZE

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
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
    checkpoint: str = PRIMARY_CHECKPOINT_ID
    mask_image: str | None = Field(default=None, alias="maskImage")


@dataclass(frozen=True)
class CheckpointSlot:
    checkpoint_id: str
    label: str
    checkpoint_path: str
    checkpoint_glob: str


@dataclass(frozen=True)
class PreparedGeneration:
    index: int
    sample: SatelliteSample
    seed: int
    steps: int
    mask_path: Path
    coordinates: tuple[float, float] | None
    temporary_mask_path: Path | None
    mask_edited: bool


def create_app(
    config: WebConfig | None = None,
    *,
    before_checkpoint_resolve: Callable[[], None] | None = None,
) -> FastAPI:
    config = config or WebConfig()
    _validate_checkpoint_path(config)
    samples = _load_samples(config)
    loaded_generators: dict[str, tuple[tuple[Path, int, int], Any]] = {}
    app = FastAPI(title="Sim Satellite", docs_url=None, redoc_url=None)

    def checkpoint_slot(checkpoint_id: str) -> CheckpointSlot:
        return _checkpoint_slot(config, checkpoint_id)

    def checkpoint_path(checkpoint_id: str) -> Path:
        return _resolve_checkpoint_path(
            checkpoint_slot(checkpoint_id),
            before_checkpoint_resolve=before_checkpoint_resolve,
        )

    def generator(checkpoint_id: str) -> Any:
        try:
            resolved_checkpoint_path = checkpoint_path(checkpoint_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        checkpoint_key = _checkpoint_cache_key(resolved_checkpoint_path)
        cached = loaded_generators.get(checkpoint_id)
        if cached is None or cached[0] != checkpoint_key:
            loaded_generators[checkpoint_id] = (
                checkpoint_key,
                load_generator(resolved_checkpoint_path, device=config.device),
            )
        return loaded_generators[checkpoint_id][1]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_PATH)

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return _state_payload(
            config,
            samples,
            checkpoint_paths=_resolved_checkpoint_paths(
                config,
                before_checkpoint_resolve=before_checkpoint_resolve,
            ),
        )

    @app.get("/api/mask")
    def mask(index: int = Query(default=0, ge=0)) -> Response:
        sample = _sample_at(samples, index)
        return Response(
            content=_mask_preview_png(sample, config),
            media_type="image/png",
        )

    @app.get("/api/image")
    def image(index: int = Query(default=0, ge=0)) -> Response:
        sample = _sample_at(samples, index)
        return Response(
            content=_image_preview_png(sample, config),
            media_type="image/png",
        )

    @app.post("/api/generate")
    def generate(payload: GenerateRequest) -> dict[str, Any]:
        slot = checkpoint_slot(payload.checkpoint)
        prepared = _prepare_generation(payload, samples, config)
        return _run_generation(
            prepared,
            config,
            generator(slot.checkpoint_id),
            checkpoint_slot=slot,
        )

    @app.post("/api/generate/stream")
    def generate_stream(payload: GenerateRequest) -> StreamingResponse:
        slot = checkpoint_slot(payload.checkpoint)
        prepared = _prepare_generation(payload, samples, config)
        return StreamingResponse(
            _generate_stream(
                prepared,
                config,
                lambda: generator(slot.checkpoint_id),
                checkpoint_slot=slot,
            ),
            media_type=STREAM_MEDIA_TYPE,
        )

    return app


def serve(config: WebConfig | None = None) -> None:
    config = config or WebConfig()
    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port)


def _validate_checkpoint_path(config: WebConfig) -> None:
    for slot in _checkpoint_slots(config):
        if slot.checkpoint_glob:
            continue
        checkpoint_path = Path(slot.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Web app checkpoint not found for {slot.label}: {checkpoint_path}. "
                "Train a model or pass --checkpoint-path."
            )


def _load_samples(config: WebConfig) -> tuple[SatelliteSample, ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Web app manifest not found: {manifest_path}. Run `sim-satellite openearthmap` first."
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Web app HTML not found: {INDEX_PATH}")

    samples = tuple(SatelliteDataset(manifest_path, split=config.split))
    if not samples:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")
    samples = samples[: config.max_samples]
    missing_prepared = [
        sample.prepared_path
        for sample in samples
        if not sample.prepared_path.exists()
        and (not sample.image_path.exists() or not sample.mask_path.exists())
    ]
    if missing_prepared:
        raise FileNotFoundError(f"Web app sample data not found: {missing_prepared[0]}")
    return samples


def _state_payload(
    config: WebConfig,
    samples: Sequence[SatelliteSample],
    *,
    checkpoint_paths: Mapping[str, Path | None],
) -> dict[str, Any]:
    checkpoints = [
        _checkpoint_payload(slot, checkpoint_paths.get(slot.checkpoint_id))
        for slot in _checkpoint_slots(config)
    ]
    primary_checkpoint_path = checkpoint_paths.get(PRIMARY_CHECKPOINT_ID)
    return {
        "checkpointPath": (
            str(primary_checkpoint_path) if primary_checkpoint_path is not None else ""
        ),
        "checkpointGlob": config.checkpoint_glob,
        "checkpointAvailable": primary_checkpoint_path is not None,
        "checkpoints": checkpoints,
        "manifestPath": config.manifest_path,
        "split": config.split,
        "device": config.device,
        "steps": config.steps,
        "guidanceScale": config.guidance_scale,
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


def _checkpoint_payload(slot: CheckpointSlot, checkpoint_path: Path | None) -> dict[str, Any]:
    return {
        "id": slot.checkpoint_id,
        "label": slot.label,
        "path": str(checkpoint_path) if checkpoint_path is not None else "",
        "glob": slot.checkpoint_glob,
        "available": checkpoint_path is not None,
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


def _checkpoint_slots(config: WebConfig) -> tuple[CheckpointSlot, ...]:
    slots = [
        CheckpointSlot(
            checkpoint_id=PRIMARY_CHECKPOINT_ID,
            label=config.checkpoint_label,
            checkpoint_path=config.checkpoint_path,
            checkpoint_glob=config.checkpoint_glob,
        )
    ]
    if config.comparison_checkpoint_path or config.comparison_checkpoint_glob:
        slots.append(
            CheckpointSlot(
                checkpoint_id=COMPARISON_CHECKPOINT_ID,
                label=config.comparison_checkpoint_label,
                checkpoint_path=config.comparison_checkpoint_path,
                checkpoint_glob=config.comparison_checkpoint_glob,
            )
        )
    return tuple(slots)


def _checkpoint_slot(config: WebConfig, checkpoint_id: str) -> CheckpointSlot:
    for slot in _checkpoint_slots(config):
        if slot.checkpoint_id == checkpoint_id:
            return slot
    raise HTTPException(status_code=400, detail=f"Unknown checkpoint: {checkpoint_id}")


def _resolved_checkpoint_paths(
    config: WebConfig,
    *,
    before_checkpoint_resolve: Callable[[], None] | None = None,
) -> dict[str, Path | None]:
    checkpoint_paths: dict[str, Path | None] = {}
    for slot in _checkpoint_slots(config):
        try:
            checkpoint_paths[slot.checkpoint_id] = _resolve_checkpoint_path(
                slot,
                before_checkpoint_resolve=before_checkpoint_resolve,
            )
        except FileNotFoundError:
            checkpoint_paths[slot.checkpoint_id] = None
    return checkpoint_paths


def _resolve_checkpoint_path(
    slot: CheckpointSlot,
    *,
    before_checkpoint_resolve: Callable[[], None] | None = None,
) -> Path:
    if before_checkpoint_resolve is not None:
        before_checkpoint_resolve()

    if slot.checkpoint_glob:
        checkpoint_path = _latest_checkpoint_path(slot.checkpoint_glob)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No web app checkpoints match checkpoint_glob for "
                f"{slot.label}: {slot.checkpoint_glob}"
            )
        return checkpoint_path

    checkpoint_path = Path(slot.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Web app checkpoint not found for {slot.label}: {checkpoint_path}. "
            "Train a model or pass --checkpoint-path."
        )
    return checkpoint_path


def _latest_checkpoint_path(checkpoint_glob: str) -> Path | None:
    matches = [Path(path) for path in glob.glob(checkpoint_glob) if Path(path).is_file()]
    if not matches:
        return None
    return max(matches, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"-step-(\d+)(?:\.[^.]+)?$", path.name)
    step = int(match.group(1)) if match else -1
    return (step, path.stat().st_mtime, path.name)


def _checkpoint_cache_key(path: Path) -> tuple[Path, int, int]:
    stat = path.stat()
    return (path, stat.st_mtime_ns, stat.st_size)


def _prepare_generation(
    payload: GenerateRequest,
    samples: Sequence[SatelliteSample],
    config: WebConfig,
) -> PreparedGeneration:
    sample = _sample_at(samples, payload.index)
    seed = config.seed if payload.seed is None else payload.seed
    steps = config.steps if payload.steps is None else payload.steps
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be positive")

    mask_path = sample.mask_path
    temporary_mask_path: Path | None = None
    if payload.mask_image is not None:
        temporary_mask_path = _write_request_mask(payload.mask_image, config)
        mask_path = temporary_mask_path
    elif not mask_path.exists():
        temporary_mask_path = _write_prepared_mask(sample)
        mask_path = temporary_mask_path

    return PreparedGeneration(
        index=payload.index,
        sample=sample,
        seed=seed,
        steps=steps,
        mask_path=mask_path,
        coordinates=_sample_coordinates(sample),
        temporary_mask_path=temporary_mask_path,
        mask_edited=payload.mask_image is not None,
    )


def _run_generation(
    prepared: PreparedGeneration,
    config: WebConfig,
    loaded_generator: Any,
    *,
    progress: ProgressCallback | None = None,
    checkpoint_slot: CheckpointSlot | None = None,
) -> dict[str, Any]:
    try:
        generation_config = GenerationConfig(
            seed=prepared.seed,
            mask_path=str(prepared.mask_path),
            coordinates=prepared.coordinates,
            steps=prepared.steps,
            guidance_scale=config.guidance_scale,
        )
        if progress is None:
            generated = generate_image_with_model(loaded_generator, generation_config)
        else:
            generated = generate_image_with_model(
                loaded_generator,
                generation_config,
                progress=progress,
            )
    finally:
        if prepared.temporary_mask_path is not None:
            prepared.temporary_mask_path.unlink(missing_ok=True)

    png = _image_tensor_png(generated.image, config.display_size)
    checkpoint_path = getattr(
        generated,
        "checkpoint_path",
        getattr(loaded_generator, "checkpoint_path", ""),
    )
    return {
        "image": _data_uri(png, "image/png"),
        "checkpointPath": str(checkpoint_path),
        "checkpointId": (
            checkpoint_slot.checkpoint_id if checkpoint_slot is not None else PRIMARY_CHECKPOINT_ID
        ),
        "checkpointLabel": (
            checkpoint_slot.label if checkpoint_slot is not None else config.checkpoint_label
        ),
        "sample": _sample_payload(prepared.index, prepared.sample),
        "seed": prepared.seed,
        "steps": prepared.steps,
        "maskEdited": prepared.mask_edited,
    }


def _generate_stream(
    prepared: PreparedGeneration,
    config: WebConfig,
    generator: Callable[[], Any],
    *,
    checkpoint_slot: CheckpointSlot | None = None,
) -> Iterator[str]:
    events: Queue[dict[str, Any] | Exception | object] = Queue()

    def report_progress(step: int, steps: int) -> None:
        events.put(
            _checkpoint_event({"event": "step", "step": step, "steps": steps}, checkpoint_slot)
        )

    def run() -> None:
        try:
            result = _run_generation(
                prepared,
                config,
                generator(),
                progress=report_progress,
                checkpoint_slot=checkpoint_slot,
            )
            events.put(_checkpoint_event({"event": "complete", **result}, checkpoint_slot))
        except Exception as exc:
            events.put(exc)
        finally:
            events.put(_STREAM_DONE)

    yield _json_event(
        _checkpoint_event({"event": "start", "step": 0, "steps": prepared.steps}, checkpoint_slot)
    )
    Thread(target=run, daemon=True).start()

    while True:
        event = events.get()
        if event is _STREAM_DONE:
            break
        if isinstance(event, Exception):
            yield _json_event(_stream_error_payload(event))
        elif isinstance(event, dict):
            yield _json_event(event)
        else:
            raise RuntimeError(f"Unexpected stream event: {event!r}")


def _stream_error_payload(error: Exception) -> dict[str, Any]:
    detail = error.detail if isinstance(error, HTTPException) else str(error)
    return {"event": "error", "detail": detail}


def _checkpoint_event(
    payload: dict[str, Any],
    checkpoint_slot: CheckpointSlot | None,
) -> dict[str, Any]:
    if checkpoint_slot is None:
        return payload
    return {
        **payload,
        "checkpointId": checkpoint_slot.checkpoint_id,
        "checkpointLabel": checkpoint_slot.label,
    }


def _json_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _mask_preview_png(sample: SatelliteSample, config: WebConfig) -> bytes:
    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    with _sample_mask_image(sample) as mask:
        resized = mask.resize(
            (config.display_size, config.display_size),
            Image.Resampling.NEAREST,
        )
        labels = mask_to_class_ids(resized, model_config)

    image = Image.new("RGB", (config.display_size, config.display_size))
    image.putdata([MASK_COLORS[label] for label in labels])
    return _pil_png(image)


def _image_preview_png(sample: SatelliteSample, config: WebConfig) -> bytes:
    with _sample_image(sample) as image:
        preview = image.convert("RGB").resize(
            (config.display_size, config.display_size),
            Image.Resampling.BILINEAR,
        )
    return _pil_png(preview)


def _sample_mask_image(sample: SatelliteSample) -> Image.Image:
    if sample.mask_path.exists():
        return Image.open(sample.mask_path)
    return _prepared_mask_image(_load_sample_prepared_data(sample))


def _sample_image(sample: SatelliteSample) -> Image.Image:
    if sample.image_path.exists():
        return Image.open(sample.image_path)
    return _prepared_image(_load_sample_prepared_data(sample))


def _write_prepared_mask(sample: SatelliteSample) -> Path:
    mask = _prepared_mask_image(_load_sample_prepared_data(sample))
    with NamedTemporaryFile(
        prefix="sim-satellite-prepared-mask-",
        suffix=".png",
        delete=False,
    ) as handle:
        path = Path(handle.name)
    try:
        mask.save(path)
    finally:
        mask.close()
    return path


def _sample_coordinates(sample: SatelliteSample) -> tuple[float, float] | None:
    if not sample.prepared_path.exists():
        return None
    return prepared_coordinates_tuple(_load_sample_prepared_data(sample))


def _load_sample_prepared_data(sample: SatelliteSample) -> PreparedSample:
    import torch

    return load_prepared_sample(sample.prepared_path, torch=torch)


def _prepared_mask_image(prepared: PreparedSample) -> Image.Image:
    width, height = prepared.size
    values = prepared.mask.byte().contiguous().view(-1).tolist()
    image = Image.new("L", (width, height))
    image.putdata(values)
    return image


def _prepared_image(prepared: PreparedSample) -> Image.Image:
    width, height = prepared.size
    image_data = prepared.image.byte().contiguous().numpy()
    image = Image.fromarray(image_data, mode="RGB")
    if image.size != (width, height):
        raise RuntimeError("Prepared image tensor shape does not match prepared image size")
    return image


def _write_request_mask(mask_image: str, config: WebConfig) -> Path:
    labels = _request_mask_labels(mask_image, config)
    with NamedTemporaryFile(
        prefix="sim-satellite-mask-",
        suffix=".png",
        delete=False,
    ) as handle:
        path = Path(handle.name)
    try:
        labels.save(path)
    finally:
        labels.close()
    return path


def _request_mask_labels(mask_image: str, config: WebConfig) -> Image.Image:
    image = _decode_mask_data_uri(mask_image, config)
    try:
        expected_size = (config.display_size, config.display_size)
        if image.size != expected_size:
            raise HTTPException(
                status_code=400,
                detail=(f"mask image must be {config.display_size}x{config.display_size} pixels"),
            )

        try:
            class_ids = mask_to_class_ids(
                image,
                ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        labels = Image.new("L", image.size)
        labels.putdata(class_ids)
        return labels
    finally:
        image.close()


def _decode_mask_data_uri(mask_image: str, config: WebConfig) -> Image.Image:
    if not mask_image.startswith(MASK_DATA_URI_PREFIX):
        raise HTTPException(status_code=400, detail="maskImage must be a PNG data URI")
    if len(mask_image) > _max_mask_image_uri_length(config):
        raise HTTPException(status_code=400, detail="maskImage is too large")

    try:
        body = base64.b64decode(mask_image[len(MASK_DATA_URI_PREFIX) :], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="maskImage is not valid base64") from exc

    try:
        with Image.open(BytesIO(body)) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=400, detail="maskImage is not a valid PNG") from exc


def _max_mask_image_uri_length(config: WebConfig) -> int:
    return len(MASK_DATA_URI_PREFIX) + max(4096, config.display_size * config.display_size * 6)


def _image_tensor_png(image: Any, display_size: int) -> bytes:
    tensor = image.detach().cpu()[0].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).contiguous()
    height = int(tensor.shape[0])
    width = int(tensor.shape[1])
    pil_image = Image.fromarray(tensor.numpy(), mode="RGB")
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


def _build_click_command() -> click.Command:
    @click.command("app", help="Launch the local mask-conditioned generation app.")
    @click.option("--checkpoint-path", default=WebConfig.checkpoint_path, show_default=True)
    @click.option("--checkpoint-glob", default=WebConfig.checkpoint_glob, show_default=True)
    @click.option(
        "--comparison-checkpoint-path",
        default=WebConfig.comparison_checkpoint_path,
        show_default=True,
    )
    @click.option(
        "--comparison-checkpoint-glob",
        default=WebConfig.comparison_checkpoint_glob,
        show_default=True,
    )
    @click.option("--manifest-path", default=WebConfig.manifest_path, show_default=True)
    @click.option("--split", default=WebConfig.split, show_default=True)
    @click.option("--device", default=WebConfig.device, show_default=True)
    @click.option("--steps", default=WebConfig.steps, show_default=True)
    @click.option("--guidance-scale", default=WebConfig.guidance_scale, show_default=True)
    @click.option("--seed", default=WebConfig.seed, show_default=True)
    @click.option("--host", default=WebConfig.host, show_default=True)
    @click.option("--port", default=WebConfig.port, show_default=True, type=int)
    @click.option("--max-samples", default=WebConfig.max_samples, show_default=True)
    def command(
        checkpoint_path: str,
        checkpoint_glob: str,
        comparison_checkpoint_path: str,
        comparison_checkpoint_glob: str,
        manifest_path: str,
        split: str,
        device: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        host: str,
        port: int,
        max_samples: int,
    ) -> None:
        serve(
            WebConfig(
                checkpoint_path=checkpoint_path,
                checkpoint_glob=checkpoint_glob,
                comparison_checkpoint_path=comparison_checkpoint_path,
                comparison_checkpoint_glob=comparison_checkpoint_glob,
                manifest_path=manifest_path,
                split=split,
                device=device,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                host=host,
                port=port,
                max_samples=max_samples,
            )
        )

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
