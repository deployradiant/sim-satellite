"""FastAPI server for the local mask-conditioned generation website."""

from __future__ import annotations

import base64
import binascii
import glob
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Annotated, Any, Literal

import click
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.prepared import PreparedSample, load_prepared_sample
from sim_satellite.data.sam_records import (
    SamImageRecord,
    image_key,
    read_sam_image_records,
    records_from_jsonl,
)
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.infer import GenerationConfig, generate_image_with_model, load_generator
from sim_satellite.model import ModelConfig
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DISPLAY_SIZE = 256
DEFAULT_MAX_SAMPLES = 32
DEFAULT_WEB_CHECKPOINT_PATH = "artifacts/tiny-sim-satellite-modal.pt"
DEFAULT_EVALUATION_LOG_PATH = "artifacts/web-evaluation-votes.jsonl"
DEFAULT_SAM_UNLABELED_MANIFEST_PATH = "data/openearthmap/unlabeled-images.jsonl"
DEFAULT_SAM_COMPARISON_PATH = "data/sam/compare/comparison.json"
DEFAULT_SAM_LABELED_BASELINE_PROPOSALS_PATH = "data/sam/compare/baseline/proposals.jsonl"
DEFAULT_SAM_LABELED_TRAINED_PROPOSALS_PATH = "data/sam/compare/trained/proposals.jsonl"
DEFAULT_SAM_UNLABELED_BASELINE_PROPOSALS_PATH = (
    "data/sam/openearthmap-unlabeled-baseline/proposals.jsonl"
)
DEFAULT_SAM_UNLABELED_TRAINED_PROPOSALS_PATH = (
    "data/sam/openearthmap-unlabeled-trained/proposals.jsonl"
)
PRIMARY_CHECKPOINT_ID = "primary"
COMPARISON_CHECKPOINT_ID = "comparison"
STATIC_DIR = Path(__file__).with_name("web_static")
INDEX_PATH = STATIC_DIR / "index.html"
SAM_INDEX_PATH = STATIC_DIR / "sam.html"
MASK_DATA_URI_PREFIX = "data:image/png;base64,"
MASK_COLORS = tuple(OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES)
SAM_LABEL_ALPHA = 132
SAM_PROPOSAL_ALPHA = 118
SAM_PROPOSAL_ALPHA_LUT = (0,) + (SAM_PROPOSAL_ALPHA,) * 255
SAM_PROPOSAL_COLORS = (
    (0, 192, 255),
    (255, 86, 124),
    (94, 226, 148),
    (255, 186, 73),
    (172, 124, 255),
    (39, 214, 196),
)
STREAM_MEDIA_TYPE = "application/x-ndjson"
_STREAM_DONE = object()
ProgressCallback = Callable[[int, int], None]
SamDatasetName = Literal["labeled", "unlabeled"]
SamOverlaySource = Literal["label", "baseline", "trained"]
SamDatasetQuery = Annotated[SamDatasetName, Query()]
SamOverlaySourceQuery = Annotated[SamOverlaySource, Query()]


@dataclass(frozen=True)
class WebConfig:
    checkpoint_path: str = DEFAULT_WEB_CHECKPOINT_PATH
    checkpoint_glob: str = ""
    checkpoint_label: str = "Primary checkpoint"
    comparison_checkpoint_path: str = ""
    comparison_checkpoint_glob: str = ""
    comparison_checkpoint_label: str = "Comparison checkpoint"
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
    evaluation_log_path: str = DEFAULT_EVALUATION_LOG_PATH
    sam_unlabeled_manifest_path: str = DEFAULT_SAM_UNLABELED_MANIFEST_PATH
    sam_comparison_path: str = DEFAULT_SAM_COMPARISON_PATH
    sam_labeled_baseline_proposals_path: str = DEFAULT_SAM_LABELED_BASELINE_PROPOSALS_PATH
    sam_labeled_trained_proposals_path: str = DEFAULT_SAM_LABELED_TRAINED_PROPOSALS_PATH
    sam_unlabeled_baseline_proposals_path: str = DEFAULT_SAM_UNLABELED_BASELINE_PROPOSALS_PATH
    sam_unlabeled_trained_proposals_path: str = DEFAULT_SAM_UNLABELED_TRAINED_PROPOSALS_PATH

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
        if not self.evaluation_log_path:
            raise ValueError("evaluation_log_path must not be empty")
        if not self.sam_unlabeled_manifest_path:
            raise ValueError("sam_unlabeled_manifest_path must not be empty")
        if not self.sam_comparison_path:
            raise ValueError("sam_comparison_path must not be empty")


class GenerateRequest(BaseModel):
    index: int = 0
    seed: int | None = None
    steps: int | None = None
    checkpoint: str = PRIMARY_CHECKPOINT_ID
    mask_image: str | None = Field(default=None, alias="maskImage")


class EvaluationImage(BaseModel):
    kind: Literal["real", "generated"]
    label: str | None = None
    checkpoint_id: str | None = Field(default=None, alias="checkpointId")
    checkpoint_label: str | None = Field(default=None, alias="checkpointLabel")
    seed: int | None = None


class EvaluationVote(BaseModel):
    task: Literal["preference", "real_fake"]
    trial_id: str = Field(alias="trialId")
    session_id: str = Field(alias="sessionId")
    sample_index: int = Field(alias="sampleIndex", ge=0)
    sample_name: str | None = Field(default=None, alias="sampleName")
    seed: int | None = None
    choice: Literal["left", "right", "tie", "bad_pair", "real", "fake", "skip"]
    confidence: int = Field(ge=1, le=5)
    left: EvaluationImage | None = None
    right: EvaluationImage | None = None
    shown: EvaluationImage | None = None

    @model_validator(mode="after")
    def validate_task_payload(self) -> EvaluationVote:
        if self.task == "preference":
            if self.left is None or self.right is None or self.shown is not None:
                raise ValueError("preference votes require left and right images only")
            if self.choice not in {"left", "right", "tie", "bad_pair"}:
                raise ValueError("preference choice must be left, right, tie, or bad_pair")
        else:
            if self.shown is None or self.left is not None or self.right is not None:
                raise ValueError("real_fake votes require one shown image only")
            if self.choice not in {"real", "fake", "skip"}:
                raise ValueError("real_fake choice must be real, fake, or skip")
        return self


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
    sam_unlabeled_images = _load_sam_unlabeled_images(config)
    loaded_generators: dict[str, tuple[tuple[Path, int, int], Any]] = {}
    app = FastAPI(title="Sim Satellite", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

    @app.get("/sam", include_in_schema=False)
    def sam_index() -> FileResponse:
        return FileResponse(SAM_INDEX_PATH)

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

    @app.get("/api/sam/state")
    def sam_state() -> dict[str, Any]:
        return _sam_state_payload(config, samples, sam_unlabeled_images)

    @app.get("/api/sam/image")
    def sam_image(
        dataset: SamDatasetQuery = "labeled",
        index: int = Query(default=0, ge=0),
    ) -> Response:
        image_path, _ = _sam_image_reference(samples, sam_unlabeled_images, dataset, index)
        return Response(
            content=_sam_image_preview_png(image_path, config),
            media_type="image/png",
        )

    @app.get("/api/sam/overlay")
    def sam_overlay(
        dataset: SamDatasetQuery = "labeled",
        source: SamOverlaySourceQuery = "baseline",
        index: int = Query(default=0, ge=0),
    ) -> Response:
        image_path, image_record_key = _sam_image_reference(
            samples,
            sam_unlabeled_images,
            dataset,
            index,
        )
        if source == "label":
            if dataset != "labeled":
                raise HTTPException(
                    status_code=400,
                    detail="Label overlays are only available for labeled samples",
                )
            sample = _sample_at(samples, index)
            content = _sam_label_overlay_png(sample, config)
        else:
            proposals_path = _sam_proposals_path(config, dataset, source)
            content = _sam_proposal_overlay_png(
                proposals_path,
                image_record_key=image_record_key,
                image_path=image_path,
                config=config,
            )
        return Response(content=content, media_type="image/png")

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

    @app.post("/api/evaluation/votes")
    def evaluation_vote(payload: EvaluationVote) -> dict[str, Any]:
        record = _evaluation_vote_record(payload)
        _append_evaluation_vote(record, config)
        return {
            "ok": True,
            "createdAtEpochUtc": record["createdAtEpochUtc"],
            "votes": _evaluation_vote_count(Path(config.evaluation_log_path)),
        }

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
    if not SAM_INDEX_PATH.exists():
        raise FileNotFoundError(f"SAM web app HTML not found: {SAM_INDEX_PATH}")

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
    return {
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
        "evaluation": {
            "logPath": config.evaluation_log_path,
            "votes": _evaluation_vote_count(Path(config.evaluation_log_path)),
        },
    }


def _load_sam_unlabeled_images(config: WebConfig) -> tuple[SamImageRecord, ...]:
    path = Path(config.sam_unlabeled_manifest_path)
    if not path.exists():
        return ()
    records = read_sam_image_records(path)
    return records[: config.max_samples]


def _sam_state_payload(
    config: WebConfig,
    samples: Sequence[SatelliteSample],
    unlabeled_images: Sequence[SamImageRecord],
) -> dict[str, Any]:
    return {
        "displaySize": config.display_size,
        "datasets": {
            "labeled": {
                "label": "Labeled OpenEarthMap",
                "manifestPath": config.manifest_path,
                "hasLabels": True,
                "samples": [_sample_payload(index, sample) for index, sample in enumerate(samples)],
            },
            "unlabeled": {
                "label": "Unlabeled OpenEarthMap",
                "manifestPath": config.sam_unlabeled_manifest_path,
                "manifestAvailable": Path(config.sam_unlabeled_manifest_path).exists(),
                "hasLabels": False,
                "samples": [
                    _sam_unlabeled_payload(index, record)
                    for index, record in enumerate(unlabeled_images)
                ],
            },
        },
        "proposalSources": {
            dataset: {
                source: _sam_proposal_source_payload(
                    _sam_proposals_path(config, dataset, source),
                    source=source,
                )
                for source in ("baseline", "trained")
            }
            for dataset in ("labeled", "unlabeled")
        },
        "comparison": _sam_comparison_payload(config),
        "classes": [
            {
                "id": index,
                "name": name,
                "rgb": OPENEARTHMAP_RGB[name],
            }
            for index, name in enumerate(OPENEARTHMAP_CLASSES)
        ],
        "commands": _sam_command_payload(config),
    }


def _sam_unlabeled_payload(index: int, record: SamImageRecord) -> dict[str, Any]:
    return {
        "index": index,
        "name": record.image_path.stem,
        "imagePath": str(record.image_path),
        "source": record.source,
        "split": record.split,
        "regionKey": record.region_key,
    }


def _sam_proposal_source_payload(path: Path, *, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "label": "SAM baseline" if source == "baseline" else "Trained SAM",
        "path": str(path),
        "available": path.exists(),
        "records": _jsonl_record_count(path),
    }


def _sam_comparison_payload(config: WebConfig) -> dict[str, Any]:
    path = Path(config.sam_comparison_path)
    payload: dict[str, Any] = {
        "path": str(path),
        "available": path.exists(),
    }
    if not path.exists():
        return payload
    try:
        payload["value"] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        payload["error"] = f"{path}: invalid JSON: {exc}"
    return payload


def _sam_command_payload(config: WebConfig) -> dict[str, str]:
    return {
        "trainingData": (
            "uv run sim-satellite sam-data "
            f"--manifest {config.manifest_path} "
            "--output-dir data/sam/openearthmap"
        ),
        "labeledComparison": (
            "uv run sim-satellite sam-compare-models "
            f"--manifest {config.manifest_path} "
            "--trained-model-id <trained-sam-model-or-checkpoint> "
            "--output-dir data/sam/compare "
            "--device cuda"
        ),
        "unlabeledBaseline": (
            "uv run sim-satellite sam-propose "
            f"--images-manifest {config.sam_unlabeled_manifest_path} "
            "--output-dir data/sam/openearthmap-unlabeled-baseline "
            "--model-id facebook/sam2.1-hiera-small "
            "--model-label baseline "
            "--device cuda"
        ),
        "unlabeledTrained": (
            "uv run sim-satellite sam-propose "
            f"--images-manifest {config.sam_unlabeled_manifest_path} "
            "--output-dir data/sam/openearthmap-unlabeled-trained "
            "--model-id <trained-sam-model-or-checkpoint> "
            "--model-label trained "
            "--device cuda"
        ),
    }


def _jsonl_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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


def _sam_unlabeled_at(
    records: Sequence[SamImageRecord],
    index: int,
) -> SamImageRecord:
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=400, detail=f"Unlabeled sample index out of range: {index}")
    return records[index]


def _sam_image_reference(
    samples: Sequence[SatelliteSample],
    unlabeled_images: Sequence[SamImageRecord],
    dataset: SamDatasetName,
    index: int,
) -> tuple[Path, str]:
    if dataset == "labeled":
        sample = _sample_at(samples, index)
        return sample.image_path, image_key(sample.image_path)
    record = _sam_unlabeled_at(unlabeled_images, index)
    return record.image_path, image_key(record.image_path)


def _sam_proposals_path(config: WebConfig, dataset: str, source: str) -> Path:
    if dataset == "labeled" and source == "baseline":
        return Path(config.sam_labeled_baseline_proposals_path)
    if dataset == "labeled" and source == "trained":
        return Path(config.sam_labeled_trained_proposals_path)
    if dataset == "unlabeled" and source == "baseline":
        return Path(config.sam_unlabeled_baseline_proposals_path)
    if dataset == "unlabeled" and source == "trained":
        return Path(config.sam_unlabeled_trained_proposals_path)
    raise HTTPException(status_code=400, detail=f"Unknown SAM proposal source: {dataset}/{source}")


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
    return {
        "image": _data_uri(png, "image/png"),
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


def _sam_image_preview_png(image_path: Path, config: WebConfig) -> bytes:
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"SAM image not found: {image_path}")
    with Image.open(image_path) as image:
        preview = image.convert("RGB").resize(
            (config.display_size, config.display_size),
            Image.Resampling.BILINEAR,
        )
    return _pil_png(preview)


def _sam_label_overlay_png(sample: SatelliteSample, config: WebConfig) -> bytes:
    size = (config.display_size, config.display_size)
    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    with _sample_mask_image(sample) as mask:
        resized = mask.resize(size, Image.Resampling.NEAREST)
        labels = mask_to_class_ids(resized, model_config)

    pixels = []
    for label in labels:
        if label == 0:
            pixels.append((0, 0, 0, 0))
        else:
            pixels.append((*MASK_COLORS[label], SAM_LABEL_ALPHA))

    overlay = Image.new("RGBA", size)
    overlay.putdata(pixels)
    return _pil_png(overlay)


def _sam_proposal_overlay_png(
    proposals_path: Path,
    *,
    image_record_key: str,
    image_path: Path,
    config: WebConfig,
) -> bytes:
    if not proposals_path.exists():
        raise HTTPException(status_code=404, detail=f"SAM proposals not found: {proposals_path}")
    overlay = Image.new("RGBA", (config.display_size, config.display_size), (0, 0, 0, 0))
    records = _sam_proposal_records_for_image(
        proposals_path,
        keys={image_record_key, image_key(image_path)},
    )
    for proposal_index, record in enumerate(records):
        mask_path = _sam_record_mask_path(proposals_path.parent, record)
        if mask_path is None or not mask_path.exists():
            continue
        color = _sam_record_overlay_color(record, fallback_index=proposal_index)
        with Image.open(mask_path) as mask:
            resized_mask = mask.convert("L").resize(
                (config.display_size, config.display_size),
                Image.Resampling.NEAREST,
            )
        layer = Image.new("RGBA", overlay.size, (*color, 0))
        layer.putalpha(resized_mask.point(SAM_PROPOSAL_ALPHA_LUT))
        overlay.alpha_composite(layer)
    return _pil_png(overlay)


def _sam_record_overlay_color(
    record: Mapping[str, Any],
    *,
    fallback_index: int,
) -> tuple[int, int, int]:
    for field in ("predicted_class_id", "class_id"):
        value = record.get(field)
        if isinstance(value, int) and 0 <= value < len(MASK_COLORS):
            return MASK_COLORS[value]
    return SAM_PROPOSAL_COLORS[fallback_index % len(SAM_PROPOSAL_COLORS)]


def _sam_proposal_records_for_image(
    proposals_path: Path,
    *,
    keys: set[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        record
        for record in records_from_jsonl(proposals_path)
        if isinstance(record.get("image_key"), str) and record["image_key"] in keys
    )


def _sam_record_mask_path(root: Path, record: Mapping[str, Any]) -> Path | None:
    value = record.get("mask")
    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


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


def _evaluation_vote_record(payload: EvaluationVote) -> dict[str, Any]:
    record = payload.model_dump(by_alias=True, exclude_none=True)
    if payload.task == "real_fake":
        if payload.shown is None:
            raise ValueError("real_fake votes require one shown image only")
        record["correctChoice"] = "real" if payload.shown.kind == "real" else "fake"
    return {"createdAtEpochUtc": int(time.time()), **record}


def _append_evaluation_vote(record: dict[str, Any], config: WebConfig) -> None:
    log_path = Path(config.evaluation_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _evaluation_vote_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    @click.option(
        "--evaluation-log-path",
        default=WebConfig.evaluation_log_path,
        show_default=True,
    )
    @click.option(
        "--sam-unlabeled-manifest-path",
        default=WebConfig.sam_unlabeled_manifest_path,
        show_default=True,
    )
    @click.option("--sam-comparison-path", default=WebConfig.sam_comparison_path, show_default=True)
    @click.option(
        "--sam-labeled-baseline-proposals-path",
        default=WebConfig.sam_labeled_baseline_proposals_path,
        show_default=True,
    )
    @click.option(
        "--sam-labeled-trained-proposals-path",
        default=WebConfig.sam_labeled_trained_proposals_path,
        show_default=True,
    )
    @click.option(
        "--sam-unlabeled-baseline-proposals-path",
        default=WebConfig.sam_unlabeled_baseline_proposals_path,
        show_default=True,
    )
    @click.option(
        "--sam-unlabeled-trained-proposals-path",
        default=WebConfig.sam_unlabeled_trained_proposals_path,
        show_default=True,
    )
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
        evaluation_log_path: str,
        sam_unlabeled_manifest_path: str,
        sam_comparison_path: str,
        sam_labeled_baseline_proposals_path: str,
        sam_labeled_trained_proposals_path: str,
        sam_unlabeled_baseline_proposals_path: str,
        sam_unlabeled_trained_proposals_path: str,
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
                evaluation_log_path=evaluation_log_path,
                sam_unlabeled_manifest_path=sam_unlabeled_manifest_path,
                sam_comparison_path=sam_comparison_path,
                sam_labeled_baseline_proposals_path=sam_labeled_baseline_proposals_path,
                sam_labeled_trained_proposals_path=sam_labeled_trained_proposals_path,
                sam_unlabeled_baseline_proposals_path=sam_unlabeled_baseline_proposals_path,
                sam_unlabeled_trained_proposals_path=sam_unlabeled_trained_proposals_path,
            )
        )

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
