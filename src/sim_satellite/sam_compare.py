"""Run and compare baseline SAM2 and trained SAM2 proposal quality.

Data used by this harness:

- `data/openearthmap/manifest.jsonl`: labeled OpenEarthMap image/mask pairs.
  This is the apples-to-apples evaluation set for baseline SAM vs trained SAM.
- `data/openearthmap/unlabeled-images.jsonl`: OpenEarthMap images that were
  present in the archive but had no matching mask. Use this for proposal
  generation and pseudo-label inspection after a trained model wins on labeled
  recall.

The evaluator is class-agnostic. For every non-empty OpenEarthMap class mask, it
finds the best IoU among SAM proposals for the same source image. A trained SAM
model is better only if it improves these held-out labeled metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.sam_records import (
    SamMaskRecord,
    image_key,
    infer_region_key,
    read_sam_image_records,
    records_from_jsonl,
    write_sam_mask_records,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_BASELINE_MODEL_ID = "facebook/sam2.1-hiera-small"
ProgressReporter = Callable[[str], None]
GeneratorFactory = Callable[["SamProposalConfig"], Any]


@dataclass(frozen=True)
class SamProposalConfig:
    output_dir: str = "data/sam/proposals"
    model_id: str = DEFAULT_BASELINE_MODEL_ID
    model_label: str = "baseline"
    manifest_path: str | None = "data/openearthmap/manifest.jsonl"
    images_manifest_path: str | None = None
    images_dir: str | None = None
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    min_mask_region_area: int = 64
    max_images: int | None = None


@dataclass(frozen=True)
class SamProposalResult:
    output_dir: Path
    records_path: Path
    images: int
    proposals: int
    model_id: str
    model_label: str


@dataclass(frozen=True)
class SamProposalEvaluationResult:
    proposals_path: Path
    manifest_path: Path
    ground_truth_masks: int
    proposal_masks: int
    mean_best_iou: float
    recall_by_threshold: dict[str, float]
    class_mean_best_iou: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "proposals_path": str(self.proposals_path),
            "manifest_path": str(self.manifest_path),
            "ground_truth_masks": self.ground_truth_masks,
            "proposal_masks": self.proposal_masks,
            "mean_best_iou": self.mean_best_iou,
            "recall_by_threshold": self.recall_by_threshold,
            "class_mean_best_iou": self.class_mean_best_iou,
        }


@dataclass(frozen=True)
class SamComparisonResult:
    baseline: SamProposalEvaluationResult
    trained: SamProposalEvaluationResult

    def to_json(self) -> dict[str, object]:
        baseline = self.baseline.to_json()
        trained = self.trained.to_json()
        return {
            "baseline": baseline,
            "trained": trained,
            "delta": {
                "mean_best_iou": self.trained.mean_best_iou - self.baseline.mean_best_iou,
                "recall_by_threshold": {
                    key: self.trained.recall_by_threshold[key]
                    - self.baseline.recall_by_threshold.get(key, 0.0)
                    for key in self.trained.recall_by_threshold
                },
            },
        }


@dataclass(frozen=True)
class _ProposalImage:
    image_path: Path
    split: str
    source: str
    region_key: str


def generate_sam_proposals(
    config: SamProposalConfig,
    *,
    generator_factory: GeneratorFactory | None = None,
    progress: ProgressReporter | None = None,
) -> SamProposalResult:
    images = _proposal_images(config)
    output_dir = Path(config.output_dir)
    mask_dir = output_dir / "masks"
    records_path = output_dir / "proposals.jsonl"
    generator = (generator_factory or _load_sam2_generator)(config)
    records: list[SamMaskRecord] = []

    for image_index, proposal_image in enumerate(images, start=1):
        with Image.open(proposal_image.image_path) as image:
            pil_image = image.convert("RGB")
            width, height = pil_image.size
            proposals = generator.generate(_image_array(pil_image))

        for proposal_index, proposal in enumerate(proposals):
            mask = _proposal_segmentation(proposal)
            area = int(_mask_area(mask))
            if area < config.min_mask_region_area:
                continue
            mask_path = mask_dir / _proposal_mask_name(
                image_path=proposal_image.image_path,
                image_index=image_index,
                proposal_index=proposal_index,
            )
            _write_binary_mask(mask_path, mask)
            records.append(
                SamMaskRecord(
                    image_path=proposal_image.image_path,
                    mask_path=mask_path,
                    split=proposal_image.split,
                    source=proposal_image.source,
                    bbox_xyxy=_proposal_bbox_xyxy(proposal, mask),
                    area=area,
                    image_size=(width, height),
                    region_key=proposal_image.region_key,
                    image_key=image_key(proposal_image.image_path),
                    label_source=f"sam2-{config.model_label}-proposal",
                    model_id=config.model_id,
                    predicted_iou=_optional_float(proposal.get("predicted_iou")),
                    stability_score=_optional_float(proposal.get("stability_score")),
                    crop_box_xywh=_optional_box(proposal.get("crop_box")),
                    point_coords_xy=_point_coords(proposal.get("point_coords")),
                )
            )
        if _should_report(image_index, len(images)):
            _report(progress, f"Processed {image_index}/{len(images)} images")

    write_sam_mask_records(records, records_path)
    _write_proposal_config(output_dir / "config.json", config)
    return SamProposalResult(
        output_dir=output_dir,
        records_path=records_path,
        images=len(images),
        proposals=len(records),
        model_id=config.model_id,
        model_label=config.model_label,
    )


def evaluate_sam_proposals(
    *,
    manifest_path: str | Path,
    proposals_path: str | Path,
    output_path: str | Path | None = None,
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    min_mask_pixels: int = 64,
    include_unlabeled: bool = False,
) -> SamProposalEvaluationResult:
    if min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be at least 1")
    if not iou_thresholds:
        raise ValueError("iou_thresholds must not be empty")

    import torch

    manifest = Path(manifest_path)
    proposals = Path(proposals_path)
    proposal_records = records_from_jsonl(proposals)
    proposal_masks = _load_proposal_masks(proposal_records, proposals.parent, torch=torch)
    best_ious: list[float] = []
    class_ious: dict[int, list[float]] = defaultdict(list)

    for sample in read_manifest(manifest):
        prepared = load_prepared_sample(sample.prepared_path, torch=torch)
        labels = prepared.mask.long()
        candidate_masks = proposal_masks.get(image_key(sample.image_path), ())
        for class_id in range(len(OPENEARTHMAP_CLASSES)):
            if class_id == 0 and not include_unlabeled:
                continue
            target = labels == class_id
            if int(target.sum().item()) < min_mask_pixels:
                continue
            best_iou = max(
                (_mask_iou(target, proposal) for proposal in candidate_masks),
                default=0.0,
            )
            best_ious.append(best_iou)
            class_ious[class_id].append(best_iou)

    mean_best_iou = sum(best_ious) / len(best_ious) if best_ious else 0.0
    recall = {
        f"{threshold:.2f}": (
            sum(1 for value in best_ious if value >= threshold) / len(best_ious)
            if best_ious
            else 0.0
        )
        for threshold in iou_thresholds
    }
    result = SamProposalEvaluationResult(
        proposals_path=proposals,
        manifest_path=manifest,
        ground_truth_masks=len(best_ious),
        proposal_masks=sum(len(masks) for masks in proposal_masks.values()),
        mean_best_iou=mean_best_iou,
        recall_by_threshold=recall,
        class_mean_best_iou={
            OPENEARTHMAP_CLASSES[class_id]: sum(values) / len(values)
            for class_id, values in sorted(class_ious.items())
        },
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def compare_sam_proposals(
    *,
    manifest_path: str | Path,
    baseline_proposals_path: str | Path,
    trained_proposals_path: str | Path,
    output_path: str | Path | None = None,
    min_mask_pixels: int = 64,
) -> SamComparisonResult:
    baseline = evaluate_sam_proposals(
        manifest_path=manifest_path,
        proposals_path=baseline_proposals_path,
        min_mask_pixels=min_mask_pixels,
    )
    trained = evaluate_sam_proposals(
        manifest_path=manifest_path,
        proposals_path=trained_proposals_path,
        min_mask_pixels=min_mask_pixels,
    )
    result = SamComparisonResult(baseline=baseline, trained=trained)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def compare_sam_models(
    *,
    manifest_path: str | Path,
    baseline_model_id: str,
    trained_model_id: str,
    output_dir: str | Path,
    device: str = "cuda",
    max_images: int | None = None,
    min_mask_pixels: int = 64,
    generator_factory: GeneratorFactory | None = None,
    progress: ProgressReporter | None = None,
) -> SamComparisonResult:
    output = Path(output_dir)
    baseline = generate_sam_proposals(
        SamProposalConfig(
            output_dir=str(output / "baseline"),
            model_id=baseline_model_id,
            model_label="baseline",
            manifest_path=str(manifest_path),
            device=device,
            max_images=max_images,
            min_mask_region_area=min_mask_pixels,
        ),
        generator_factory=generator_factory,
        progress=progress,
    )
    trained = generate_sam_proposals(
        SamProposalConfig(
            output_dir=str(output / "trained"),
            model_id=trained_model_id,
            model_label="trained",
            manifest_path=str(manifest_path),
            device=device,
            max_images=max_images,
            min_mask_region_area=min_mask_pixels,
        ),
        generator_factory=generator_factory,
        progress=progress,
    )
    return compare_sam_proposals(
        manifest_path=manifest_path,
        baseline_proposals_path=baseline.records_path,
        trained_proposals_path=trained.records_path,
        output_path=output / "comparison.json",
        min_mask_pixels=min_mask_pixels,
    )


def _proposal_images(config: SamProposalConfig) -> tuple[_ProposalImage, ...]:
    sources = sum(
        value is not None
        for value in (config.manifest_path, config.images_manifest_path, config.images_dir)
    )
    if sources != 1:
        raise ValueError("Use exactly one of manifest_path, images_manifest_path, or images_dir")

    images: list[_ProposalImage] = []
    if config.manifest_path:
        seen: set[Path] = set()
        for sample in read_manifest(config.manifest_path):
            if sample.image_path in seen:
                continue
            seen.add(sample.image_path)
            images.append(
                _ProposalImage(
                    image_path=sample.image_path,
                    split=sample.split,
                    source=sample.source,
                    region_key=infer_region_key(sample.image_path),
                )
            )
    elif config.images_manifest_path:
        images = [
            _ProposalImage(
                image_path=record.image_path,
                split=record.split,
                source=record.source,
                region_key=record.region_key,
            )
            for record in read_sam_image_records(config.images_manifest_path)
        ]
    else:
        root = Path(config.images_dir or "")
        for path in sorted(
            item for item in root.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
        ):
            images.append(
                _ProposalImage(
                    image_path=path,
                    split="unlabeled",
                    source="unlabeled-satellite",
                    region_key=infer_region_key(path),
                )
            )

    if config.max_images is not None:
        if config.max_images < 1:
            raise ValueError("max_images must be at least 1 when provided")
        images = images[: config.max_images]
    return tuple(images)


def _load_sam2_generator(config: SamProposalConfig) -> Any:
    import importlib

    try:
        module = importlib.import_module("sam2.automatic_mask_generator")
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 is not installed. Install Meta's sam2 package before running SAM proposals."
        ) from exc

    model_path = Path(config.model_id)
    if model_path.exists():
        model = _load_sam2_model_from_checkpoint(model_path, device=config.device)
        return module.SAM2AutomaticMaskGenerator(
            model,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=config.crop_n_layers,
            min_mask_region_area=config.min_mask_region_area,
        )

    return module.SAM2AutomaticMaskGenerator.from_pretrained(
        config.model_id,
        device=config.device,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        crop_n_layers=config.crop_n_layers,
        min_mask_region_area=config.min_mask_region_area,
    )


def _load_sam2_model_from_checkpoint(checkpoint_path: Path, *, device: str) -> Any:
    try:
        import torch
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 is not installed. Install Meta's sam2 package before loading checkpoints."
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config_file = checkpoint.get("sam2_config_file", "configs/sam2.1/sam2.1_hiera_s.yaml")
    if not isinstance(config_file, str):
        raise RuntimeError(f"SAM2 checkpoint has invalid sam2_config_file: {checkpoint_path}")
    return build_sam2(
        config_file=config_file,
        ckpt_path=str(checkpoint_path),
        device=device,
        mode="eval",
    )


def _image_array(image: Image.Image) -> Any:
    import numpy as np

    return np.asarray(image)


def _proposal_segmentation(proposal: Mapping[str, Any]) -> Any:
    if "segmentation" not in proposal:
        raise RuntimeError("SAM proposal is missing segmentation")
    return proposal["segmentation"]


def _mask_area(mask: Any) -> int:
    if hasattr(mask, "sum"):
        return int(mask.sum())
    return sum(1 for row in mask for value in row if value)


def _write_binary_mask(path: Path, mask: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(mask, dtype=np.uint8) * 255
    height, width = values.shape
    Image.frombytes("L", (width, height), values.tobytes()).save(path)


def _proposal_mask_name(*, image_path: Path, image_index: int, proposal_index: int) -> str:
    fingerprint = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    return f"{image_index:06d}-{proposal_index:04d}-{image_path.stem}-{fingerprint}.png"


def _proposal_bbox_xyxy(proposal: Mapping[str, Any], mask: Any) -> tuple[int, int, int, int]:
    bbox = proposal.get("bbox")
    if bbox is not None:
        x, y, width, height = [int(round(float(value))) for value in bbox]
        return x, y, max(x, x + width - 1), max(y, y + height - 1)

    import numpy as np

    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        return 0, 0, 0, 0
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value)
    if len(values) != 4:
        raise RuntimeError(f"Expected a 4-value box, got {value}")
    return values


def _point_coords(value: Any) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    return tuple((float(point[0]), float(point[1])) for point in value)


def _load_proposal_masks(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    torch: Any,
) -> dict[str, tuple[Any, ...]]:
    masks: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        key = record.get("image_key")
        mask_path = record.get("mask")
        if not isinstance(key, str) or not isinstance(mask_path, str):
            raise RuntimeError("Proposal records must include image_key and mask")
        masks[key].append(_load_binary_mask(_resolve(root, mask_path), torch=torch))
    return {key: tuple(value) for key, value in masks.items()}


def _load_binary_mask(path: Path, *, torch: Any) -> Any:
    with Image.open(path) as image:
        mask = image.convert("L")
        values = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
        return values.view(mask.height, mask.width) > 0


def _mask_iou(left: Any, right: Any) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(
            f"Mask shapes do not match for IoU: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    intersection = (left & right).sum().item()
    union = (left | right).sum().item()
    return 0.0 if union == 0 else float(intersection / union)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _write_proposal_config(path: Path, config: SamProposalConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == 1 or current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _build_propose_click_command():
    import click

    @click.command("sam-propose", help="Generate SAM2 proposals from a model id or checkpoint.")
    @click.option("--manifest", type=click.Path(path_type=Path), default=None)
    @click.option("--images-manifest", type=click.Path(path_type=Path), default=None)
    @click.option("--images-dir", type=click.Path(path_type=Path), default=None)
    @click.option(
        "--output-dir", type=click.Path(path_type=Path), default=Path("data/sam/proposals")
    )
    @click.option("--model-id", default=DEFAULT_BASELINE_MODEL_ID, show_default=True)
    @click.option("--model-label", default="baseline", show_default=True)
    @click.option("--device", default="cuda", show_default=True)
    @click.option("--max-images", type=int, default=None)
    def command(
        manifest: Path | None,
        images_manifest: Path | None,
        images_dir: Path | None,
        output_dir: Path,
        model_id: str,
        model_label: str,
        device: str,
        max_images: int | None,
    ) -> None:
        try:
            result = generate_sam_proposals(
                SamProposalConfig(
                    output_dir=str(output_dir),
                    model_id=model_id,
                    model_label=model_label,
                    manifest_path=str(manifest) if manifest is not None else None,
                    images_manifest_path=(
                        str(images_manifest) if images_manifest is not None else None
                    ),
                    images_dir=str(images_dir) if images_dir is not None else None,
                    device=device,
                    max_images=max_images,
                ),
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"records_path={result.records_path}")
        click.echo(f"images={result.images}")
        click.echo(f"proposals={result.proposals}")

    return command


def _build_evaluate_click_command():
    import click

    @click.command("sam-eval", help="Evaluate SAM proposals on labeled OpenEarthMap masks.")
    @click.option(
        "--manifest",
        type=click.Path(path_type=Path),
        default=Path("data/openearthmap/manifest.jsonl"),
    )
    @click.option("--proposals", type=click.Path(path_type=Path), required=True)
    @click.option("--output", type=click.Path(path_type=Path), default=None)
    @click.option("--min-mask-pixels", type=int, default=64, show_default=True)
    def command(
        manifest: Path,
        proposals: Path,
        output: Path | None,
        min_mask_pixels: int,
    ) -> None:
        try:
            result = evaluate_sam_proposals(
                manifest_path=manifest,
                proposals_path=proposals,
                output_path=output,
                min_mask_pixels=min_mask_pixels,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


def _build_compare_click_command():
    import click

    @click.command("sam-compare", help="Compare baseline and trained SAM proposal files.")
    @click.option(
        "--manifest",
        type=click.Path(path_type=Path),
        default=Path("data/openearthmap/manifest.jsonl"),
    )
    @click.option("--baseline-proposals", type=click.Path(path_type=Path), required=True)
    @click.option("--trained-proposals", type=click.Path(path_type=Path), required=True)
    @click.option(
        "--output", type=click.Path(path_type=Path), default=Path("data/sam/comparison.json")
    )
    @click.option("--min-mask-pixels", type=int, default=64, show_default=True)
    def command(
        manifest: Path,
        baseline_proposals: Path,
        trained_proposals: Path,
        output: Path,
        min_mask_pixels: int,
    ) -> None:
        try:
            result = compare_sam_proposals(
                manifest_path=manifest,
                baseline_proposals_path=baseline_proposals,
                trained_proposals_path=trained_proposals,
                output_path=output,
                min_mask_pixels=min_mask_pixels,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


def _build_compare_models_click_command():
    import click

    @click.command(
        "sam-compare-models",
        help="Run baseline and trained SAM2 models on labeled OpenEarthMap and compare recall.",
    )
    @click.option(
        "--manifest",
        type=click.Path(path_type=Path),
        default=Path("data/openearthmap/manifest.jsonl"),
    )
    @click.option("--baseline-model-id", default=DEFAULT_BASELINE_MODEL_ID, show_default=True)
    @click.option("--trained-model-id", required=True)
    @click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/sam/compare"))
    @click.option("--device", default="cuda", show_default=True)
    @click.option("--max-images", type=int, default=None)
    @click.option("--min-mask-pixels", type=int, default=64, show_default=True)
    def command(
        manifest: Path,
        baseline_model_id: str,
        trained_model_id: str,
        output_dir: Path,
        device: str,
        max_images: int | None,
        min_mask_pixels: int,
    ) -> None:
        try:
            result = compare_sam_models(
                manifest_path=manifest,
                baseline_model_id=baseline_model_id,
                trained_model_id=trained_model_id,
                output_dir=output_dir,
                device=device,
                max_images=max_images,
                min_mask_pixels=min_mask_pixels,
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


propose_cli = _build_propose_click_command()
evaluate_cli = _build_evaluate_click_command()
compare_cli = _build_compare_click_command()
compare_models_cli = _build_compare_models_click_command()
