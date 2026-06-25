"""Fine-tune SAM2 on SAM-style OpenEarthMap binary masks.

This is intentionally a small, explicit loop instead of a wrapper around a
missing training entrypoint in the packaged `sam2` wheel. It trains the SAM2 mask
decoder by default while keeping the image encoder frozen, using the binary
records produced by `sim-satellite sam-data`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.sam_records import records_from_jsonl

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"
DEFAULT_SAM2_CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_SEED = 2026
PROMPT_MODES = ("positive-point", "positive-points", "box", "box-points")
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class SamFineTuneConfig:
    records_path: str = "data/sam/openearthmap/records.jsonl"
    checkpoint_path: str = "data/sam/finetuned/sam2-openearthmap.pt"
    model_id: str = DEFAULT_SAM2_MODEL_ID
    device: str = "cuda"
    epochs: int = 1
    max_steps: int | None = 100
    max_records: int | None = None
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    dice_weight: float = 1.0
    iou_weight: float = 0.05
    prompt_mode: str = "positive-point"
    seed: int = DEFAULT_SEED
    train_image_encoder: bool = False
    train_prompt_encoder: bool = False
    checkpoint_every_steps: int | None = None


@dataclass(frozen=True)
class SamFineTuneResult:
    checkpoint_path: Path
    records: int
    steps: int
    final_loss: float
    mean_loss: float
    mean_mask_loss: float
    mean_dice_loss: float
    mean_iou_loss: float
    trainable_parameters: int

    def to_json(self) -> dict[str, object]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "records": self.records,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "mean_loss": self.mean_loss,
            "mean_mask_loss": self.mean_mask_loss,
            "mean_dice_loss": self.mean_dice_loss,
            "mean_iou_loss": self.mean_iou_loss,
            "trainable_parameters": self.trainable_parameters,
        }


def fine_tune_sam2(
    config: SamFineTuneConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SamFineTuneResult:
    _validate_config(config)
    records_path = Path(config.records_path)
    records = _training_records(records_path, max_records=config.max_records, seed=config.seed)
    if not records:
        raise ValueError(f"No SAM training records found in {records_path}")

    import torch
    import torch.nn.functional as F
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _seed_everything(config.seed, torch=torch)
    predictor = SAM2ImagePredictor.from_pretrained(config.model_id, device=config.device)
    model = predictor.model
    model.train()
    _configure_trainable_parameters(
        model,
        train_image_encoder=config.train_image_encoder,
        train_prompt_encoder=config.train_prompt_encoder,
    )
    if not config.train_image_encoder:
        model.image_encoder.eval()
    if not config.train_prompt_encoder:
        model.sam_prompt_encoder.eval()
    trainable_parameters = _count_trainable_parameters(model)
    if trainable_parameters == 0:
        raise RuntimeError("SAM2 fine-tune configuration left no trainable parameters")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses: list[float] = []
    mask_losses: list[float] = []
    dice_losses: list[float] = []
    iou_losses: list[float] = []
    final_loss = 0.0
    steps = 0
    shuffled = list(records)
    checkpoint_path = Path(config.checkpoint_path)

    _report(
        progress,
        f"Fine-tuning SAM2 on {len(records)} records; trainable_parameters={trainable_parameters}",
    )
    for epoch in range(config.epochs):
        random.Random(config.seed + epoch).shuffle(shuffled)
        for record in shuffled:
            if config.max_steps is not None and steps >= config.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)

            prediction = _forward_record(
                predictor,
                record,
                device=config.device,
                prompt_mode=config.prompt_mode,
                train_image_encoder=config.train_image_encoder,
                torch=torch,
            )
            target = _target_mask_tensor(
                record,
                orig_hw=prediction.orig_hw,
                device=config.device,
                torch=torch,
            )
            mask_loss = F.binary_cross_entropy_with_logits(prediction.mask_logits, target)
            dice_loss = _dice_loss(prediction.mask_logits, target, torch=torch)
            iou_target = _soft_iou(prediction.mask_logits.detach(), target, torch=torch)
            iou_prediction = prediction.iou_predictions[:, :1]
            iou_loss = F.mse_loss(iou_prediction, iou_target)
            loss = mask_loss + config.dice_weight * dice_loss + config.iou_weight * iou_loss
            loss.backward()
            optimizer.step()

            steps += 1
            final_loss = float(loss.detach().cpu().item())
            losses.append(final_loss)
            mask_losses.append(float(mask_loss.detach().cpu().item()))
            dice_losses.append(float(dice_loss.detach().cpu().item()))
            iou_losses.append(float(iou_loss.detach().cpu().item()))

            if _should_report(steps, config.max_steps, len(records)):
                _report(progress, f"step={steps} loss={final_loss:.4f}")
            if (
                config.checkpoint_every_steps is not None
                and steps % config.checkpoint_every_steps == 0
            ):
                _save_checkpoint(checkpoint_path, model, config, steps=steps, torch=torch)

        if config.max_steps is not None and steps >= config.max_steps:
            break

    if steps == 0:
        raise RuntimeError("SAM2 fine-tune completed zero optimization steps")

    _save_checkpoint(checkpoint_path, model, config, steps=steps, torch=torch)
    result = SamFineTuneResult(
        checkpoint_path=checkpoint_path,
        records=len(records),
        steps=steps,
        final_loss=final_loss,
        mean_loss=sum(losses) / len(losses),
        mean_mask_loss=sum(mask_losses) / len(mask_losses),
        mean_dice_loss=sum(dice_losses) / len(dice_losses),
        mean_iou_loss=sum(iou_losses) / len(iou_losses),
        trainable_parameters=trainable_parameters,
    )
    _write_metrics(checkpoint_path.with_suffix(".json"), config, result)
    return result


@dataclass(frozen=True)
class _SamPrediction:
    mask_logits: Any
    iou_predictions: Any
    orig_hw: tuple[int, int]


def _training_records(
    records_path: Path,
    *,
    max_records: int | None,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    root = records_path.parent
    records = [_resolve_record_paths(record, root) for record in records_from_jsonl(records_path)]
    if max_records is None:
        return tuple(records)
    if max_records < 1:
        raise ValueError("max_records must be at least 1 when provided")
    random.Random(seed).shuffle(records)
    return tuple(records[:max_records])


def _resolve_record_paths(record: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(record)
    for field in ("image", "mask", "prepared", "source_mask"):
        value = resolved.get(field)
        if isinstance(value, str):
            path = Path(value)
            resolved[field] = str(path if path.is_absolute() else root / path)
    return resolved


def _configure_trainable_parameters(
    model: Any,
    *,
    train_image_encoder: bool,
    train_prompt_encoder: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    _set_requires_grad(model.sam_mask_decoder, True)
    if train_prompt_encoder:
        _set_requires_grad(model.sam_prompt_encoder, True)
    if train_image_encoder:
        _set_requires_grad(model.image_encoder, True)


def _set_requires_grad(module: Any, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _forward_record(
    predictor: Any,
    record: Mapping[str, Any],
    *,
    device: str,
    prompt_mode: str,
    train_image_encoder: bool,
    torch: Any,
) -> _SamPrediction:
    image, orig_hw = _record_image(record)
    predictor.reset_predictor()
    predictor._orig_hw = [orig_hw]
    image_tensor = predictor._transforms(image)[None, ...].to(device)

    if train_image_encoder:
        features = _image_features(predictor, image_tensor, batch_size=1)
    else:
        with torch.no_grad():
            features = _image_features(predictor, image_tensor, batch_size=1)
    predictor._features = features
    predictor._is_image_set = True

    point_coords, point_labels, box = _record_prompt_arrays(record, prompt_mode=prompt_mode)
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        point_coords,
        point_labels,
        box,
        None,
        True,
    )
    del mask_input
    low_res_masks, iou_predictions = _predict_mask_logits(
        predictor,
        point_coords=unnorm_coords,
        point_labels=labels,
        boxes=unnorm_box,
    )
    mask_logits = predictor._transforms.postprocess_masks(low_res_masks, orig_hw)
    return _SamPrediction(
        mask_logits=mask_logits[:, :1],
        iou_predictions=iou_predictions,
        orig_hw=orig_hw,
    )


def _image_features(predictor: Any, image_tensor: Any, *, batch_size: int) -> dict[str, Any]:
    model = predictor.model
    backbone_out = model.forward_image(image_tensor)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(
            vision_feats[::-1],
            predictor._bb_feat_sizes[::-1],
            strict=True,
        )
    ][::-1]
    return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


def _predict_mask_logits(
    predictor: Any,
    *,
    point_coords: Any,
    point_labels: Any,
    boxes: Any,
) -> tuple[Any, Any]:
    import torch

    concat_points = (point_coords, point_labels) if point_coords is not None else None
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)
        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )
    batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
    high_res_features = [
        feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
    ]
    low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )
    return torch.clamp(low_res_masks, -32.0, 32.0), iou_predictions


def _record_image(record: Mapping[str, Any]) -> tuple[Image.Image, tuple[int, int]]:
    image_path = _required_path(record, "image")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    return rgb, (height, width)


def _target_mask_tensor(
    record: Mapping[str, Any],
    *,
    orig_hw: tuple[int, int],
    device: str,
    torch: Any,
) -> Any:
    mask_path = _required_path(record, "mask")
    import numpy as np

    target_size = (orig_hw[1], orig_hw[0])
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        values = torch.as_tensor(
            np.asarray(mask, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
    return values.reshape(1, 1, orig_hw[0], orig_hw[1]).clamp(0.0, 255.0) / 255.0


def _record_prompt_arrays(
    record: Mapping[str, Any],
    *,
    prompt_mode: str = "box-points",
) -> tuple[Any, Any, Any]:
    import numpy as np

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    positive_points = _points(record.get("positive_points_xy"))
    negative_points = _points(record.get("negative_points_xy"))
    if prompt_mode == "positive-point":
        if positive_points:
            points.append(positive_points[0])
            labels.append(1)
    elif prompt_mode in {"positive-points", "box-points"}:
        for point in positive_points:
            points.append(point)
            labels.append(1)
        if prompt_mode == "box-points":
            for point in negative_points:
                points.append(point)
                labels.append(0)

    point_array = np.asarray(points, dtype=np.float32) if points else None
    label_array = np.asarray(labels, dtype=np.int64) if labels else None
    bbox = record.get("bbox_xyxy")
    box_array = None
    if prompt_mode in {"box", "box-points"} and isinstance(bbox, list) and len(bbox) == 4:
        box_array = np.asarray(bbox, dtype=np.float32)
    if point_array is None and box_array is None:
        raise ValueError("SAM training record must include points or bbox_xyxy")
    return point_array, label_array, box_array


def _points(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        return ()
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        points.append((float(item[0]), float(item[1])))
    return tuple(points)


def _required_path(record: Mapping[str, Any], field: str) -> Path:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"SAM training record is missing `{field}` path")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"SAM training record {field} not found: {path}")
    return path


def _dice_loss(logits: Any, target: Any, *, torch: Any) -> Any:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(2, 3))
    denominator = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _soft_iou(logits: Any, target: Any, *, torch: Any) -> Any:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    return ((intersection + 1.0) / (union + 1.0)).detach()


def _save_checkpoint(
    checkpoint_path: Path,
    model: Any,
    config: SamFineTuneConfig,
    *,
    steps: int,
    torch: Any,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "base_model_id": config.model_id,
            "sam2_config_file": _sam2_config_file(config.model_id),
            "sim_satellite_sam_finetune": {
                "steps": steps,
                "config": asdict(config),
            },
        },
        checkpoint_path,
    )


def _write_metrics(path: Path, config: SamFineTuneConfig, result: SamFineTuneResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "result": result.to_json(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sam2_config_file(model_id: str) -> str:
    try:
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
    except ImportError:
        return DEFAULT_SAM2_CONFIG_FILE
    filenames = HF_MODEL_ID_TO_FILENAMES.get(model_id)
    return filenames[0] if filenames is not None else DEFAULT_SAM2_CONFIG_FILE


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: SamFineTuneConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be at least 1 when provided")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.dice_weight < 0 or config.iou_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if config.prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of: {', '.join(PROMPT_MODES)}")
    if config.checkpoint_every_steps is not None and config.checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be at least 1 when provided")
    if not Path(config.records_path).exists():
        raise FileNotFoundError(f"SAM training records not found: {config.records_path}")


def _should_report(steps: int, max_steps: int | None, records: int) -> bool:
    if steps == 1:
        return True
    total = max_steps or records
    if steps == total:
        return True
    return steps % max(1, total // 10) == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _build_click_command():
    import click

    @click.command("sam-finetune", help="Fine-tune SAM2 on SAM binary mask records.")
    @click.option(
        "--records",
        "records_path",
        type=click.Path(path_type=Path),
        default=Path(SamFineTuneConfig.records_path),
        help="records.jsonl produced by `sim-satellite sam-data`.",
    )
    @click.option(
        "--checkpoint",
        "checkpoint_path",
        type=click.Path(path_type=Path),
        default=Path(SamFineTuneConfig.checkpoint_path),
        help="Output SAM2 checkpoint path.",
    )
    @click.option("--model-id", default=SamFineTuneConfig.model_id, show_default=True)
    @click.option("--device", default=SamFineTuneConfig.device, show_default=True)
    @click.option("--epochs", type=int, default=SamFineTuneConfig.epochs, show_default=True)
    @click.option("--max-steps", type=int, default=SamFineTuneConfig.max_steps, show_default=True)
    @click.option("--max-records", type=int, default=None)
    @click.option(
        "--learning-rate",
        type=float,
        default=SamFineTuneConfig.learning_rate,
        show_default=True,
    )
    @click.option(
        "--weight-decay",
        type=float,
        default=SamFineTuneConfig.weight_decay,
        show_default=True,
    )
    @click.option(
        "--dice-weight",
        type=float,
        default=SamFineTuneConfig.dice_weight,
        show_default=True,
    )
    @click.option(
        "--iou-weight",
        type=float,
        default=SamFineTuneConfig.iou_weight,
        show_default=True,
    )
    @click.option(
        "--prompt-mode",
        type=click.Choice(PROMPT_MODES),
        default=SamFineTuneConfig.prompt_mode,
        show_default=True,
    )
    @click.option("--seed", type=int, default=SamFineTuneConfig.seed, show_default=True)
    @click.option("--train-image-encoder", is_flag=True)
    @click.option("--train-prompt-encoder", is_flag=True)
    @click.option("--checkpoint-every-steps", type=int, default=None)
    def command(
        records_path: Path,
        checkpoint_path: Path,
        model_id: str,
        device: str,
        epochs: int,
        max_steps: int | None,
        max_records: int | None,
        learning_rate: float,
        weight_decay: float,
        dice_weight: float,
        iou_weight: float,
        prompt_mode: str,
        seed: int,
        train_image_encoder: bool,
        train_prompt_encoder: bool,
        checkpoint_every_steps: int | None,
    ) -> None:
        try:
            result = fine_tune_sam2(
                SamFineTuneConfig(
                    records_path=str(records_path),
                    checkpoint_path=str(checkpoint_path),
                    model_id=model_id,
                    device=device,
                    epochs=epochs,
                    max_steps=max_steps,
                    max_records=max_records,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dice_weight=dice_weight,
                    iou_weight=iou_weight,
                    prompt_mode=prompt_mode,
                    seed=seed,
                    train_image_encoder=train_image_encoder,
                    train_prompt_encoder=train_prompt_encoder,
                    checkpoint_every_steps=checkpoint_every_steps,
                ),
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


cli = _build_click_command()
