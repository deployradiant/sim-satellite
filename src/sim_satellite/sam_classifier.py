"""Classify SAM proposal masks into OpenEarthMap semantic classes.

SAM/SAM2 gives class-agnostic binary masks. This module adds the next primitive
stage: assign labels to proposals on labeled OpenEarthMap data, train a small
mask-aware classifier, then apply that classifier to unlabeled SAM proposals.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.sam_records import image_key, records_from_jsonl
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.model import ModelConfig

CLASSIFIER_RECORD_VERSION = 1
DEFAULT_CLASSIFIER_RECORDS_PATH = "data/sam/classifier/records.jsonl"
DEFAULT_CLASSIFIER_CHECKPOINT_PATH = "data/sam/classifier/classifier.pt"
DEFAULT_CLASSIFIER_PREDICTIONS_PATH = "data/sam/classifier/predictions.jsonl"


@dataclass(frozen=True)
class SamClassifierLabelConfig:
    manifest_path: str = "data/openearthmap/manifest.jsonl"
    proposals_path: str = "data/sam/compare/baseline/proposals.jsonl"
    output_path: str = DEFAULT_CLASSIFIER_RECORDS_PATH
    min_mask_pixels: int = 64
    min_class_fraction: float = 0.55
    include_unlabeled: bool = False
    max_records: int | None = None
    seed: int = 2026


@dataclass(frozen=True)
class SamClassifierTrainConfig:
    records_path: str = DEFAULT_CLASSIFIER_RECORDS_PATH
    checkpoint_path: str = DEFAULT_CLASSIFIER_CHECKPOINT_PATH
    device: str = "cpu"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-3
    input_size: int = 64
    validation_fraction: float = 0.2
    seed: int = 2026


@dataclass(frozen=True)
class SamClassifierPredictConfig:
    proposals_path: str = "data/sam/openearthmap-unlabeled-baseline/proposals.jsonl"
    checkpoint_path: str = DEFAULT_CLASSIFIER_CHECKPOINT_PATH
    output_path: str = DEFAULT_CLASSIFIER_PREDICTIONS_PATH
    device: str = "cpu"
    input_size: int | None = None
    batch_size: int = 16


@dataclass(frozen=True)
class SamClassifierLabelResult:
    output_path: Path
    proposals: int
    labeled_records: int
    skipped_small: int
    skipped_unknown_image: int
    skipped_low_purity: int
    class_counts: dict[str, int]


@dataclass(frozen=True)
class SamClassifierTrainResult:
    checkpoint_path: Path
    records: int
    train_records: int
    validation_records: int
    final_train_loss: float
    train_accuracy: float
    validation_accuracy: float | None


@dataclass(frozen=True)
class SamClassifierPredictResult:
    output_path: Path
    proposals: int
    predictions: int
    class_counts: dict[str, int]


def label_sam_proposals_for_classifier(
    config: SamClassifierLabelConfig,
) -> SamClassifierLabelResult:
    if config.min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be at least 1")
    if config.min_class_fraction <= 0.0 or config.min_class_fraction > 1.0:
        raise ValueError("min_class_fraction must be in (0, 1]")

    manifest_samples = {
        image_key(sample.image_path): sample for sample in read_manifest(config.manifest_path)
    }
    proposals_path = Path(config.proposals_path)
    proposal_records = list(records_from_jsonl(proposals_path))
    if config.max_records is not None:
        if config.max_records < 1:
            raise ValueError("max_records must be at least 1 when provided")
        random.Random(config.seed).shuffle(proposal_records)
        proposal_records = proposal_records[: config.max_records]

    labeled: list[dict[str, Any]] = []
    skipped_small = 0
    skipped_unknown = 0
    skipped_low_purity = 0
    import numpy as np

    target_label_cache: dict[tuple[str, tuple[int, int]], Any] = {}

    for index, proposal in enumerate(proposal_records):
        key = proposal.get("image_key")
        if not isinstance(key, str) or key not in manifest_samples:
            skipped_unknown += 1
            continue
        sample = manifest_samples[key]
        mask_path = _record_path(proposals_path.parent, proposal, field="mask")
        if mask_path is None:
            skipped_unknown += 1
            continue

        with Image.open(mask_path) as proposal_mask_image:
            proposal_mask = proposal_mask_image.convert("L")
            mask_size = proposal_mask.size
            binary = np.asarray(proposal_mask, dtype=np.uint8) > 0

        area = int(binary.sum())
        if area < config.min_mask_pixels:
            skipped_small += 1
            continue

        cache_key = (key, mask_size)
        class_ids = target_label_cache.get(cache_key)
        if class_ids is None:
            with Image.open(sample.mask_path) as target_mask:
                labels = target_mask.resize(mask_size, Image.Resampling.NEAREST)
                class_ids = mask_to_class_ids(
                    labels,
                    ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
                )
                width, height = mask_size
                class_ids = np.asarray(class_ids, dtype=np.int64).reshape((height, width))
            target_label_cache[cache_key] = class_ids

        selected_class_ids = class_ids[binary]
        if not config.include_unlabeled:
            selected_class_ids = selected_class_ids[selected_class_ids != 0]
        if selected_class_ids.size == 0:
            skipped_low_purity += 1
            continue

        counts = np.bincount(selected_class_ids, minlength=len(OPENEARTHMAP_CLASSES))
        class_id = int(counts.argmax())
        class_pixels = int(counts[class_id])
        class_fraction = class_pixels / area
        if class_fraction < config.min_class_fraction:
            skipped_low_purity += 1
            continue

        bbox = _bbox_from_record_or_mask(proposal, binary=binary, size=mask_size)
        labeled.append(
            {
                "version": CLASSIFIER_RECORD_VERSION,
                "proposal_index": index,
                "image": str(sample.image_path),
                "mask": str(mask_path),
                "source_mask": str(sample.mask_path),
                "split": sample.split,
                "source": sample.source,
                "image_key": key,
                "bbox_xyxy": list(bbox),
                "area": area,
                "class_id": class_id,
                "class_name": OPENEARTHMAP_CLASSES[class_id],
                "class_fraction": class_fraction,
                "label_source": "openearthmap-majority-overlap",
            }
        )

    output = Path(config.output_path)
    _write_jsonl(output, labeled)
    class_counts = Counter(str(record["class_name"]) for record in labeled)
    return SamClassifierLabelResult(
        output_path=output,
        proposals=len(proposal_records),
        labeled_records=len(labeled),
        skipped_small=skipped_small,
        skipped_unknown_image=skipped_unknown,
        skipped_low_purity=skipped_low_purity,
        class_counts=dict(sorted(class_counts.items())),
    )


def train_sam_mask_classifier(config: SamClassifierTrainConfig) -> SamClassifierTrainResult:
    _validate_train_config(config)
    import torch
    from torch.utils.data import DataLoader, Subset

    records = tuple(records_from_jsonl(config.records_path))
    if not records:
        raise ValueError(f"No classifier records found in {config.records_path}")

    _seed_everything(config.seed, torch=torch)
    dataset = _SamProposalClassificationDataset(records, input_size=config.input_size, torch=torch)
    train_indexes, validation_indexes = _split_record_indexes(
        records,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    data_loader: Any = DataLoader
    subset: Any = Subset
    train_loader = data_loader(
        subset(dataset, train_indexes),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = (
        data_loader(subset(dataset, validation_indexes), batch_size=config.batch_size)
        if validation_indexes
        else None
    )
    device = torch.device(config.device)
    model = _SamMaskClassifier(num_classes=len(OPENEARTHMAP_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    final_loss = 0.0
    for _ in range(config.epochs):
        model.train()
        losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        final_loss = sum(losses) / len(losses) if losses else 0.0

    train_accuracy = _classifier_accuracy(model, train_loader, device=device, torch=torch)
    validation_accuracy = (
        _classifier_accuracy(model, validation_loader, device=device, torch=torch)
        if validation_loader is not None
        else None
    )
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": list(OPENEARTHMAP_CLASSES),
            "input_size": config.input_size,
            "config": asdict(config),
            "metrics": {
                "final_train_loss": final_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            },
        },
        checkpoint_path,
    )
    return SamClassifierTrainResult(
        checkpoint_path=checkpoint_path,
        records=len(records),
        train_records=len(train_indexes),
        validation_records=len(validation_indexes),
        final_train_loss=final_loss,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
    )


def classify_sam_proposals(config: SamClassifierPredictConfig) -> SamClassifierPredictResult:
    import torch
    from torch.utils.data import DataLoader

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    input_size = int(config.input_size or checkpoint.get("input_size", 64))
    proposals_path = Path(config.proposals_path)
    records = tuple(records_from_jsonl(proposals_path))
    dataset = _SamProposalPredictionDataset(
        records,
        proposals_root=proposals_path.parent,
        input_size=input_size,
        torch=torch,
    )
    data_loader: Any = DataLoader
    loader = data_loader(dataset, batch_size=config.batch_size)
    device = torch.device(config.device)
    model = _SamMaskClassifier(num_classes=len(OPENEARTHMAP_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        offset = 0
        for features in loader:
            features = features.to(device)
            probabilities = torch.softmax(model(features), dim=1).cpu()
            for row in probabilities:
                source_record = _resolve_prediction_record(records[offset], proposals_path.parent)
                class_id = int(row.argmax().item())
                predictions.append(
                    {
                        **source_record,
                        "predicted_class_id": class_id,
                        "predicted_class_name": OPENEARTHMAP_CLASSES[class_id],
                        "predicted_class_confidence": float(row[class_id].item()),
                    }
                )
                offset += 1

    output = Path(config.output_path)
    _write_jsonl(output, predictions)
    class_counts = Counter(record["predicted_class_name"] for record in predictions)
    return SamClassifierPredictResult(
        output_path=output,
        proposals=len(records),
        predictions=len(predictions),
        class_counts=dict(sorted(class_counts.items())),
    )


class _SamMaskClassifier:
    def __new__(cls, *, num_classes: int) -> Any:
        import torch

        return torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes),
        )


class _SamProposalClassificationDataset:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        input_size: int,
        torch: Any,
    ) -> None:
        self.records = tuple(records)
        self.input_size = input_size
        self.torch = torch
        self._cache: dict[int, tuple[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        record = self.records[index]
        features = _proposal_features(record, input_size=self.input_size, torch=self.torch)
        class_id = record.get("class_id")
        if not isinstance(class_id, int):
            raise RuntimeError("Classifier record is missing integer class_id")
        item = features, self.torch.tensor(class_id, dtype=self.torch.long)
        self._cache[index] = item
        return item


class _SamProposalPredictionDataset:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        proposals_root: Path,
        input_size: int,
        torch: Any,
    ) -> None:
        self.records = tuple(records)
        self.proposals_root = proposals_root
        self.input_size = input_size
        self.torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        record = _resolve_prediction_record(self.records[index], self.proposals_root)
        return _proposal_features(record, input_size=self.input_size, torch=self.torch)


def _proposal_features(record: Mapping[str, Any], *, input_size: int, torch: Any) -> Any:
    image_path = _required_path(record, "image")
    mask_path = _required_path(record, "mask")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        bbox = _bbox_from_record_or_mask(
            record,
            binary=[value > 0 for value in _image_data(mask)],
            size=mask.size,
        )
        mask_crop = mask.crop(_pil_crop_box(bbox)).resize(
            (input_size, input_size),
            Image.Resampling.NEAREST,
        )
    with Image.open(image_path) as image:
        image_crop = image.convert("RGB").crop(_pil_crop_box(bbox)).resize(
            (input_size, input_size),
            Image.Resampling.BILINEAR,
        )

    image_values = torch.frombuffer(bytearray(image_crop.tobytes()), dtype=torch.uint8)
    image_tensor = image_values.view(input_size, input_size, 3).permute(2, 0, 1).float() / 255.0
    mask_values = torch.frombuffer(bytearray(mask_crop.tobytes()), dtype=torch.uint8)
    mask_tensor = mask_values.view(1, input_size, input_size).float() / 255.0
    return torch.cat((image_tensor, mask_tensor), dim=0)


def _split_record_indexes(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indexes = list(range(len(records)))
    random.Random(seed).shuffle(indexes)
    validation_count = int(round(len(indexes) * validation_fraction))
    if validation_fraction > 0.0 and len(indexes) > 1:
        validation_count = min(max(1, validation_count), len(indexes) - 1)
    else:
        validation_count = 0
    return indexes[validation_count:], indexes[:validation_count]


def _classifier_accuracy(model: Any, loader: Any, *, device: Any, torch: Any) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return 0.0 if total == 0 else correct / total


def _resolve_prediction_record(record: Mapping[str, Any], proposals_root: Path) -> dict[str, Any]:
    resolved = dict(record)
    for field in ("image", "mask"):
        value = record.get(field)
        if isinstance(value, str):
            path = Path(value)
            resolved[field] = str(path if path.is_absolute() else proposals_root / path)
    return resolved


def _record_path(root: Path, record: Mapping[str, Any], *, field: str) -> Path | None:
    value = record.get(field)
    if not isinstance(value, str):
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required_path(record: Mapping[str, Any], field: str) -> Path:
    value = record.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"Record is missing path field {field!r}")
    return Path(value)


def _bbox_from_record_or_mask(
    record: Mapping[str, Any],
    *,
    binary: Any,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    value = record.get("bbox_xyxy")
    if isinstance(value, list) and len(value) == 4:
        return tuple(int(item) for item in value)  # type: ignore[return-value]

    import numpy as np

    width, height = size
    selected = np.asarray(binary, dtype=bool).reshape((height, width))
    ys, xs = np.nonzero(selected)
    if xs.size == 0:
        return 0, 0, width - 1, height - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pil_crop_box(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    return x0, y0, max(x0 + 1, x1 + 1), max(y0 + 1, y1 + 1)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def _validate_train_config(config: SamClassifierTrainConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.input_size < 8:
        raise ValueError("input_size must be at least 8")
    if config.validation_fraction < 0.0 or config.validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _image_data(image: Image.Image) -> Any:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return get_flattened_data()
    return image.getdata()


def _build_label_click_command() -> Any:
    import click

    @click.command(
        "sam-classifier-label",
        help="Assign OpenEarthMap class labels to SAM proposals by ground-truth overlap.",
    )
    @click.option(
        "--manifest",
        type=click.Path(path_type=Path),
        default=Path("data/openearthmap/manifest.jsonl"),
    )
    @click.option("--proposals", type=click.Path(path_type=Path), required=True)
    @click.option(
        "--output",
        type=click.Path(path_type=Path),
        default=Path(DEFAULT_CLASSIFIER_RECORDS_PATH),
    )
    @click.option("--min-mask-pixels", type=int, default=64, show_default=True)
    @click.option("--min-class-fraction", type=float, default=0.55, show_default=True)
    @click.option("--include-unlabeled", is_flag=True)
    @click.option("--max-records", type=int, default=None)
    def command(
        manifest: Path,
        proposals: Path,
        output: Path,
        min_mask_pixels: int,
        min_class_fraction: float,
        include_unlabeled: bool,
        max_records: int | None,
    ) -> None:
        try:
            result = label_sam_proposals_for_classifier(
                SamClassifierLabelConfig(
                    manifest_path=str(manifest),
                    proposals_path=str(proposals),
                    output_path=str(output),
                    min_mask_pixels=min_mask_pixels,
                    min_class_fraction=min_class_fraction,
                    include_unlabeled=include_unlabeled,
                    max_records=max_records,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))

    return command


def _build_train_click_command() -> Any:
    import click

    @click.command("sam-classifier-train", help="Train a small classifier for SAM proposal masks.")
    @click.option(
        "--records",
        type=click.Path(path_type=Path),
        default=Path(DEFAULT_CLASSIFIER_RECORDS_PATH),
    )
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=Path(DEFAULT_CLASSIFIER_CHECKPOINT_PATH),
    )
    @click.option("--device", default="cpu", show_default=True)
    @click.option("--epochs", type=int, default=8, show_default=True)
    @click.option("--batch-size", type=int, default=16, show_default=True)
    @click.option("--learning-rate", type=float, default=1e-3, show_default=True)
    @click.option("--input-size", type=int, default=64, show_default=True)
    def command(
        records: Path,
        checkpoint: Path,
        device: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        input_size: int,
    ) -> None:
        try:
            result = train_sam_mask_classifier(
                SamClassifierTrainConfig(
                    records_path=str(records),
                    checkpoint_path=str(checkpoint),
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    input_size=input_size,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))

    return command


def _build_predict_click_command() -> Any:
    import click

    @click.command(
        "sam-classifier-predict",
        help="Classify SAM proposals with a trained mask classifier.",
    )
    @click.option("--proposals", type=click.Path(path_type=Path), required=True)
    @click.option(
        "--checkpoint",
        type=click.Path(path_type=Path),
        default=Path(DEFAULT_CLASSIFIER_CHECKPOINT_PATH),
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path),
        default=Path(DEFAULT_CLASSIFIER_PREDICTIONS_PATH),
    )
    @click.option("--device", default="cpu", show_default=True)
    @click.option("--input-size", type=int, default=None)
    @click.option("--batch-size", type=int, default=16, show_default=True)
    def command(
        proposals: Path,
        checkpoint: Path,
        output: Path,
        device: str,
        input_size: int | None,
        batch_size: int,
    ) -> None:
        try:
            result = classify_sam_proposals(
                SamClassifierPredictConfig(
                    proposals_path=str(proposals),
                    checkpoint_path=str(checkpoint),
                    output_path=str(output),
                    device=device,
                    input_size=input_size,
                    batch_size=batch_size,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))

    return command


label_cli = _build_label_click_command()
train_cli = _build_train_click_command()
predict_cli = _build_predict_click_command()
