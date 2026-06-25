"""Compare small supervised semantic segmentation models on OpenEarthMap tiles."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample, read_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.tensors import image_to_uint8_tensor, mask_to_label_tensor
from sim_satellite.model import ModelConfig

SegmentationModelName = Literal["unet", "deeplab-lite", "segformer-lite"]
DEFAULT_MODELS: tuple[SegmentationModelName, ...] = (
    "unet",
    "deeplab-lite",
    "segformer-lite",
)
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True)
class SegmentationComparisonConfig:
    manifest_path: str = "data/openearthmap/manifest.jsonl"
    output_path: str = "data/segmentation/comparison.json"
    models: tuple[SegmentationModelName, ...] = DEFAULT_MODELS
    tile_grid: int = 3
    image_size: int = 128
    train_images: int = 32
    val_images: int = 8
    max_steps: int = 40
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 2026
    include_unlabeled_iou: bool = False


@dataclass(frozen=True)
class SegmentationModelResult:
    model: str
    train_loss: float
    mean_iou: float
    pixel_accuracy: float
    class_iou: dict[str, float | None]
    train_tiles: int
    val_tiles: int
    steps: int
    parameters: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationComparisonResult:
    config: SegmentationComparisonConfig
    results: tuple[SegmentationModelResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "results": [result.to_json() for result in self.results],
            "best_by_mean_iou": max(self.results, key=lambda result: result.mean_iou).model
            if self.results
            else None,
        }


@dataclass(frozen=True)
class _TileRef:
    sample: SatelliteSample
    crop_box: tuple[int, int, int, int]


def compare_segmentation_models(
    config: SegmentationComparisonConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationComparisonResult:
    _validate_config(config)
    import torch

    _seed_everything(config.seed, torch=torch)
    train_tiles, val_tiles = _split_tiles(config)
    if not train_tiles:
        raise ValueError("No training tiles selected")
    if not val_tiles:
        raise ValueError("No validation tiles selected")

    train_dataset = _TileSegmentationDataset(
        train_tiles,
        image_size=config.image_size,
        torch=torch,
    )
    val_dataset = _TileSegmentationDataset(
        val_tiles,
        image_size=config.image_size,
        torch=torch,
    )
    results: list[SegmentationModelResult] = []
    for model_name in config.models:
        _seed_everything(config.seed, torch=torch)
        _report(progress, f"Training {model_name} for {config.max_steps} steps")
        model = _build_model(model_name, num_classes=len(OPENEARTHMAP_CLASSES), torch=torch)
        model.to(config.device)
        train_loss = _train_model(
            model,
            train_dataset,
            config=config,
            torch=torch,
            progress=lambda message, name=model_name: _report(progress, f"{name}: {message}"),
        )
        metrics = _evaluate_model(
            model,
            val_dataset,
            include_unlabeled_iou=config.include_unlabeled_iou,
            device=config.device,
            torch=torch,
        )
        result = SegmentationModelResult(
            model=model_name,
            train_loss=train_loss,
            mean_iou=metrics["mean_iou"],
            pixel_accuracy=metrics["pixel_accuracy"],
            class_iou=metrics["class_iou"],
            train_tiles=len(train_tiles),
            val_tiles=len(val_tiles),
            steps=config.max_steps,
            parameters=_count_parameters(model),
        )
        results.append(result)
        _report(
            progress,
            f"{model_name}: mean_iou={result.mean_iou:.4f} "
            f"pixel_accuracy={result.pixel_accuracy:.4f}",
        )

    comparison = SegmentationComparisonResult(config=config, results=tuple(results))
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison.to_json(), indent=2, sort_keys=True) + "\n")
    return comparison


def _split_tiles(
    config: SegmentationComparisonConfig,
) -> tuple[tuple[_TileRef, ...], tuple[_TileRef, ...]]:
    samples = sorted(read_manifest(config.manifest_path), key=lambda sample: str(sample.image_path))
    random.Random(config.seed).shuffle(samples)
    needed = config.train_images + config.val_images
    selected = samples[:needed]
    if len(selected) < needed:
        raise ValueError(
            f"Requested {needed} images but manifest only contains {len(samples)} samples"
        )
    train_samples = selected[: config.train_images]
    val_samples = selected[config.train_images :]
    return _tiles_for_samples(train_samples, config.tile_grid), _tiles_for_samples(
        val_samples,
        config.tile_grid,
    )


def _tiles_for_samples(
    samples: Sequence[SatelliteSample],
    tile_grid: int,
) -> tuple[_TileRef, ...]:
    tiles: list[_TileRef] = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_size = image.size
        for crop_box in _tile_windows(image_size, tile_grid):
            tiles.append(_TileRef(sample=sample, crop_box=crop_box))
    return tuple(tiles)


def _tile_windows(
    image_size: tuple[int, int],
    tile_grid: int,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image_size
    windows: list[tuple[int, int, int, int]] = []
    for row in range(tile_grid):
        y0 = round(row * height / tile_grid)
        y1 = round((row + 1) * height / tile_grid)
        for col in range(tile_grid):
            x0 = round(col * width / tile_grid)
            x1 = round((col + 1) * width / tile_grid)
            windows.append((x0, y0, x1, y1))
    return tuple(windows)


class _TileSegmentationDataset:
    def __init__(
        self,
        tiles: Sequence[_TileRef],
        *,
        image_size: int,
        torch: Any,
    ) -> None:
        self.tiles = tuple(tiles)
        self.image_size = image_size
        self.torch = torch
        self.model_config = ModelConfig(
            image_size=image_size,
            mask_channels=len(OPENEARTHMAP_CLASSES),
        )
        self._cache: dict[int, tuple[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        tile = self.tiles[index]
        size = (self.image_size, self.image_size)
        with Image.open(tile.sample.image_path) as image:
            image_tensor = image_to_uint8_tensor(
                image,
                torch=self.torch,
                crop_box=tile.crop_box,
                size=size,
            )
        with Image.open(tile.sample.mask_path) as mask:
            label_tensor = mask_to_label_tensor(
                mask,
                torch=self.torch,
                config=self.model_config,
                crop_box=tile.crop_box,
                size=size,
            )
        item = image_tensor.permute(2, 0, 1).float() / 255.0, label_tensor.long()
        self._cache[index] = item
        return item


def _train_model(
    model: Any,
    dataset: _TileSegmentationDataset,
    *,
    config: SegmentationComparisonConfig,
    torch: Any,
    progress: ProgressReporter | None,
) -> float:
    import torch.nn.functional as F

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    rng = random.Random(config.seed)
    for step in range(1, config.max_steps + 1):
        indexes = [rng.randrange(len(dataset)) for _ in range(config.batch_size)]
        images, labels = _batch(dataset, indexes, device=config.device, torch=torch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step == 1 or step == config.max_steps or step % max(1, config.max_steps // 4) == 0:
            _report(progress, f"step={step} loss={losses[-1]:.4f}")
    return sum(losses) / len(losses)


def _evaluate_model(
    model: Any,
    dataset: _TileSegmentationDataset,
    *,
    include_unlabeled_iou: bool,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), len(OPENEARTHMAP_CLASSES)),
        dtype=torch.long,
    )
    with torch.no_grad():
        for index in range(len(dataset)):
            images, labels = _batch(dataset, [index], device=device, torch=torch)
            predictions = model(images).argmax(dim=1)
            confusion += _confusion_matrix(
                predictions.cpu(),
                labels.cpu(),
                num_classes=len(OPENEARTHMAP_CLASSES),
                torch=torch,
            )
    metrics = _segmentation_metrics(
        confusion,
        include_unlabeled_iou=include_unlabeled_iou,
        torch=torch,
    )
    return metrics


def _batch(
    dataset: _TileSegmentationDataset,
    indexes: Sequence[int],
    *,
    device: str,
    torch: Any,
) -> tuple[Any, Any]:
    images: list[Any] = []
    labels: list[Any] = []
    for index in indexes:
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return torch.stack(images).to(device), torch.stack(labels).to(device)


def _confusion_matrix(predictions: Any, labels: Any, *, num_classes: int, torch: Any) -> Any:
    valid = (labels >= 0) & (labels < num_classes)
    encoded = num_classes * labels[valid].long() + predictions[valid].long()
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )


def _segmentation_metrics(
    confusion: Any,
    *,
    include_unlabeled_iou: bool,
    torch: Any,
) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    union = predicted + actual - true_positive
    class_iou_values: list[float] = []
    class_iou: dict[str, float | None] = {}
    for class_id, class_name in enumerate(OPENEARTHMAP_CLASSES):
        if union[class_id].item() == 0:
            class_iou[class_name] = None
            continue
        value = float((true_positive[class_id] / union[class_id]).item())
        class_iou[class_name] = value
        if include_unlabeled_iou or class_id != 0:
            class_iou_values.append(value)
    total = confusion.sum().item()
    pixel_accuracy = float(true_positive.sum().item() / total) if total else 0.0
    mean_iou = sum(class_iou_values) / len(class_iou_values) if class_iou_values else 0.0
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "class_iou": class_iou,
    }


def _build_model(model_name: SegmentationModelName, *, num_classes: int, torch: Any) -> Any:
    if model_name == "unet":
        return _UNet(num_classes=num_classes)
    if model_name == "deeplab-lite":
        return _DeepLabLite(num_classes=num_classes)
    if model_name == "segformer-lite":
        return _SegFormerLite(num_classes=num_classes)
    raise ValueError(f"Unsupported segmentation model: {model_name}")


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _UNet(torch.nn.Module):
    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        self.enc1 = _conv_block(3, 16)
        self.enc2 = _conv_block(16, 32)
        self.bottleneck = _conv_block(32, 64)
        self.dec2 = _conv_block(96, 32)
        self.dec1 = _conv_block(48, 16)
        self.head = torch.nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))
        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.head(dec1)


class _DeepLabLite(torch.nn.Module):
    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        nn = torch.nn
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = _ASPP(48, 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        features = self.decoder(self.aspp(self.encoder(x)))
        return F.interpolate(features, size=x.shape[-2:], mode="bilinear", align_corners=False)


class _ASPP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        nn = torch.nn
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class _SegFormerLite(torch.nn.Module):
    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        nn = torch.nn
        self.patch = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(channels=64, heads=4, mlp_channels=128) for _ in range(2)]
        )
        self.decode = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        height, width = x.shape[-2:]
        features = self.patch(x)
        batch, channels, feature_height, feature_width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        encoded = tokens.transpose(1, 2).reshape(batch, channels, feature_height, feature_width)
        logits = self.decode(encoded)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


class _TransformerBlock(torch.nn.Module):
    def __init__(self, *, channels: int, heads: int, mlp_channels: int) -> None:
        super().__init__()
        nn = torch.nn
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = _SelfAttention(channels=channels, heads=heads)
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_channels),
            nn.GELU(),
            nn.Linear(mlp_channels, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm_attention(tokens))
        return tokens + self.mlp(self.norm_mlp(tokens))


class _SelfAttention(torch.nn.Module):
    def __init__(self, *, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(channels, channels * 3)
        self.project = torch.nn.Linear(channels, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        mixed = attention @ value
        mixed = mixed.transpose(1, 2).reshape(batch, length, channels)
        return self.project(mixed)


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: SegmentationComparisonConfig) -> None:
    if config.tile_grid < 1:
        raise ValueError("tile_grid must be positive")
    if config.image_size < 16:
        raise ValueError("image_size must be at least 16")
    if config.train_images < 1 or config.val_images < 1:
        raise ValueError("train_images and val_images must be positive")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not config.models:
        raise ValueError("at least one model must be selected")
    invalid = [model for model in config.models if model not in DEFAULT_MODELS]
    if invalid:
        raise ValueError(f"Unsupported segmentation models: {invalid}")
    if not Path(config.manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest_path}")


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _parse_models(value: str) -> tuple[SegmentationModelName, ...]:
    models = tuple(model.strip() for model in value.split(",") if model.strip())
    invalid = [model for model in models if model not in DEFAULT_MODELS]
    if invalid:
        raise ValueError(f"Unsupported segmentation models: {invalid}")
    return models  # type: ignore[return-value]


def _build_click_command():
    import click

    @click.command(
        "segmentation-compare",
        help="Train and compare supervised segmentation models on OpenEarthMap tiles.",
    )
    @click.option(
        "--manifest",
        "manifest_path",
        type=click.Path(path_type=Path),
        default=Path(SegmentationComparisonConfig.manifest_path),
    )
    @click.option(
        "--output",
        "output_path",
        type=click.Path(path_type=Path),
        default=Path(SegmentationComparisonConfig.output_path),
    )
    @click.option(
        "--models",
        default=",".join(DEFAULT_MODELS),
        show_default=True,
        help="Comma-separated: unet,deeplab-lite,segformer-lite.",
    )
    @click.option(
        "--tile-grid",
        type=int,
        default=SegmentationComparisonConfig.tile_grid,
        show_default=True,
    )
    @click.option(
        "--image-size",
        type=int,
        default=SegmentationComparisonConfig.image_size,
        show_default=True,
    )
    @click.option(
        "--train-images",
        type=int,
        default=SegmentationComparisonConfig.train_images,
        show_default=True,
    )
    @click.option(
        "--val-images",
        type=int,
        default=SegmentationComparisonConfig.val_images,
        show_default=True,
    )
    @click.option(
        "--max-steps",
        type=int,
        default=SegmentationComparisonConfig.max_steps,
        show_default=True,
    )
    @click.option(
        "--batch-size",
        type=int,
        default=SegmentationComparisonConfig.batch_size,
        show_default=True,
    )
    @click.option(
        "--learning-rate",
        type=float,
        default=SegmentationComparisonConfig.learning_rate,
        show_default=True,
    )
    @click.option(
        "--weight-decay",
        type=float,
        default=SegmentationComparisonConfig.weight_decay,
        show_default=True,
    )
    @click.option("--device", default=SegmentationComparisonConfig.device, show_default=True)
    @click.option("--seed", type=int, default=SegmentationComparisonConfig.seed, show_default=True)
    @click.option("--include-unlabeled-iou", is_flag=True)
    def command(
        manifest_path: Path,
        output_path: Path,
        models: str,
        tile_grid: int,
        image_size: int,
        train_images: int,
        val_images: int,
        max_steps: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        seed: int,
        include_unlabeled_iou: bool,
    ) -> None:
        try:
            result = compare_segmentation_models(
                SegmentationComparisonConfig(
                    manifest_path=str(manifest_path),
                    output_path=str(output_path),
                    models=_parse_models(models),
                    tile_grid=tile_grid,
                    image_size=image_size,
                    train_images=train_images,
                    val_images=val_images,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    device=device,
                    seed=seed,
                    include_unlabeled_iou=include_unlabeled_iou,
                ),
                progress=click.echo,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result.to_json(), indent=2, sort_keys=True))

    return command


cli = _build_click_command()
