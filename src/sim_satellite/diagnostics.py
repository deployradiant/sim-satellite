"""Color and class-mean diagnostics for generated satellite images."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import torch
from PIL import Image

from sim_satellite.data.loader import read_manifest
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.data.tensors import mask_to_label_tensor
from sim_satellite.model import ModelConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TENSOR_SUFFIXES = {".pt", ".pth"}


@dataclass(frozen=True)
class ImageStats:
    path: str
    width: int
    height: int
    saturation_mean: float
    rgb_std: float
    rgb_mean: tuple[float, float, float]


@dataclass(frozen=True)
class ReferenceStats:
    image_count: int
    pixel_count: int
    saturation_mean: float
    rgb_std: float
    rgb_mean: tuple[float, float, float]
    class_means: dict[int, tuple[float, float, float]]
    class_pixel_counts: dict[int, int]


@dataclass(frozen=True)
class TargetDiagnostics:
    image: ImageStats
    saturation_delta: float | None
    rgb_std_delta: float | None
    class_mean_l2: float | None
    classes_compared: int


@dataclass(frozen=True)
class ReferenceInput:
    image_path: Path
    mask_path: Path | None = None
    prepared_path: Path | None = None


def run_diagnostics(
    targets: tuple[Path, ...],
    *,
    target_masks: tuple[Path | None, ...] | None = None,
    references: tuple[ReferenceInput, ...] = (),
) -> dict[str, Any]:
    if not targets:
        raise ValueError("At least one target image is required")
    if target_masks is None:
        target_masks = tuple(None for _ in targets)
    if len(target_masks) != len(targets):
        raise ValueError("target_masks must be empty or match the number of targets")

    reference = build_reference_stats(references) if references else None
    target_results = tuple(
        diagnose_target(target, mask_path=mask_path, reference=reference)
        for target, mask_path in zip(targets, target_masks, strict=True)
    )
    return {
        "reference": _reference_payload(reference),
        "targets": [_target_payload(result) for result in target_results],
    }


def diagnose_target(
    target_path: Path,
    *,
    mask_path: Path | None,
    reference: ReferenceStats | None,
) -> TargetDiagnostics:
    image = load_rgb_image(target_path)
    image_stats = image_statistics(image, path=target_path)
    target_mask = load_mask(mask_path) if mask_path is not None else load_embedded_mask(target_path)

    saturation_delta = None
    rgb_std_delta = None
    class_mean_l2 = None
    classes_compared = 0
    if reference is not None:
        saturation_delta = image_stats.saturation_mean - reference.saturation_mean
        rgb_std_delta = image_stats.rgb_std - reference.rgb_std
        if target_mask is not None and reference.class_means:
            class_mean_l2, classes_compared = class_mean_distance(
                image,
                target_mask,
                reference.class_means,
            )

    return TargetDiagnostics(
        image=image_stats,
        saturation_delta=saturation_delta,
        rgb_std_delta=rgb_std_delta,
        class_mean_l2=class_mean_l2,
        classes_compared=classes_compared,
    )


def build_reference_stats(references: tuple[ReferenceInput, ...]) -> ReferenceStats:
    if not references:
        raise ValueError("At least one reference image is required")

    image_count = 0
    pixel_count = 0
    saturation_sum = 0.0
    rgb_sum = torch.zeros(3, dtype=torch.float64)
    rgb_squared_sum = torch.zeros(3, dtype=torch.float64)
    class_rgb_sums: dict[int, torch.Tensor] = {}
    class_pixel_counts: dict[int, int] = {}

    for reference in references:
        image, mask = load_reference(reference)
        flat = image.reshape(-1, 3).double()
        image_count += 1
        pixel_count += flat.shape[0]
        saturation_sum += float(_saturation(image).sum().item())
        rgb_sum += flat.sum(dim=0)
        rgb_squared_sum += (flat * flat).sum(dim=0)

        if mask is not None:
            mask = _resize_mask_if_needed(mask, image)
            for class_id, class_mean, class_pixels in class_means(image, mask):
                class_rgb_sums.setdefault(class_id, torch.zeros(3, dtype=torch.float64))
                class_rgb_sums[class_id] += class_mean.double() * class_pixels
                class_pixel_counts[class_id] = class_pixel_counts.get(class_id, 0) + class_pixels

    if pixel_count == 0:
        raise ValueError("Reference images contain no pixels")

    rgb_mean_tensor = rgb_sum / pixel_count
    rgb_variance = (
        rgb_squared_sum / pixel_count - rgb_mean_tensor * rgb_mean_tensor
    ).clamp_min(0.0)
    reference_class_means = {
        class_id: _rgb_tuple(class_rgb_sums[class_id] / class_pixel_counts[class_id])
        for class_id in sorted(class_rgb_sums)
        if class_pixel_counts[class_id] > 0
    }
    return ReferenceStats(
        image_count=image_count,
        pixel_count=pixel_count,
        saturation_mean=saturation_sum / pixel_count,
        rgb_std=float(torch.sqrt(rgb_variance).mean().item()),
        rgb_mean=_rgb_tuple(rgb_mean_tensor),
        class_means=reference_class_means,
        class_pixel_counts=dict(sorted(class_pixel_counts.items())),
    )


def load_reference(reference: ReferenceInput) -> tuple[torch.Tensor, torch.Tensor | None]:
    if reference.prepared_path is None:
        return load_rgb_image(reference.image_path), (
            load_mask(reference.mask_path) if reference.mask_path is not None else None
        )

    prepared = load_prepared_sample(reference.prepared_path, torch=torch)
    image = _rgb_tensor_from_any(prepared.image, path=reference.prepared_path)
    mask = prepared.mask.long()
    return image, mask


def load_rgb_image(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return _pil_image_to_rgb_tensor(image)
    if suffix in TENSOR_SUFFIXES:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "image" not in payload:
                raise ValueError(f"Tensor file is missing image: {path}")
            payload = payload["image"]
        return _rgb_tensor_from_any(payload, path=path)
    raise ValueError(f"Unsupported image file suffix {path.suffix!r}: {path}")


def load_embedded_mask(path: Path) -> torch.Tensor | None:
    if path.suffix.lower() not in TENSOR_SUFFIXES:
        return None

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    if "semantic_mask" in payload:
        return _mask_tensor_from_any(payload["semantic_mask"], path=path)
    if "mask" in payload:
        return _mask_tensor_from_any(payload["mask"], path=path)
    return None


def load_mask(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as mask:
            labels = mask_to_label_tensor(mask, torch=torch, config=ModelConfig())
        return labels.long()
    if suffix in TENSOR_SUFFIXES:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "semantic_mask" in payload:
                payload = payload["semantic_mask"]
            elif "mask" in payload:
                payload = payload["mask"]
            else:
                raise ValueError(f"Tensor file is missing mask or semantic_mask: {path}")
        return _mask_tensor_from_any(payload, path=path)
    raise ValueError(f"Unsupported mask file suffix {path.suffix!r}: {path}")


def image_statistics(image: torch.Tensor, *, path: Path) -> ImageStats:
    height, width, _ = image.shape
    flat = image.reshape(-1, 3)
    rgb_mean = flat.mean(dim=0)
    rgb_std = flat.std(dim=0, unbiased=False).mean()
    return ImageStats(
        path=str(path),
        width=width,
        height=height,
        saturation_mean=float(_saturation(image).mean().item()),
        rgb_std=float(rgb_std.item()),
        rgb_mean=_rgb_tuple(rgb_mean),
    )


def class_mean_distance(
    image: torch.Tensor,
    mask: torch.Tensor,
    reference_class_means: dict[int, tuple[float, float, float]],
) -> tuple[float | None, int]:
    mask = _resize_mask_if_needed(mask, image)
    weighted_distance = 0.0
    total_pixels = 0
    classes_compared = 0
    for class_id, target_mean, class_pixels in class_means(image, mask):
        reference_mean = reference_class_means.get(class_id)
        if reference_mean is None:
            continue
        distance = torch.linalg.vector_norm(
            target_mean - torch.tensor(reference_mean, dtype=target_mean.dtype)
        )
        weighted_distance += float(distance.item()) * class_pixels
        total_pixels += class_pixels
        classes_compared += 1
    if total_pixels == 0:
        return None, classes_compared
    return weighted_distance / total_pixels, classes_compared


def class_means(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> list[tuple[int, torch.Tensor, int]]:
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(image.shape[:2])}"
        )

    results: list[tuple[int, torch.Tensor, int]] = []
    flat_image = image.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    for class_id_tensor in torch.unique(flat_mask, sorted=True):
        class_id = int(class_id_tensor.item())
        class_pixels = flat_mask == class_id
        count = int(class_pixels.sum().item())
        if count == 0:
            continue
        results.append((class_id, flat_image[class_pixels].mean(dim=0), count))
    return results


def reference_inputs_from_manifest(
    manifest_path: Path,
    *,
    max_samples: int,
) -> tuple[ReferenceInput, ...]:
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    samples = read_manifest(manifest_path)
    return tuple(
        ReferenceInput(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            prepared_path=sample.prepared_path,
        )
        for sample in samples[:max_samples]
    )


def _pil_image_to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return values.view(height, width, 3).float() / 255.0


def _rgb_tensor_from_any(value: Any, *, path: Path) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected image tensor in {path}")
    tensor = value.detach().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected a single image tensor in {path}, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected image tensor with 3 dimensions in {path}, got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] == 3:
        pass
    elif tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    else:
        raise ValueError(f"Expected RGB image tensor in {path}, got shape {tuple(tensor.shape)}")

    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0

    tensor = tensor.float()
    minimum = float(tensor.min().item()) if tensor.numel() else 0.0
    maximum = float(tensor.max().item()) if tensor.numel() else 0.0
    if minimum < 0.0:
        return ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).contiguous()
    if maximum > 1.0:
        return (tensor.clamp(0.0, 255.0) / 255.0).contiguous()
    return tensor.clamp(0.0, 1.0).contiguous()


def _mask_tensor_from_any(value: Any, *, path: Path) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected mask tensor in {path}")
    tensor = value.detach().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected a single mask tensor in {path}, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[0]
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif 1 < tensor.shape[0] <= 64:
            tensor = tensor.argmax(dim=0)
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        elif 1 < tensor.shape[-1] <= 64:
            tensor = tensor.argmax(dim=-1)
        else:
            raise ValueError(f"Expected mask channels in {path}, got shape {tuple(tensor.shape)}")
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected mask tensor with 2 dimensions in {path}, got {tuple(tensor.shape)}"
        )
    return tensor.long()


def _resize_mask_if_needed(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    target_shape = image.shape[:2]
    if mask.shape == target_shape:
        return mask.long()
    resized = torch.nn.functional.interpolate(
        mask.float().view(1, 1, mask.shape[0], mask.shape[1]),
        size=target_shape,
        mode="nearest",
    )
    return resized.view(*target_shape).long()


def _saturation(image: torch.Tensor) -> torch.Tensor:
    maximum = image.amax(dim=-1)
    minimum = image.amin(dim=-1)
    chroma = maximum - minimum
    return torch.where(maximum > 0.0, chroma / maximum.clamp_min(1e-12), torch.zeros_like(maximum))


def _rgb_tuple(tensor: torch.Tensor) -> tuple[float, float, float]:
    values = tensor.detach().cpu().tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


def _reference_payload(reference: ReferenceStats | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "image_count": reference.image_count,
        "pixel_count": reference.pixel_count,
        "saturation_mean": reference.saturation_mean,
        "rgb_std": reference.rgb_std,
        "rgb_mean": list(reference.rgb_mean),
        "class_count": len(reference.class_means),
        "classes": {
            str(class_id): {
                "rgb_mean": list(reference.class_means[class_id]),
                "pixels": reference.class_pixel_counts[class_id],
            }
            for class_id in sorted(reference.class_means)
        },
    }


def _target_payload(result: TargetDiagnostics) -> dict[str, Any]:
    return {
        "path": result.image.path,
        "width": result.image.width,
        "height": result.image.height,
        "saturation_mean": result.image.saturation_mean,
        "rgb_std": result.image.rgb_std,
        "rgb_mean": list(result.image.rgb_mean),
        "saturation_delta": result.saturation_delta,
        "rgb_std_delta": result.rgb_std_delta,
        "class_mean_l2": result.class_mean_l2,
        "classes_compared": result.classes_compared,
    }


def _format_table(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    reference = payload["reference"]
    if reference is None:
        lines.append("reference: none")
    else:
        lines.append(
            "reference: "
            f"images={reference['image_count']} "
            f"pixels={reference['pixel_count']} "
            f"sat={reference['saturation_mean']:.4f} "
            f"rgb_std={reference['rgb_std']:.4f} "
            f"class_baselines={reference['class_count']}"
        )
    lines.append(
        "target\tsize\tsat\trgb_std\tsat_delta\trgb_std_delta\tclass_l2\tclasses"
    )
    for target in payload["targets"]:
        size = f"{target['width']}x{target['height']}"
        lines.append(
            "\t".join(
                (
                    target["path"],
                    size,
                    _format_float(target["saturation_mean"]),
                    _format_float(target["rgb_std"]),
                    _format_float(target["saturation_delta"]),
                    _format_float(target["rgb_std_delta"]),
                    _format_float(target["class_mean_l2"]),
                    str(target["classes_compared"]),
                )
            )
        )
    return "\n".join(lines)


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _reference_inputs_from_options(
    reference_images: tuple[Path, ...],
    reference_masks: tuple[Path, ...],
    reference_prepared: tuple[Path, ...],
    reference_manifest: Path | None,
    *,
    max_reference_samples: int,
) -> tuple[ReferenceInput, ...]:
    if reference_masks and len(reference_masks) != len(reference_images):
        raise click.UsageError("--reference-mask must be provided once per --reference-image")

    references: list[ReferenceInput] = []
    for index, image_path in enumerate(reference_images):
        mask_path = reference_masks[index] if reference_masks else None
        references.append(ReferenceInput(image_path=image_path, mask_path=mask_path))
    for prepared_path in reference_prepared:
        references.append(
            ReferenceInput(
                image_path=prepared_path,
                mask_path=None,
                prepared_path=prepared_path,
            )
        )
    if reference_manifest is not None:
        references.extend(
            reference_inputs_from_manifest(
                reference_manifest,
                max_samples=max_reference_samples,
            )
        )
    return tuple(references)


def _target_masks_from_options(
    targets: tuple[Path, ...],
    target_masks: tuple[Path, ...],
) -> tuple[Path | None, ...]:
    if not target_masks:
        return tuple(None for _ in targets)
    if len(target_masks) != len(targets):
        raise click.UsageError("--target-mask must be provided once per target")
    return target_masks


def _build_click_command() -> click.Command:
    @click.command("diagnostics", help="Measure generated image color and class-mean metrics.")
    @click.argument(
        "targets",
        nargs=-1,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--target-mask",
        "target_masks",
        multiple=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Mask for a target image. Repeat once per target.",
    )
    @click.option(
        "--reference-image",
        "reference_images",
        multiple=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Reference RGB image. Repeat for multiple references.",
    )
    @click.option(
        "--reference-mask",
        "reference_masks",
        multiple=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Reference mask paired by order with --reference-image.",
    )
    @click.option(
        "--reference-prepared",
        "reference_prepared",
        multiple=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Prepared OpenEarthMap .pt sample containing image and mask tensors.",
    )
    @click.option(
        "--reference-manifest",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Manifest whose prepared samples are used as reference data.",
    )
    @click.option(
        "--max-reference-samples",
        type=click.IntRange(min=1),
        default=128,
        show_default=True,
    )
    @click.option(
        "--format",
        "output_format",
        type=click.Choice(["table", "json"]),
        default="table",
        show_default=True,
    )
    def command(
        targets: tuple[Path, ...],
        target_masks: tuple[Path, ...],
        reference_images: tuple[Path, ...],
        reference_masks: tuple[Path, ...],
        reference_prepared: tuple[Path, ...],
        reference_manifest: Path | None,
        max_reference_samples: int,
        output_format: str,
    ) -> None:
        if not targets:
            raise click.UsageError("At least one target image is required")
        references = _reference_inputs_from_options(
            reference_images,
            reference_masks,
            reference_prepared,
            reference_manifest,
            max_reference_samples=max_reference_samples,
        )
        payload = run_diagnostics(
            targets,
            target_masks=_target_masks_from_options(targets, target_masks),
            references=references,
        )
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(_format_table(payload))

    return command


cli = _build_click_command()


if __name__ == "__main__":
    cli()
