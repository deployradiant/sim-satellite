"""Color and class-mean diagnostics for generated satellite images."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sim_satellite import training_data
from sim_satellite.data.loader import SatelliteSample, read_manifest
from sim_satellite.data.prepared import PreparedSample, load_prepared_sample
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


@dataclass(frozen=True)
class GeneratedTarget:
    path: str
    image: Any
    mask: Any | None = None


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
    return diagnostics_payload(reference, target_results)


def run_generated_diagnostics(
    targets: tuple[GeneratedTarget, ...],
    *,
    references: tuple[ReferenceInput, ...] = (),
) -> dict[str, Any]:
    if not targets:
        raise ValueError("At least one generated target is required")

    reference = build_reference_stats(references) if references else None
    target_results = tuple(
        diagnose_generated_target(target, reference=reference) for target in targets
    )
    return diagnostics_payload(reference, target_results)


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


def diagnose_generated_target(
    target: GeneratedTarget,
    *,
    reference: ReferenceStats | None,
) -> TargetDiagnostics:
    return diagnose_image_tensor(
        target.image,
        path=target.path,
        mask=target.mask,
        reference=reference,
    )


def diagnose_image_tensor(
    image: Any,
    *,
    path: str | Path,
    mask: Any | None,
    reference: ReferenceStats | None,
) -> TargetDiagnostics:
    image = _rgb_tensor_from_any(image, path=Path(path))
    target_mask = _mask_tensor_from_any(mask, path=Path(path)) if mask is not None else None
    image_stats = image_statistics(image, path=path)

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
    rgb_variance = (rgb_squared_sum / pixel_count - rgb_mean_tensor * rgb_mean_tensor).clamp_min(
        0.0
    )
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


def image_statistics(image: torch.Tensor, *, path: str | Path) -> ImageStats:
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


def reference_inputs_from_samples(
    samples: Sequence[SatelliteSample],
    *,
    max_samples: int,
) -> tuple[ReferenceInput, ...]:
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    return tuple(
        ReferenceInput(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            prepared_path=sample.prepared_path,
        )
        for sample in samples[:max_samples]
    )


def run_training_generation_diagnostics(
    *,
    model: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    samples: Sequence[SatelliteSample],
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
    sample_count: int,
    reference_sample_count: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if reference_sample_count < 1:
        raise ValueError("reference_sample_count must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")

    diagnostic_samples = tuple(samples[:sample_count])
    if not diagnostic_samples:
        raise ValueError("At least one sample is required for diagnostics")

    reference_inputs = reference_inputs_from_samples(
        samples,
        max_samples=min(reference_sample_count, len(samples)),
    )
    generated_targets = _generate_diagnostic_targets(
        diagnostic_samples,
        model=model,
        torch=torch,
        model_config=model_config,
        device=device,
        preloaded_samples=preloaded_samples,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return run_generated_diagnostics(generated_targets, references=reference_inputs)


def write_diagnostics_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def diagnostics_payload(
    reference: ReferenceStats | None,
    targets: Sequence[TargetDiagnostics],
) -> dict[str, Any]:
    return {
        "reference": _reference_payload(reference),
        "targets": [_target_payload(result) for result in targets],
    }


def _generate_diagnostic_targets(
    samples: tuple[SatelliteSample, ...],
    *,
    model: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
    steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[GeneratedTarget, ...]:
    was_training = bool(getattr(model, "training", False))
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets: list[GeneratedTarget] = []
    try:
        with torch.no_grad():
            for index, sample in enumerate(samples):
                crop_box = training_data.center_square_crop_box(
                    sample,
                    crop_size=model_config.image_size,
                    preloaded_samples=preloaded_samples,
                )
                _, semantic_mask = training_data.load_batch(
                    (sample,),
                    [crop_box],
                    torch=torch,
                    model_config=model_config,
                    device=device,
                    preloaded_samples=preloaded_samples,
                )
                image = _generate_image_from_mask(
                    semantic_mask,
                    model=model,
                    torch=torch,
                    model_config=model_config,
                    device=device,
                    generator=generator,
                    steps=steps,
                    guidance_scale=guidance_scale,
                )
                targets.append(
                    GeneratedTarget(
                        path=f"{sample.image_path}#diagnostic-{index}",
                        image=image,
                        mask=semantic_mask[0].argmax(dim=0).detach().cpu(),
                    )
                )
    finally:
        if was_training:
            model.train()
    return tuple(targets)


def _generate_image_from_mask(
    semantic_mask: Any,
    *,
    model: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    generator: Any,
    steps: int,
    guidance_scale: float,
) -> Any:
    image = torch.randn(
        (
            1,
            model_config.image_channels,
            model_config.image_size,
            model_config.image_size,
        ),
        generator=generator,
        device="cpu",
    ).to(device)
    for step in reversed(range(1, steps + 1)):
        timestep = torch.full(
            (1,),
            step / float(steps),
            dtype=torch.float32,
            device=device,
        )
        velocity = _guided_prediction(
            model,
            image,
            semantic_mask,
            timestep,
            guidance_scale=guidance_scale,
            torch=torch,
        )
        image = image - velocity / float(steps)
    return ((image[0].clamp(-1.0, 1.0) + 1.0) / 2.0).permute(1, 2, 0).detach().cpu()


def _guided_prediction(
    model: Any,
    image: Any,
    semantic_mask: Any,
    timestep: Any,
    *,
    guidance_scale: float,
    torch: Any,
) -> Any:
    conditional = model(image, semantic_mask, timestep)
    if guidance_scale == 1.0:
        return conditional

    empty_mask = torch.zeros_like(semantic_mask)
    unconditional = model(image, empty_mask, timestep)
    return unconditional + guidance_scale * (conditional - unconditional)


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
