"""Inpaint training helpers."""

from __future__ import annotations

import math
from types import ModuleType

import torch

TRAINING_TASK_BASE = "base"
TRAINING_TASK_INPAINT = "inpaint"


def training_task_for_step(inpaint_probability: float, step_number: int) -> str:
    if inpaint_probability <= 0.0 or step_number < 1:
        return TRAINING_TASK_BASE
    if inpaint_probability >= 1.0:
        return TRAINING_TASK_INPAINT

    inpaint_count = math.floor(step_number * inpaint_probability + 1e-12)
    previous_inpaint_count = math.floor((step_number - 1) * inpaint_probability + 1e-12)
    return TRAINING_TASK_INPAINT if inpaint_count > previous_inpaint_count else TRAINING_TASK_BASE


def validation_tasks(inpaint_probability: float) -> tuple[str, ...]:
    if inpaint_probability <= 0.0:
        return (TRAINING_TASK_BASE,)
    return (TRAINING_TASK_BASE, TRAINING_TASK_INPAINT)


def flow_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    inpaint_mask: torch.Tensor | None,
    torch: ModuleType,
) -> torch.Tensor:
    loss = torch.nn.functional.mse_loss(
        prediction.float(),
        target.float(),
        reduction="none",
    )
    if inpaint_mask is None:
        return loss.mean()

    weights = inpaint_mask.expand_as(loss).float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def append_inpaint_condition(
    semantic_mask: torch.Tensor,
    *,
    inpaint_mask: torch.Tensor | None,
    use_inpaint_mask: bool,
    torch: ModuleType,
) -> torch.Tensor:
    if not use_inpaint_mask:
        return semantic_mask
    if inpaint_mask is None:
        inpaint_mask = torch.ones(
            (semantic_mask.shape[0], 1, semantic_mask.shape[-2], semantic_mask.shape[-1]),
            device=semantic_mask.device,
            dtype=semantic_mask.dtype,
        )
    return torch.cat([semantic_mask, inpaint_mask.to(dtype=semantic_mask.dtype)], dim=1)


def sample_inpaint_mask(
    batch_size: int,
    image_size: int,
    *,
    extension_probability: float,
    min_size_ratio: float,
    max_size_ratio: float,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, 1, image_size, image_size), device=device)
    min_size = max(1, min(image_size, round(image_size * min_size_ratio)))
    max_size = max(min_size, min(image_size, round(image_size * max_size_ratio)))

    for batch_index in range(batch_size):
        if torch.rand((), generator=generator, device=device).item() < extension_probability:
            size = int(
                torch.randint(min_size, max_size + 1, (), generator=generator, device=device).item()
            )
            side = int(torch.randint(0, 4, (), generator=generator, device=device).item())
            _fill_extension_mask(
                mask[batch_index],
                image_size=image_size,
                size=size,
                side=side,
            )
            continue

        shape_kind = int(torch.randint(0, 5, (), generator=generator, device=device).item())
        if shape_kind == 0:
            _fill_rectangle_mask(
                mask[batch_index],
                image_size=image_size,
                min_size=min_size,
                max_size=max_size,
                torch=torch,
                generator=generator,
                device=device,
            )
        elif shape_kind == 1:
            _fill_multi_rectangle_mask(
                mask[batch_index],
                image_size=image_size,
                min_size=min_size,
                max_size=max_size,
                torch=torch,
                generator=generator,
                device=device,
            )
        elif shape_kind == 2:
            _fill_brush_mask(
                mask[batch_index],
                image_size=image_size,
                min_size=min_size,
                max_size=max_size,
                torch=torch,
                generator=generator,
                device=device,
            )
        elif shape_kind == 3:
            _fill_round_mask(
                mask[batch_index],
                image_size=image_size,
                min_size=min_size,
                max_size=max_size,
                torch=torch,
                generator=generator,
                device=device,
            )
        else:
            _fill_blob_mask(
                mask[batch_index],
                image_size=image_size,
                min_size=min_size,
                max_size=max_size,
                torch=torch,
                generator=generator,
                device=device,
            )

    return mask


def _fill_rectangle_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    min_size: int,
    max_size: int,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    inpaint_width = int(
        torch.randint(min_size, max_size + 1, (), generator=generator, device=device).item()
    )
    inpaint_height = int(
        torch.randint(min_size, max_size + 1, (), generator=generator, device=device).item()
    )
    left = int(
        torch.randint(
            0,
            image_size - inpaint_width + 1,
            (),
            generator=generator,
            device=device,
        ).item()
    )
    top = int(
        torch.randint(
            0,
            image_size - inpaint_height + 1,
            (),
            generator=generator,
            device=device,
        ).item()
    )
    mask[:, top : top + inpaint_height, left : left + inpaint_width] = 1.0


def _fill_multi_rectangle_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    min_size: int,
    max_size: int,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    count = int(torch.randint(2, 5, (), generator=generator, device=device).item())
    piece_max_size = max(min_size, max_size // 2)
    for _ in range(count):
        _fill_rectangle_mask(
            mask,
            image_size=image_size,
            min_size=max(1, min_size // 2),
            max_size=piece_max_size,
            torch=torch,
            generator=generator,
            device=device,
        )


def _fill_brush_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    min_size: int,
    max_size: int,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    stroke_count = int(torch.randint(1, 4, (), generator=generator, device=device).item())
    min_radius = max(1, min_size // 12)
    max_radius = max(min_radius, max_size // 8)
    for _ in range(stroke_count):
        radius = int(
            torch.randint(min_radius, max_radius + 1, (), generator=generator, device=device).item()
        )
        point_count = int(torch.randint(3, 8, (), generator=generator, device=device).item())
        x = int(torch.randint(0, image_size, (), generator=generator, device=device).item())
        y = int(torch.randint(0, image_size, (), generator=generator, device=device).item())
        for _ in range(point_count):
            next_x = int(
                torch.randint(0, image_size, (), generator=generator, device=device).item()
            )
            next_y = int(
                torch.randint(0, image_size, (), generator=generator, device=device).item()
            )
            _fill_brush_segment(mask, x, y, next_x, next_y, radius=radius)
            x = next_x
            y = next_y


def _fill_brush_segment(
    mask: torch.Tensor,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    radius: int,
) -> None:
    steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
    stamp_stride = max(1, radius)
    for step in range(0, steps + 1, stamp_stride):
        fraction = step / steps
        x = round(start_x + (end_x - start_x) * fraction)
        y = round(start_y + (end_y - start_y) * fraction)
        _fill_square_stamp(mask, x, y, radius=radius)


def _fill_square_stamp(mask: torch.Tensor, center_x: int, center_y: int, *, radius: int) -> None:
    image_size = mask.shape[-1]
    left = max(0, center_x - radius)
    right = min(image_size, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(image_size, center_y + radius + 1)
    mask[:, top:bottom, left:right] = 1.0


def _fill_round_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    min_size: int,
    max_size: int,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    diameter = int(
        torch.randint(min_size, max_size + 1, (), generator=generator, device=device).item()
    )
    radius = max(1, diameter // 2)
    if radius * 2 < image_size:
        center_x = int(
            torch.randint(
                radius, image_size - radius, (), generator=generator, device=device
            ).item()
        )
        center_y = int(
            torch.randint(
                radius, image_size - radius, (), generator=generator, device=device
            ).item()
        )
    else:
        center_x = int(torch.randint(0, image_size, (), generator=generator, device=device).item())
        center_y = int(torch.randint(0, image_size, (), generator=generator, device=device).item())

    y_coordinates = torch.arange(image_size, device=device).view(image_size, 1)
    x_coordinates = torch.arange(image_size, device=device).view(1, image_size)
    circle = (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2 <= radius**2
    mask[:, circle] = 1.0


def _fill_blob_mask(
    mask: torch.Tensor,
    *,
    image_size: int,
    min_size: int,
    max_size: int,
    torch: ModuleType,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    blob_count = int(torch.randint(1, 4, (), generator=generator, device=device).item())
    min_radius = max(1, min_size // 3)
    max_radius = max(min_radius, max_size // 2)
    y_coordinates = torch.arange(image_size, device=device).view(image_size, 1)
    x_coordinates = torch.arange(image_size, device=device).view(1, image_size)
    for _ in range(blob_count):
        center_x = int(torch.randint(0, image_size, (), generator=generator, device=device).item())
        center_y = int(torch.randint(0, image_size, (), generator=generator, device=device).item())
        radius_x = int(
            torch.randint(min_radius, max_radius + 1, (), generator=generator, device=device).item()
        )
        radius_y = int(
            torch.randint(min_radius, max_radius + 1, (), generator=generator, device=device).item()
        )
        ellipse = (
            ((x_coordinates - center_x).float() / max(1, radius_x)) ** 2
            + ((y_coordinates - center_y).float() / max(1, radius_y)) ** 2
        ) <= 1.0
        mask[:, ellipse] = 1.0


def _fill_extension_mask(mask: torch.Tensor, *, image_size: int, size: int, side: int) -> None:
    if side == 0:
        mask[:, :size, :] = 1.0
    elif side == 1:
        mask[:, image_size - size :, :] = 1.0
    elif side == 2:
        mask[:, :, :size] = 1.0
    else:
        mask[:, :, image_size - size :] = 1.0
