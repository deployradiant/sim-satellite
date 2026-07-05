"""Shared mask-conditioned image generation primitives."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from sim_satellite.model import ModelConfig


def generate_conditioned_images(
    model: Any,
    semantic_mask: Any,
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    seed: int,
    steps: int,
    guidance_scale: float,
    precision: str = "float32",
    noise: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_shape = (
        semantic_mask.shape[0],
        model_config.image_channels,
        model_config.image_size,
        model_config.image_size,
    )
    if noise is None:
        noise = torch.randn(image_shape, generator=generator, device="cpu")
    elif tuple(noise.shape) != image_shape:
        raise ValueError(
            f"Generation noise shape {tuple(noise.shape)} does not match images {image_shape}"
        )
    image = noise.to(device)
    timesteps, step_size = flow_inference_tensors(steps, torch=torch, device=device)

    with torch.inference_mode():
        for completed_steps, timestep in enumerate(timesteps, start=1):
            with autocast_context(torch, precision, device):
                velocity = guided_prediction(
                    model,
                    image,
                    semantic_mask,
                    timestep,
                    guidance_scale=guidance_scale,
                    torch=torch,
                )
            image = image - velocity * step_size
            if progress is not None:
                progress(completed_steps, steps)
    return (image.clamp(-1.0, 1.0) + 1.0) / 2.0


def flow_inference_tensors(steps: int, *, torch: Any, device: Any) -> tuple[Any, float]:
    step_size = 1.0 / float(steps)
    timesteps = torch.tensor(
        [(step - 0.5) * step_size for step in reversed(range(1, steps + 1))],
        dtype=torch.float32,
        device=device,
    ).view(steps, 1)
    return timesteps, step_size


def guided_prediction(
    model: Any,
    image: Any,
    semantic_mask: Any,
    timestep: Any,
    *,
    guidance_scale: float,
    torch: Any,
) -> Any:
    """Apply classifier-free guidance between empty-mask and conditional predictions.

    A scale of 1.0 is the plain mask-conditioned prediction. Values above 1.0
    extrapolate farther toward the semantic-mask condition, which can improve
    mask adherence at the cost of more artificial-looking samples.
    """

    conditional = model(image, semantic_mask, timestep)
    if guidance_scale == 1.0:
        return conditional

    empty_mask = torch.zeros_like(semantic_mask)
    unconditional = model(image, empty_mask, timestep)
    return unconditional + guidance_scale * (conditional - unconditional)


def autocast_context(torch: Any, precision: str, device: Any) -> Any:
    if precision == "float32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be one of: bf16, float32")
    return torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16)


def _device_type(device: Any) -> str:
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    return str(device).split(":", maxsplit=1)[0]
