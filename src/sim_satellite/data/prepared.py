"""Prepared sample data for training-time loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.tensors import image_to_uint8_tensor, mask_to_label_tensor
from sim_satellite.model import ModelConfig

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

PREPARED_DATA_VERSION = 3


@dataclass(frozen=True)
class PreparedSample:
    image: Tensor
    mask: Tensor
    size: tuple[int, int]


def write_prepared_sample(
    sample: SatelliteSample,
    *,
    torch: Any,
    config: ModelConfig,
    force: bool = False,
) -> None:
    with Image.open(sample.image_path) as image, Image.open(sample.mask_path) as mask:
        write_prepared_data(
            sample.prepared_path,
            image=image,
            mask=mask,
            torch=torch,
            config=config,
            force=force,
            source_description=f"{sample.image_path} / {sample.mask_path}",
        )


def write_prepared_data(
    prepared_path: Path,
    *,
    image: Image.Image,
    mask: Image.Image,
    torch: Any,
    config: ModelConfig,
    force: bool = False,
    source_description: str,
) -> None:
    if prepared_path.exists() and not force:
        return

    image_tensor = image_to_uint8_tensor(image, torch=torch)
    image_size = image.size
    mask_size = mask.size
    mask_tensor = mask_to_label_tensor(mask, torch=torch, config=config).to(torch.uint8)

    if image_size != mask_size:
        raise ValueError(
            f"Image and mask sizes must match for prepared data: "
            f"{source_description} has sizes {image_size} and {mask_size}"
        )

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = prepared_path.with_suffix(f"{prepared_path.suffix}.tmp")

    torch.save(
        {
            "version": PREPARED_DATA_VERSION,
            "image": image_tensor.contiguous(),
            "mask": mask_tensor.contiguous(),
            "size": image_size,
        },
        partial_path,
    )
    partial_path.replace(prepared_path)


def load_prepared_sample(path: Path, *, torch: Any) -> PreparedSample:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Prepared sample data must be a mapping: {path}")
    if payload.get("version") != PREPARED_DATA_VERSION:
        raise RuntimeError(
            f"Unsupported prepared sample data version in {path}. "
            "Run `sim-satellite openearthmap --force-prepared` to rebuild prepared tensors."
        )

    image = payload.get("image")
    mask = payload.get("mask")
    size = payload.get("size")
    if (
        not isinstance(image, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not isinstance(size, tuple)
    ):
        raise RuntimeError(f"Prepared sample data is missing image, mask, or size: {path}")
    return PreparedSample(
        image=cast(Tensor, image),
        mask=cast(Tensor, mask),
        size=size,
    )
