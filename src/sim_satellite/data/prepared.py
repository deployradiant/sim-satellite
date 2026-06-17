"""Prepared sample data for training-time loading."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import rasterio
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.warp import transform

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.tensors import image_to_uint8_tensor, mask_to_label_tensor
from sim_satellite.model import ModelConfig

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

PREPARED_DATA_VERSION = 2


@dataclass(frozen=True)
class PreparedSample:
    image: Tensor
    mask: Tensor
    size: tuple[int, int]
    coordinates: Tensor | None


def prepared_coordinates_tuple(prepared: PreparedSample) -> tuple[float, float] | None:
    if prepared.coordinates is None:
        return None
    values = prepared.coordinates.float().view(-1).tolist()
    if len(values) != 2:
        raise RuntimeError(f"Prepared coordinates must have two values, got: {values}")
    return float(values[0]), float(values[1])


def read_geo_data(image_path: Path) -> tuple[float, float] | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            image = rasterio.open(image_path)
        with image:
            if image.crs is None:
                return None
            bounds = image.bounds
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.top + bounds.bottom) / 2
            longitude, latitude = cast(
                tuple[list[float], list[float]],
                transform(
                    image.crs,
                    "EPSG:4326",
                    [center_x],
                    [center_y],
                ),
            )
    except RasterioIOError:
        return None

    return float(longitude[0]), float(latitude[0])


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
            source_coordinates=read_geo_data(sample.image_path),
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
    source_coordinates: tuple[float, float] | None,
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
    coords_tensor = (
        None
        if source_coordinates is None
        else torch.tensor(list(source_coordinates), dtype=torch.float32)
    )

    torch.save(
        {
            "version": PREPARED_DATA_VERSION,
            "image": image_tensor.contiguous(),
            "mask": mask_tensor.contiguous(),
            "size": image_size,
            "coordinates": coords_tensor,
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
    coordinates = payload.get("coordinates")
    if (
        not isinstance(image, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not isinstance(size, tuple)
        or (coordinates is not None and not isinstance(coordinates, torch.Tensor))
    ):
        raise RuntimeError(
            f"Prepared sample data is missing image, mask, size, or coordinates: {path}"
        )
    coordinates_tensor = cast(Tensor, coordinates) if coordinates is not None else None
    if coordinates_tensor is not None and tuple(coordinates_tensor.shape) != (2,):
        raise RuntimeError(f"Prepared sample coordinates must have shape (2,): {path}")
    return PreparedSample(
        image=cast(Tensor, image),
        mask=cast(Tensor, mask),
        size=size,
        coordinates=coordinates_tensor,
    )
