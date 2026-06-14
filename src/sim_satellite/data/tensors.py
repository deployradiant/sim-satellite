"""Tensor conversion for prepared satellite image/mask pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS, OPENEARTHMAP_RGB
from sim_satellite.model import ModelConfig

_RGB_CLASS_IDS = {rgb: OPENEARTHMAP_CLASS_IDS[name] for name, rgb in OPENEARTHMAP_RGB.items()}
_RGB_MASK_MODES = {"RGB", "RGBA", "P"}
_GRAYSCALE_CLASS_ID_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L"}


def load_image_tensor(
    path: Path,
    *,
    torch: Any,
    config: ModelConfig,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Any:
    image = Image.open(path).convert("RGB")
    return image_to_tensor(image, torch=torch, config=config, crop_box=crop_box)


def image_to_tensor(
    image: Image.Image,
    *,
    torch: Any,
    config: ModelConfig,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Any:
    if crop_box is not None:
        image = image.crop(crop_box)
    image = image.resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
    values = torch.tensor(list(_image_data(image)), dtype=torch.float32)
    return values.view(config.image_size, config.image_size, 3).permute(2, 0, 1) / 255.0


def load_mask_tensor(
    path: Path,
    *,
    torch: Any,
    config: ModelConfig,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Any:
    mask = Image.open(path)
    return mask_to_tensor(mask, torch=torch, config=config, crop_box=crop_box)


def mask_to_tensor(
    mask: Image.Image,
    *,
    torch: Any,
    config: ModelConfig,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Any:
    if crop_box is not None:
        mask = mask.crop(crop_box)
    mask = mask.resize((config.image_size, config.image_size), Image.Resampling.NEAREST)
    labels = torch.tensor(mask_to_class_ids(mask, config), dtype=torch.long)
    labels = labels.view(config.image_size, config.image_size)
    return (
        torch.nn.functional.one_hot(labels, num_classes=config.mask_channels)
        .permute(2, 0, 1)
        .float()
    )


def mask_to_class_ids(mask: Image.Image, config: ModelConfig) -> list[int]:
    if mask.mode in _RGB_MASK_MODES:
        rgb_mask = mask.convert("RGB")
        return [_class_id_from_rgb(pixel, config) for pixel in _image_data(rgb_mask)]

    if mask.mode in _GRAYSCALE_CLASS_ID_MODES:
        return [_class_id_from_grayscale(int(value), config) for value in _image_data(mask)]

    raise ValueError(
        f"Unsupported mask mode {mask.mode!r}; expected RGB/RGBA colors or grayscale class ids"
    )


def _class_id_from_rgb(pixel: tuple[int, int, int], config: ModelConfig) -> int:
    class_id = _RGB_CLASS_IDS.get(pixel)
    if class_id is None:
        raise ValueError(f"Mask contains unmapped RGB color: {pixel}")
    if class_id >= config.mask_channels:
        raise ValueError(
            f"Mask color {pixel} maps to class {class_id}, "
            f"but mask_channels is {config.mask_channels}"
        )
    return class_id


def _class_id_from_grayscale(value: int, config: ModelConfig) -> int:
    if value < 0 or value >= config.mask_channels:
        raise ValueError(
            f"Mask contains class id {value}, but mask_channels is {config.mask_channels}"
        )
    return value


def _image_data(image: Image.Image) -> Any:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return get_flattened_data()
    return image.getdata()
