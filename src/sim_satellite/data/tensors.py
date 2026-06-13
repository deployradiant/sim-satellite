"""Tensor conversion for prepared satellite image/mask pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS, OPENEARTHMAP_RGB
from sim_satellite.model import ModelConfig


def load_image_tensor(path: Path, *, torch: Any, config: ModelConfig) -> Any:
    image = Image.open(path).convert("RGB")
    image = image.resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
    values = torch.tensor(list(image.getdata()), dtype=torch.float32)
    return values.view(config.image_size, config.image_size, 3).permute(2, 0, 1) / 255.0


def load_mask_tensor(path: Path, *, torch: Any, config: ModelConfig) -> Any:
    mask = Image.open(path)
    mask = mask.resize((config.image_size, config.image_size), Image.Resampling.NEAREST)
    labels = _mask_to_labels(mask, torch, config)
    return (
        torch.nn.functional.one_hot(labels, num_classes=config.mask_channels)
        .permute(2, 0, 1)
        .float()
    )


def _mask_to_labels(mask: Any, torch: Any, config: ModelConfig) -> Any:
    if mask.mode in {"RGB", "RGBA"}:
        labels = torch.zeros((config.image_size, config.image_size), dtype=torch.long)
        pixels = torch.tensor(list(mask.convert("RGB").getdata()), dtype=torch.uint8)
        pixels = pixels.view(config.image_size, config.image_size, 3)
        for name, rgb in OPENEARTHMAP_RGB.items():
            class_id = OPENEARTHMAP_CLASS_IDS[name]

            # ignore class if we have less channels for the mask than classes
            if class_id >= config.mask_channels:
                continue
            color = torch.tensor(rgb, dtype=torch.uint8)
            labels[(pixels == color).all(dim=-1)] = class_id
        return labels

    labels = torch.tensor(list(mask.convert("L").getdata()), dtype=torch.long)
    labels = labels.view(config.image_size, config.image_size)
    return labels.clamp(0, config.mask_channels - 1)
