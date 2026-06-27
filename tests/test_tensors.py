from __future__ import annotations

import pytest
import torch
from PIL import Image

from sim_satellite.data.tensors import image_to_uint8_tensor, mask_to_class_ids, mask_to_tensor
from sim_satellite.model import ModelConfig

OPENEARTHMAP_CLASSES = (
    "unlabeled",
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "building",
)
OPENEARTHMAP_RGB = {
    "unlabeled": (0, 0, 0),
    "bareland": (128, 0, 0),
    "rangeland": (0, 255, 36),
    "developed space": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "agriculture land": (75, 181, 73),
    "building": (222, 31, 7),
}


def test_masks_decode_openearthmap_palette_and_grayscale_ids() -> None:
    rgb_mask = Image.new("RGB", (len(OPENEARTHMAP_CLASSES), 1))
    rgb_mask.putdata([OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES])
    palette_mask = Image.new("P", (len(OPENEARTHMAP_CLASSES), 1))
    palette = [0] * 768
    for class_id, name in enumerate(OPENEARTHMAP_CLASSES):
        rgb = OPENEARTHMAP_RGB[name]
        palette[class_id * 3 : class_id * 3 + 3] = rgb
    palette_mask.putpalette(palette)
    palette_mask.putdata(range(len(OPENEARTHMAP_CLASSES)))
    grayscale_mask = Image.new("L", (2, 1))
    grayscale_mask.putdata([0, 8])

    assert mask_to_class_ids(
        rgb_mask,
        ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
    ) == list(range(len(OPENEARTHMAP_CLASSES)))
    assert mask_to_class_ids(
        palette_mask,
        ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
    ) == list(range(len(OPENEARTHMAP_CLASSES)))
    assert mask_to_class_ids(grayscale_mask, ModelConfig(mask_channels=9)) == [0, 8]


def test_masks_reject_unmapped_or_out_of_range_classes() -> None:
    with pytest.raises(ValueError, match="unmapped RGB color"):
        mask_to_class_ids(
            Image.new("RGB", (1, 1), (1, 2, 3)),
            ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
        )

    with pytest.raises(ValueError, match="maps to class 8"):
        mask_to_class_ids(
            Image.new("RGB", (1, 1), OPENEARTHMAP_RGB["building"]),
            ModelConfig(mask_channels=8),
        )

    grayscale_mask = Image.new("L", (1, 1), 9)
    with pytest.raises(ValueError, match="class id 9"):
        mask_to_tensor(grayscale_mask, torch=torch, config=ModelConfig(mask_channels=9))

    with pytest.raises(ValueError, match="class id 9"):
        mask_to_class_ids(grayscale_mask, ModelConfig(mask_channels=9))


def test_image_to_uint8_tensor_uses_rgb_byte_values() -> None:
    image = Image.new("RGB", (8, 8), (0, 0, 0))
    image.putpixel((0, 0), (0, 127, 255))
    image.putpixel((1, 0), (255, 0, 127))

    tensor = image_to_uint8_tensor(image, torch=torch)

    assert tensor.shape == (8, 8, 3)
    assert torch.allclose(
        tensor[0, 0],
        torch.tensor([0, 127, 255], dtype=torch.uint8),
    )
    assert torch.allclose(
        tensor[0, 1],
        torch.tensor([255, 0, 127], dtype=torch.uint8),
    )
