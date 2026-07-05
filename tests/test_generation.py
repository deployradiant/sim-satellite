from __future__ import annotations

import torch

from sim_satellite.generation import guided_prediction


class _MaskOffsetModel:
    def __call__(self, image: torch.Tensor, semantic_mask: torch.Tensor, timestep: torch.Tensor):
        return image + semantic_mask.sum(dim=1, keepdim=True) + timestep.view(-1, 1, 1, 1)


def test_guided_prediction_extrapolates_from_unconditional_to_conditional() -> None:
    image = torch.zeros(1, 1, 2, 2)
    semantic_mask = torch.ones(1, 2, 2, 2)
    timestep = torch.tensor([0.25])

    guided = guided_prediction(
        _MaskOffsetModel(),
        image,
        semantic_mask,
        timestep,
        guidance_scale=1.5,
        torch=torch,
    )

    assert torch.allclose(guided, torch.full_like(image, 3.25))


def test_guided_prediction_scale_one_is_plain_conditional_prediction() -> None:
    image = torch.zeros(1, 1, 2, 2)
    semantic_mask = torch.ones(1, 2, 2, 2)
    timestep = torch.tensor([0.25])

    guided = guided_prediction(
        _MaskOffsetModel(),
        image,
        semantic_mask,
        timestep,
        guidance_scale=1.0,
        torch=torch,
    )

    assert torch.allclose(guided, torch.full_like(image, 2.25))
