from __future__ import annotations

from click.testing import CliRunner

from sim_satellite.cli import cli
from sim_satellite.segmentation_train import (
    _class_weights_from_counts,
    _multiclass_dice_loss,
)


def test_class_weights_downweight_unlabeled_and_rare_classes_weigh_more() -> None:
    import torch

    weights = _class_weights_from_counts(
        torch.tensor([1000, 100, 10]),
        power=0.5,
        unlabeled_weight=0.25,
        torch=torch,
    )

    assert weights[0] < weights[1] < weights[2]


def test_multiclass_dice_loss_is_lower_for_correct_logits() -> None:
    import torch

    labels = torch.tensor([[[1, 2], [1, 2]]])
    correct_logits = torch.full((1, 3, 2, 2), -5.0)
    correct_logits[:, 1, :, 0] = 5.0
    correct_logits[:, 2, :, 1] = 5.0
    wrong_logits = torch.zeros((1, 3, 2, 2))

    assert _multiclass_dice_loss(
        correct_logits,
        labels,
        include_unlabeled=False,
        torch=torch,
    ) < _multiclass_dice_loss(
        wrong_logits,
        labels,
        include_unlabeled=False,
        torch=torch,
    )


def test_cli_exposes_segmentation_train_command() -> None:
    result = CliRunner().invoke(cli, ["segmentation-train", "--help"])

    assert result.exit_code == 0
    assert "custom supervised segmentation checkpoint" in result.output
