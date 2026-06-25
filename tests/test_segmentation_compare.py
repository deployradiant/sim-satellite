from __future__ import annotations

import pytest
from click.testing import CliRunner

from sim_satellite.cli import cli
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.segmentation_compare import (
    DEFAULT_MODELS,
    _build_model,
    _segmentation_metrics,
    _tile_windows,
)


def test_tile_windows_cover_image_with_grid() -> None:
    assert _tile_windows((10, 8), 3) == (
        (0, 0, 3, 3),
        (3, 0, 7, 3),
        (7, 0, 10, 3),
        (0, 3, 3, 5),
        (3, 3, 7, 5),
        (7, 3, 10, 5),
        (0, 5, 3, 8),
        (3, 5, 7, 8),
        (7, 5, 10, 8),
    )


def test_segmentation_metrics_exclude_unlabeled_from_mean_iou() -> None:
    import torch

    confusion = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), len(OPENEARTHMAP_CLASSES)),
        dtype=torch.long,
    )
    confusion[0, 0] = 10
    confusion[4, 4] = 3
    confusion[4, 8] = 1
    confusion[8, 8] = 2
    metrics = _segmentation_metrics(confusion, include_unlabeled_iou=False, torch=torch)

    assert metrics["class_iou"]["unlabeled"] == 1.0
    assert metrics["class_iou"]["road"] == 0.75
    assert metrics["class_iou"]["building"] == pytest.approx(2 / 3)
    assert metrics["mean_iou"] == pytest.approx((0.75 + 2 / 3) / 2)


def test_segmentation_models_return_class_logits() -> None:
    import torch

    batch = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    for model_name in DEFAULT_MODELS:
        model = _build_model(model_name, num_classes=len(OPENEARTHMAP_CLASSES), torch=torch)
        logits = model(batch)
        assert tuple(logits.shape) == (2, len(OPENEARTHMAP_CLASSES), 32, 32)


def test_cli_exposes_segmentation_compare_command() -> None:
    result = CliRunner().invoke(cli, ["segmentation-compare", "--help"])

    assert result.exit_code == 0
    assert "supervised segmentation models" in result.output
