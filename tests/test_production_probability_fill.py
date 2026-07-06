from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES


def _load_script() -> Any:
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "train_production_probability_fill_labeller.py"
    )
    spec = importlib.util.spec_from_file_location(
        "production_probability_fill_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()
ProductionProbabilityFillModel = _SCRIPT.ProductionProbabilityFillModel
_production_condition_features = _SCRIPT._production_condition_features
_resize_probability_map = _SCRIPT._resize_probability_map
_unknown_region_cross_entropy = _SCRIPT._unknown_region_cross_entropy


def test_probability_fill_condition_features_include_full_probabilities() -> None:
    class_count = len(OPENEARTHMAP_CLASSES)
    raw = torch.rand((1, class_count, 2, 2), dtype=torch.float32)
    probabilities = raw / raw.sum(dim=1, keepdim=True)
    predictions = torch.tensor([[[1, 2], [3, 4]]])
    confidence = torch.tensor([[[0.95, 0.40], [0.81, 0.20]]])
    known = torch.tensor([[[True, False], [True, False]]])

    features = _production_condition_features(
        probabilities,
        predictions,
        confidence,
        known,
        num_classes=class_count,
    )

    assert features.shape == (1, (class_count * 2) + 2, 2, 2)
    assert torch.equal(features[:, :class_count], probabilities)
    one_hot = features[:, class_count : class_count * 2]
    known_channel = features[:, class_count * 2]
    confidence_channel = features[:, (class_count * 2) + 1]
    assert one_hot[0, 1, 0, 0] == 1.0
    assert one_hot[:, :, 0, 1].sum() == 0.0
    assert torch.equal(known_channel, known.float())
    assert torch.equal(confidence_channel, confidence)


def test_probability_fill_model_residual_can_fall_back_to_production_logits() -> None:
    class_count = len(OPENEARTHMAP_CLASSES)
    model = ProductionProbabilityFillModel(num_classes=class_count)
    model.eval()
    for parameter in model.head.parameters():
        parameter.data.zero_()

    images = torch.rand((2, 3, 32, 32), dtype=torch.float32)
    condition = torch.rand((2, (class_count * 2) + 2, 32, 32), dtype=torch.float32)
    base_logits = torch.randn((2, class_count, 32, 32), dtype=torch.float32)

    with torch.inference_mode():
        logits = model(images, condition, base_logits)

    assert logits.shape == base_logits.shape
    assert torch.equal(logits, base_logits)


def test_probability_fill_loss_supervises_only_unknown_labeled_pixels() -> None:
    building_id = OPENEARTHMAP_CLASSES.index("building")
    road_id = OPENEARTHMAP_CLASSES.index("road")
    labels = torch.tensor([[[road_id, building_id], [0, road_id]]])
    known = torch.tensor([[[True, False], [False, True]]])
    weights = torch.ones(len(OPENEARTHMAP_CLASSES), dtype=torch.float32)

    good_logits = torch.zeros((1, len(OPENEARTHMAP_CLASSES), 2, 2), dtype=torch.float32)
    good_logits[:, road_id, 0, 0] = -10.0
    good_logits[:, building_id, 0, 1] = 10.0
    bad_logits = good_logits.clone()
    bad_logits[:, building_id, 0, 1] = -10.0
    bad_logits[:, road_id, 0, 1] = 10.0

    good_loss = _unknown_region_cross_entropy(
        good_logits,
        labels,
        known_mask=known,
        class_weights=weights,
    )
    bad_loss = _unknown_region_cross_entropy(
        bad_logits,
        labels,
        known_mask=known,
        class_weights=weights,
    )

    assert good_loss is not None
    assert bad_loss is not None
    assert good_loss.item() < 0.01
    assert bad_loss.item() > 10.0


def test_probability_map_resize_preserves_per_pixel_distribution() -> None:
    probabilities = torch.rand((len(OPENEARTHMAP_CLASSES), 3, 5), dtype=torch.float32)
    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

    resized = _resize_probability_map(
        probabilities,
        image_size=8,
        device=torch.device("cpu"),
    )

    assert resized.shape == (len(OPENEARTHMAP_CLASSES), 8, 8)
    assert torch.allclose(resized.sum(dim=0), torch.ones((8, 8)), atol=1e-6)
