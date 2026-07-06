from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES


def _load_script() -> Any:
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "train_production_conditioned_mask_completion_vae.py"
    )
    spec = importlib.util.spec_from_file_location("mask_completion_vae_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()
ProductionConditionedMaskCompletionVAE = _SCRIPT.ProductionConditionedMaskCompletionVAE
_conditional_kl_divergence = _SCRIPT._conditional_kl_divergence
_production_condition_features = _SCRIPT._production_condition_features
_unknown_region_cross_entropy = _SCRIPT._unknown_region_cross_entropy


def test_mask_completion_vae_forward_and_prior_shapes() -> None:
    model = ProductionConditionedMaskCompletionVAE(
        num_classes=len(OPENEARTHMAP_CLASSES),
        latent_channels=4,
    )
    model.eval()
    images = torch.rand((2, 3, 32, 32), dtype=torch.float32)
    condition = torch.rand(
        (2, len(OPENEARTHMAP_CLASSES) + 2, 32, 32),
        dtype=torch.float32,
    )
    labels = torch.randint(0, len(OPENEARTHMAP_CLASSES), (2, 32, 32))

    with torch.inference_mode():
        output = model(images, condition, labels=labels)
        prior_logits = model.predict_prior(images, condition)

    assert output.logits.shape == (2, len(OPENEARTHMAP_CLASSES), 32, 32)
    assert prior_logits.shape == output.logits.shape
    assert output.prior_mu.shape == (2, 4, 8, 8)
    assert output.posterior_mu.shape == output.prior_mu.shape
    assert output.prior_logvar.shape == output.prior_mu.shape
    assert output.posterior_logvar.shape == output.prior_mu.shape


def test_mask_completion_vae_kl_is_zero_for_matching_distributions() -> None:
    mu = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    logvar = torch.zeros_like(mu)

    kl = _conditional_kl_divergence(mu, logvar, mu, logvar)

    assert kl.item() == pytest.approx(0.0)


def test_mask_completion_condition_features_hide_unknown_labels() -> None:
    predictions = torch.tensor([[[1, 2], [3, 4]]])
    confidence = torch.tensor([[[0.95, 0.40], [0.81, 0.20]]])
    known = torch.tensor([[[True, False], [True, False]]])

    features = _production_condition_features(
        predictions,
        confidence,
        known,
        num_classes=len(OPENEARTHMAP_CLASSES),
    )

    one_hot = features[:, : len(OPENEARTHMAP_CLASSES)]
    assert one_hot[0, 1, 0, 0] == 1.0
    assert one_hot[:, :, 0, 1].sum() == 0.0
    assert torch.equal(features[:, len(OPENEARTHMAP_CLASSES)], known.float())
    assert torch.equal(features[:, len(OPENEARTHMAP_CLASSES) + 1], confidence)


def test_mask_completion_loss_supervises_only_unknown_labeled_pixels() -> None:
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
