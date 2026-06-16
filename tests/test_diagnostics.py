from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.prepared import PREPARED_DATA_VERSION, PreparedSample
from sim_satellite.diagnostics import (
    GeneratedTarget,
    ReferenceInput,
    build_reference_stats,
    class_mean_distance,
    image_statistics,
    load_embedded_mask,
    load_rgb_image,
    run_diagnostics,
    run_generated_diagnostics,
    run_training_generation_diagnostics,
    write_diagnostics_payload,
)
from sim_satellite.model import ModelConfig


def test_image_statistics_report_saturation_and_rgb_std(tmp_path: Path) -> None:
    gray_path = tmp_path / "gray.png"
    color_path = tmp_path / "color.png"
    Image.new("RGB", (2, 2), (128, 128, 128)).save(gray_path)
    color = Image.new("RGB", (2, 2))
    color.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]
    )
    color.save(color_path)

    gray = image_statistics(load_rgb_image(gray_path), path=gray_path)
    colorful = image_statistics(load_rgb_image(color_path), path=color_path)

    assert gray.saturation_mean == 0.0
    assert gray.rgb_std == 0.0
    assert colorful.saturation_mean == 1.0
    assert colorful.rgb_std > 0.4


def test_class_mean_distance_uses_reference_class_baselines(tmp_path: Path) -> None:
    reference_tensor = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )
    reference_image = tmp_path / "reference.png"
    reference_mask_path = tmp_path / "reference-mask.png"
    reference_pil = Image.new("RGB", (2, 2))
    reference_pil.putdata(
        [
            (255, 0, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 0, 255),
        ]
    )
    reference_pil.save(reference_image)
    reference_mask_image = Image.new("L", (2, 2))
    reference_mask_image.putdata([0, 0, 1, 1])
    reference_mask_image.save(reference_mask_path)
    reference_mask = torch.tensor([[0, 0], [1, 1]])
    reference_stats = build_reference_stats(
        (ReferenceInput(image_path=reference_image, mask_path=reference_mask_path),)
    )

    target_image = reference_tensor.clone()
    target_image[reference_mask == 0] = torch.tensor([0.5, 0.0, 0.0])

    distance, classes_compared = class_mean_distance(
        target_image,
        reference_mask,
        reference_stats.class_means,
    )

    assert reference_stats.image_count == 1
    assert classes_compared == 2
    assert distance == pytest.approx(0.25)


def test_run_diagnostics_compares_target_to_references(tmp_path: Path) -> None:
    reference_image = tmp_path / "reference.png"
    reference_mask = tmp_path / "reference-mask.png"
    target_image = tmp_path / "target.png"
    target_mask = tmp_path / "target-mask.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(reference_image)
    Image.new("L", (2, 2), 1).save(reference_mask)
    Image.new("RGB", (2, 2), (128, 128, 128)).save(target_image)
    Image.new("L", (2, 2), 1).save(target_mask)

    payload = run_diagnostics(
        (target_image,),
        target_masks=(target_mask,),
        references=(ReferenceInput(image_path=reference_image, mask_path=reference_mask),),
    )

    target = payload["targets"][0]
    assert payload["reference"]["class_count"] == 1
    assert target["saturation_mean"] == 0.0
    assert target["saturation_delta"] == pytest.approx(-1.0)
    assert target["classes_compared"] == 1
    assert target["class_mean_l2"] == pytest.approx(0.8672, abs=0.0001)


def test_tensor_file_loading_supports_inference_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "sample.pt"
    torch.save(
        {
            "image": torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]]),
            "semantic_mask": torch.tensor([[[[0.0]], [[1.0]]]]),
        },
        output_path,
    )

    image = load_rgb_image(output_path)
    mask = load_embedded_mask(output_path)

    assert image.shape == (1, 1, 3)
    assert image[0, 0].tolist() == [0.0, 0.5, 1.0]
    assert mask is not None
    assert mask.tolist() == [[1]]


def test_run_generated_diagnostics_accepts_in_memory_targets() -> None:
    payload = run_generated_diagnostics(
        (
            GeneratedTarget(
                path="generated-sample",
                image=torch.tensor([[[0.25, 0.5, 0.75]]]),
                mask=torch.tensor([[0]]),
            ),
        )
    )

    assert payload["reference"] is None
    assert payload["targets"][0]["path"] == "generated-sample"
    assert payload["targets"][0]["rgb_mean"] == pytest.approx([0.25, 0.5, 0.75])


def test_training_generation_diagnostics_uses_preloaded_samples(tmp_path: Path) -> None:
    image = torch.full((32, 32, 3), 128, dtype=torch.uint8)
    mask = torch.zeros((32, 32), dtype=torch.uint8)
    prepared_path = tmp_path / "prepared.pt"
    torch.save(
        {
            "version": PREPARED_DATA_VERSION,
            "image": image,
            "mask": mask,
            "size": (32, 32),
        },
        prepared_path,
    )
    sample = SatelliteSample(
        image_path=tmp_path / "image.png",
        mask_path=tmp_path / "mask.png",
        prepared_path=prepared_path,
        split="train",
    )
    model = _ZeroVelocityModel()

    payload = run_training_generation_diagnostics(
        model=model,
        torch=torch,
        model_config=ModelConfig(image_size=32, mask_channels=2),
        device=torch.device("cpu"),
        samples=(sample,),
        preloaded_samples={
            sample: PreparedSample(image=image, mask=mask, size=(32, 32)),
        },
        sample_count=1,
        reference_sample_count=1,
        steps=1,
        guidance_scale=1.0,
        seed=7,
    )

    assert model.training
    assert payload["reference"]["image_count"] == 1
    assert payload["targets"][0]["classes_compared"] == 1


def test_write_diagnostics_payload_outputs_json(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.json"

    write_diagnostics_payload(path, {"targets": []})

    assert json.loads(path.read_text(encoding="utf-8")) == {"targets": []}


class _ZeroVelocityModel:
    training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def __call__(
        self,
        image: torch.Tensor,
        _semantic_mask: torch.Tensor,
        _timestep: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(image)
