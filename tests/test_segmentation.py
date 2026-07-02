from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner
from PIL import Image

import sim_satellite.segmentation.cli as segmentation_cli
from sim_satellite.cli import cli
from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import PreparedSample, write_prepared_sample
from sim_satellite.model import ModelConfig
from sim_satellite.segmentation.checkpoint import _load_segmentation_checkpoint, _save_checkpoint
from sim_satellite.segmentation.config import (
    SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    VAE_ASPP_UNET_MODEL_NAME,
    SegmentationEvaluationConfig,
    SegmentationPredictionConfig,
    SegmentationTrainingConfig,
)
from sim_satellite.segmentation.data import (
    _augment_segmentation_sample,
    _context_crop_box,
    _prepared_crop,
    _samples_for_split,
    _shuffled_limit_samples,
    _tile_windows,
)
from sim_satellite.segmentation.inference import (
    _apply_class_logit_biases,
    _read_image_records,
    _scaled_tile_windows_config,
    evaluate_segmentation_model,
    predict_segmentation_masks,
)
from sim_satellite.segmentation.metrics import (
    _class_weights_from_counts,
    _multiclass_dice_loss,
    _segmentation_metrics,
)
from sim_satellite.segmentation.models import build_segmentation_model
from sim_satellite.segmentation.training import train_segmentation_model
from sim_satellite.segmentation.vae import _load_vae_feature_encoder
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig


def test_segmentation_metrics_exclude_unlabeled_from_mean_iou() -> None:
    confusion = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), len(OPENEARTHMAP_CLASSES)),
        dtype=torch.long,
    )
    confusion[0, 0] = 10
    confusion[4, 4] = 3
    confusion[4, 8] = 1
    confusion[8, 8] = 2

    metrics = _segmentation_metrics(confusion, include_unlabeled_iou=False)

    assert metrics["class_iou"]["unlabeled"] == 1.0
    assert metrics["class_iou"]["road"] == pytest.approx(0.75)
    assert metrics["class_iou"]["building"] == pytest.approx(2 / 3)
    assert metrics["mean_iou"] == pytest.approx((0.75 + 2 / 3) / 2)
    assert metrics["labeled_pixel_accuracy"] == pytest.approx(5 / 6)


def test_segmentation_tile_windows_support_overlap_and_cover_edges() -> None:
    assert _tile_windows((17, 16), 16, stride=8) == (
        (0, 0, 16, 16),
        (1, 0, 17, 16),
    )


def test_segmentation_tile_scales_preserve_stride_ratio() -> None:
    assert _scaled_tile_windows_config(
        image_size=16,
        tile_stride=8,
        tile_scales=(0.75, 1.0, 1.25),
    ) == ((12, 6), (16, 8), (20, 10))


def test_segmentation_class_logit_biases_adjust_logits() -> None:
    road_id = OPENEARTHMAP_CLASSES.index("road")
    building_id = OPENEARTHMAP_CLASSES.index("building")
    logits = torch.zeros((len(OPENEARTHMAP_CLASSES), 1, 1), dtype=torch.float32)
    logits[road_id] = 1.0
    logits[building_id] = 0.9

    biased = _apply_class_logit_biases(logits, {"building": 0.2})

    assert biased[building_id, 0, 0] > biased[road_id, 0, 0]


def test_segmentation_class_weights_and_dice_loss_are_directional() -> None:
    weights = _class_weights_from_counts(
        torch.tensor([1000, 100, 10]),
        power=0.5,
        unlabeled_weight=0.25,
    )
    assert weights[0] < weights[1] < weights[2]

    labels = torch.tensor([[[1, 2], [1, 2]]])
    correct_logits = torch.full((1, 3, 2, 2), -5.0)
    correct_logits[:, 1, :, 0] = 5.0
    correct_logits[:, 2, :, 1] = 5.0
    wrong_logits = torch.zeros((1, 3, 2, 2))

    assert _multiclass_dice_loss(
        correct_logits,
        labels,
        include_unlabeled=False,
    ) < _multiclass_dice_loss(
        wrong_logits,
        labels,
        include_unlabeled=False,
    )


def test_context_segmentation_model_accepts_local_and_context_rgb() -> None:
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        input_channels=6,
        vae_latent_channels=4,
    )
    batch = torch.zeros((2, 6, 32, 32), dtype=torch.float32)
    vae_latents = torch.zeros((2, 4, 8, 8), dtype=torch.float32)

    logits = model(batch, vae_latents)

    assert tuple(logits.shape) == (2, len(OPENEARTHMAP_CLASSES), 32, 32)


def test_segmentation_model_requires_vae_latent_channels() -> None:
    with pytest.raises(ValueError, match="vae_latent_channels"):
        build_segmentation_model(
            num_classes=len(OPENEARTHMAP_CLASSES),
            vae_latent_channels=0,
        )


def test_smp_unet_b4_scse_model_does_not_require_vae_latents() -> None:
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=0,
        model_name=SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    )
    model.eval()
    image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    with torch.no_grad():
        logits = model(image)

    assert tuple(logits.shape) == (1, len(OPENEARTHMAP_CLASSES), 32, 32)


def test_vae_aspp_unet_segmenter_accepts_vae_latents() -> None:
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=24,
    )
    image = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
    vae_latents = torch.zeros((2, 24, 64, 64), dtype=torch.float32)

    logits = model(image, vae_latents)

    assert tuple(logits.shape) == (2, len(OPENEARTHMAP_CLASSES), 256, 256)


def test_prepared_context_crop_keeps_local_label_and_adds_context_channels() -> None:
    image = torch.zeros((8, 8, 3), dtype=torch.uint8)
    image[:, :, 0] = torch.arange(8, dtype=torch.uint8).view(1, 8)
    mask = torch.arange(64, dtype=torch.uint8).view(8, 8)
    prepared = PreparedSample(image=image, mask=mask, size=(8, 8))

    cropped_image, cropped_label = _prepared_crop(
        prepared,
        (2, 2, 6, 6),
        image_size=4,
        context_size=8,
    )

    assert tuple(cropped_image.shape) == (6, 4, 4)
    assert tuple(cropped_label.shape) == (4, 4)
    assert torch.equal(cropped_label, mask[2:6, 2:6].long())
    assert not torch.equal(cropped_image[:3], cropped_image[3:])


def test_context_crop_box_clamps_to_image_bounds() -> None:
    assert _context_crop_box((10, 10), (0, 0, 4, 4), 8) == (0, 0, 8, 8)
    assert _context_crop_box((10, 10), (6, 6, 10, 10), 8) == (2, 2, 10, 10)


def test_segmentation_augmentation_keeps_mask_aligned_with_image() -> None:
    import random

    image = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4) / 100.0
    label = torch.arange(4 * 4, dtype=torch.long).view(4, 4)

    augmented_image, augmented_label = _augment_segmentation_sample(
        image,
        label,
        SegmentationTrainingConfig(
            flip_probability=1.0,
            rotate_probability=0.0,
            color_jitter_strength=0.0,
            blur_probability=0.0,
        ),
        random.Random(123),
    )

    assert torch.allclose(augmented_image, torch.flip(image, dims=(1, 2)))
    assert torch.equal(augmented_label, torch.flip(label, dims=(0, 1)))


def test_segmentation_sample_limits_are_seeded_not_manifest_prefix(tmp_path: Path) -> None:
    samples = tuple(
        SatelliteSample(
            image_path=tmp_path / f"image-{index}.png",
            mask_path=tmp_path / f"mask-{index}.png",
            prepared_path=tmp_path / f"prepared-{index}.pt",
            split="train",
            source="fixture",
        )
        for index in range(8)
    )

    limited = _shuffled_limit_samples(samples, max_samples=3, seed=2026)

    assert len(limited) == 3
    assert limited != samples[:3]
    assert limited == _shuffled_limit_samples(samples, max_samples=3, seed=2026)


def test_segmentation_split_selection_accepts_custom_lists(tmp_path: Path) -> None:
    samples = tuple(
        SatelliteSample(
            image_path=tmp_path / f"image-{index}.png",
            mask_path=tmp_path / f"mask-{index}.png",
            prepared_path=tmp_path / f"prepared-{index}.pt",
            split=split,
            source="fixture",
        )
        for index, split in enumerate(("openearth-train", "custom-train", "custom-val"))
    )
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(samples, manifest_path)

    train_samples = _samples_for_split(manifest_path, "openearth-train, custom-train")
    all_samples = _samples_for_split(manifest_path, "all")

    assert [sample.split for sample in train_samples] == ["openearth-train", "custom-train"]
    assert len(all_samples) == 3


def test_segmentation_train_and_evaluate_use_prepared_tensors(tmp_path: Path) -> None:
    manifest_path = _write_segmentation_fixture(tmp_path, samples=4)
    vae_path = tmp_path / "vae.pt"
    _write_vae_checkpoint(vae_path)
    checkpoint_path = tmp_path / "segmenter.pt"

    result = train_segmentation_model(
        SegmentationTrainingConfig(
            manifest_path=str(manifest_path),
            checkpoint_path=str(checkpoint_path),
            vae_checkpoint_path=str(vae_path),
            validation_fraction=0.5,
            max_validation_samples=2,
            steps=2,
            batch_size=2,
            image_size=16,
            eval_every=2,
            device="cpu",
            cache_samples=8,
        )
    )

    assert checkpoint_path.exists()
    assert Path(result.metrics_path).exists()
    assert result.train_samples == 2
    assert result.validation_samples == 2
    assert result.parameters > 0

    evaluation = evaluate_segmentation_model(
        SegmentationEvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            manifest_path=str(manifest_path),
            max_samples=2,
            device="cpu",
        )
    )

    assert evaluation.samples == 2
    assert evaluation.tiles == 2
    assert 0.0 <= evaluation.mean_iou <= 1.0


def test_segmentation_predict_writes_class_id_and_color_masks(tmp_path: Path) -> None:
    vae_path = tmp_path / "vae.pt"
    _write_vae_checkpoint(vae_path)
    checkpoint_path = tmp_path / "segmenter.pt"
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=4,
    )
    _save_checkpoint(
        checkpoint_path,
        model,
        config=asdict(
            SegmentationTrainingConfig(
                image_size=16,
                vae_checkpoint_path=str(vae_path),
            )
        ),
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["model"] == VAE_ASPP_UNET_MODEL_NAME
    image_path = tmp_path / "unlabeled" / "tile.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (17, 19), (32, 64, 96)).save(image_path)
    images_manifest = tmp_path / "unlabeled.jsonl"
    images_manifest.write_text(
        json.dumps({"image": "unlabeled/tile.png", "source": "test-unlabeled"}) + "\n",
        encoding="utf-8",
    )

    result = predict_segmentation_masks(
        SegmentationPredictionConfig(
            checkpoint_path=str(checkpoint_path),
            images_manifest_path=str(images_manifest),
            output_dir=str(tmp_path / "predictions"),
            device="cpu",
        )
    )

    records = [
        json.loads(line)
        for line in Path(result.manifest_path).read_text(encoding="utf-8").splitlines()
    ]
    assert result.images == 1
    assert len(records) == 1
    assert Path(records[0]["mask"]).exists()
    assert Path(records[0]["color_mask"]).exists()
    assert Path(records[0]["confidence_mask"]).exists()
    assert records[0]["prediction_source"] == "segmentation-classifier"
    assert records[0]["class_logit_biases"] == {}
    assert records[0]["confidence_threshold"] == 0.0
    assert records[0]["feather_blending"] is False
    assert 0.0 <= records[0]["mean_confidence"] <= 1.0
    assert records[0]["high_confidence_fraction"] == 1.0
    with Image.open(records[0]["mask"]) as mask:
        assert mask.mode == "L"
        assert mask.size == (17, 19)
    with Image.open(records[0]["confidence_mask"]) as confidence:
        assert confidence.mode == "L"
        assert confidence.size == (17, 19)


def test_segmentation_predict_can_gate_low_confidence_pixels(tmp_path: Path) -> None:
    vae_path = tmp_path / "vae.pt"
    _write_vae_checkpoint(vae_path)
    checkpoint_path = tmp_path / "segmenter.pt"
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=4,
    )
    _save_checkpoint(
        checkpoint_path,
        model,
        config=asdict(
            SegmentationTrainingConfig(
                image_size=16,
                vae_checkpoint_path=str(vae_path),
            )
        ),
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )
    image_path = tmp_path / "unlabeled" / "tile.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (32, 64, 96)).save(image_path)
    images_manifest = tmp_path / "unlabeled.jsonl"
    images_manifest.write_text(
        json.dumps({"image": "unlabeled/tile.png", "source": "test-unlabeled"}) + "\n",
        encoding="utf-8",
    )

    result = predict_segmentation_masks(
        SegmentationPredictionConfig(
            checkpoint_path=str(checkpoint_path),
            images_manifest_path=str(images_manifest),
            output_dir=str(tmp_path / "predictions"),
            device="cpu",
            confidence_threshold=1.0,
        )
    )

    record = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert record["confidence_threshold"] == 1.0
    assert record["high_confidence_fraction"] == 0.0
    assert record["class_counts"] == {"unlabeled": 256}
    with Image.open(record["mask"]) as mask:
        assert set(mask.tobytes()) == {0}


def test_prediction_manifest_requires_object_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "images.jsonl"
    manifest_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image record must be an object"):
        _read_image_records(manifest_path)


def test_vae_checkpoint_path_resolves_from_classifier_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    vae_path = artifact_root / "openearthmap-custom-vae.pt"
    checkpoint_path = artifact_root / "segmentation-study" / "classifier.pt"
    vae_config = _write_vae_checkpoint(vae_path)
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=vae_config.latent_channels,
    )
    checkpoint_config = asdict(
        SegmentationTrainingConfig(
            image_size=16,
            vae_checkpoint_path="artifacts/openearthmap-custom-vae.pt",
        )
    )
    checkpoint_config["vae_latent_channels"] = vae_config.latent_channels
    _save_checkpoint(
        checkpoint_path,
        model,
        config=checkpoint_config,
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device="cpu")

    assert checkpoint.vae_encoder is not None
    assert checkpoint.vae_encoder.latent_channels == vae_config.latent_channels


def test_legacy_deeplab_unet_checkpoint_name_still_loads(
    tmp_path: Path,
) -> None:
    vae_path = tmp_path / "vae.pt"
    vae_config = _write_vae_checkpoint(vae_path)
    checkpoint_path = tmp_path / "legacy-name.pt"
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=vae_config.latent_channels,
    )
    checkpoint_config = asdict(
        SegmentationTrainingConfig(
            image_size=16,
            vae_checkpoint_path=str(vae_path),
        )
    )
    checkpoint_config["vae_latent_channels"] = vae_config.latent_channels
    _save_checkpoint(
        checkpoint_path,
        model,
        config=checkpoint_config,
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model"] = "deeplab-unet-v1"
    torch.save(checkpoint, checkpoint_path)

    loaded = _load_segmentation_checkpoint(checkpoint_path, device="cpu")

    assert loaded.vae_encoder is not None
    assert loaded.vae_encoder.latent_channels == vae_config.latent_channels


def test_segmentation_checkpoint_rejects_mismatched_vae_latent_channels(
    tmp_path: Path,
) -> None:
    vae_path = tmp_path / "vae.pt"
    vae_config = _write_vae_checkpoint(vae_path)
    checkpoint_path = tmp_path / "segmenter.pt"
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        vae_latent_channels=vae_config.latent_channels,
    )
    checkpoint_config = asdict(
        SegmentationTrainingConfig(
            image_size=16,
            vae_checkpoint_path=str(vae_path),
        )
    )
    checkpoint_config["vae_latent_channels"] = vae_config.latent_channels + 1
    _save_checkpoint(
        checkpoint_path,
        model,
        config=checkpoint_config,
        metrics={"mean_iou": 0.0, "pixel_accuracy": 0.0, "class_iou": {}},
        step=0,
    )

    with pytest.raises(RuntimeError, match="vae_latent_channels"):
        _load_segmentation_checkpoint(checkpoint_path, device="cpu")


def test_vae_normalization_requires_latent_stats(tmp_path: Path) -> None:
    vae_path = tmp_path / "vae-without-stats.pt"
    _write_vae_checkpoint(vae_path, include_latent_stats=False)

    with pytest.raises(RuntimeError, match="latent_stats"):
        _load_vae_feature_encoder(
            {
                "vae_checkpoint_path": str(vae_path),
                "vae_normalize_latents": True,
            },
            device=torch.device("cpu"),
        )


def test_cli_exposes_segmentation_commands() -> None:
    for command in (
        "segmentation-train",
        "segmentation-evaluate",
        "segmentation-predict",
    ):
        result = CliRunner().invoke(cli, [command, "--help"])

        assert result.exit_code == 0


def test_segmentation_train_is_modal_only_and_evaluate_exposes_targets() -> None:
    runner = CliRunner()

    train_help = runner.invoke(cli, ["segmentation-train", "--help"])
    evaluate_help = runner.invoke(cli, ["segmentation-evaluate", "--help"])

    assert train_help.exit_code == 0
    assert evaluate_help.exit_code == 0
    assert "--target" not in train_help.output
    assert "--target [modal|local]" in evaluate_help.output


def test_segmentation_evaluate_production_preset_applies_calibrated_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[SegmentationEvaluationConfig, str]] = []

    def fake_run(config: SegmentationEvaluationConfig, target: str) -> dict[str, object]:
        captured.append((config, target))
        return {"execution_target": target}

    monkeypatch.setattr(segmentation_cli, "_run_evaluation_with_target", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "segmentation-evaluate",
            "--target",
            "modal",
            "--inference-preset",
            segmentation_cli.PRODUCTION_OPENEARTH_PRESET_NAME,
        ],
    )

    assert result.exit_code == 0
    config, target = captured[0]
    assert target == "modal"
    assert config.checkpoint_path == segmentation_cli.PRODUCTION_OPENEARTH_CHECKPOINT_PATH
    assert config.class_logit_biases == segmentation_cli.PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES
    assert config.confidence_threshold == segmentation_cli.PRODUCTION_OPENEARTH_CONFIDENCE_THRESHOLD
    assert config.tile_stride == segmentation_cli.PRODUCTION_OPENEARTH_TILE_STRIDE
    assert config.tile_scales == segmentation_cli.PRODUCTION_OPENEARTH_TILE_SCALES
    assert config.feather_blending is True


def test_segmentation_predict_production_preset_applies_calibrated_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[SegmentationPredictionConfig, str]] = []

    def fake_run(config: SegmentationPredictionConfig, target: str) -> dict[str, object]:
        captured.append((config, target))
        return {"execution_target": target}

    monkeypatch.setattr(segmentation_cli, "_run_prediction_with_target", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "segmentation-predict",
            "--target",
            "modal",
            "--inference-preset",
            segmentation_cli.PRODUCTION_OPENEARTH_PRESET_NAME,
            "--images-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    config, target = captured[0]
    assert target == "modal"
    assert config.checkpoint_path == segmentation_cli.PRODUCTION_OPENEARTH_CHECKPOINT_PATH
    assert config.class_logit_biases == segmentation_cli.PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES
    assert config.confidence_threshold == segmentation_cli.PRODUCTION_OPENEARTH_CONFIDENCE_THRESHOLD
    assert config.tile_stride == segmentation_cli.PRODUCTION_OPENEARTH_TILE_STRIDE
    assert config.tile_scales == segmentation_cli.PRODUCTION_OPENEARTH_TILE_SCALES
    assert config.feather_blending is True
    assert config.output_dir == "artifacts/segmentation-predictions/production-openearth-v1"


def _write_segmentation_fixture(tmp_path: Path, *, samples: int) -> Path:
    manifest_samples: list[SatelliteSample] = []
    config = ModelConfig(image_size=16, mask_channels=len(OPENEARTHMAP_CLASSES))
    for index in range(samples):
        image_path = tmp_path / "images" / f"tile-{index}.png"
        mask_path = tmp_path / "masks" / f"tile-{index}.png"
        prepared_path = tmp_path / "prepared" / f"tile-{index}.pt"
        _write_rgb_fixture(image_path, index=index)
        _write_label_fixture(mask_path, index=index)
        sample = SatelliteSample(
            image_path=image_path,
            mask_path=mask_path,
            prepared_path=prepared_path,
            split="train",
            source="fixture",
        )
        write_prepared_sample(sample, torch=torch, config=config)
        manifest_samples.append(sample)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_samples, manifest_path)
    return manifest_path


def _write_rgb_fixture(path: Path, *, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), (index * 20, 32, 64))
    for x in range(8, 16):
        for y in range(16):
            image.putpixel((x, y), (222, 31, 7))
    image.save(path)


def _write_label_fixture(path: Path, *, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (16, 16), 5 if index % 2 else 4)
    for x in range(8, 16):
        for y in range(16):
            mask.putpixel((x, y), 8)
    mask.save(path)


def _write_vae_checkpoint(
    path: Path,
    *,
    latent_channels: int = 4,
    include_latent_stats: bool = True,
) -> SatelliteVAEConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    vae_config = SatelliteVAEConfig(
        latent_channels=latent_channels,
        hidden_channels=8,
        downsample_factor=4,
    )
    vae = SatelliteVAE(vae_config)
    payload = {
        "vae_config": asdict(vae_config),
        "model_state": vae.state_dict(),
    }
    if include_latent_stats:
        payload["latent_stats"] = {
            "mean": [0.0] * vae_config.latent_channels,
            "std": [1.0] * vae_config.latent_channels,
        }
    torch.save(payload, path)
    return vae_config
