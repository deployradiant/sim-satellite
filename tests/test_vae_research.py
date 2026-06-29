from __future__ import annotations

from pathlib import Path

import torch
from click.testing import CliRunner

from sim_satellite.cli import cli
from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.vae.research import (
    VAEResearchManifestConfig,
    _sample_group,
    build_vae_research_manifest,
    load_vae_research_manifest_config_values,
    load_vae_research_sweep_config_values,
)


def test_build_vae_research_manifest_balances_classes_and_groups(tmp_path: Path) -> None:
    base_manifest_path = _write_research_fixture_manifest(tmp_path)
    output_manifest_path = tmp_path / "research" / "balanced.jsonl"
    report_path = tmp_path / "artifacts" / "balanced.json"

    result = build_vae_research_manifest(
        VAEResearchManifestConfig(
            base_manifest_path=str(base_manifest_path),
            output_manifest_path=str(output_manifest_path),
            report_path=str(report_path),
            samples=8,
            seed=123,
            min_class_pixels=16,
            max_group_fraction=0.25,
        )
    )

    assert result.selected_samples == 8
    assert result.selected_groups == 4
    assert result.missing_classes == []
    assert max(result.group_counts.values()) <= 2
    assert max(result.location_bucket_counts.values()) <= 2
    assert output_manifest_path.exists()
    assert report_path.exists()
    assert len(read_manifest(output_manifest_path)) == 8
    for class_name in OPENEARTHMAP_CLASSES[1:]:
        assert result.class_hits[class_name] >= 1


def test_vae_research_sweep_config_accepts_class_and_edge_loss_weights() -> None:
    config = load_vae_research_sweep_config_values(
        {
            "samples": 128,
            "validation_samples": 16,
            "max_location_bucket_fraction": 0.25,
            "edge_weight": 0.2,
            "class_balanced_l1_weight": 0.4,
            "class_balanced_edge_weight": 0.1,
            "class_balanced_crop_probability": 0.5,
        }
    )

    assert config.edge_weight == 0.2
    assert config.max_location_bucket_fraction == 0.25
    assert config.class_balanced_l1_weight == 0.4
    assert config.class_balanced_edge_weight == 0.1
    assert config.class_balanced_crop_probability == 0.5


def test_vae_research_manifest_config_accepts_class_index_and_location_balance() -> None:
    config = load_vae_research_manifest_config_values(
        {
            "class_index_path": "data/openearthmap/research/openearthmap-class-index-v1.json",
            "samples": 128,
            "max_group_fraction": 0.1,
            "max_location_bucket_fraction": 0.05,
        }
    )

    assert config.class_index_path.endswith("openearthmap-class-index-v1.json")
    assert config.samples == 128
    assert config.max_group_fraction == 0.1
    assert config.max_location_bucket_fraction == 0.05


def test_vae_research_group_keeps_multiword_openearthmap_region() -> None:
    root = Path("/data/openearthmap")
    sample = SatelliteSample(
        image_path=root / "extracted/OpenEarthMap_wo_xBD/al_qurnah/images/al_qurnah_1.tif",
        mask_path=root / "extracted/OpenEarthMap_wo_xBD/al_qurnah/labels/al_qurnah_1.tif",
        prepared_path=root / "prepared/OpenEarthMap_wo_xBD/al_qurnah/images/al_qurnah_1.pt",
        split="train",
        source="openearthmap",
    )

    assert _sample_group(sample) == "openearthmap:al_qurnah"


def test_cli_exposes_vae_research_manifest_command() -> None:
    result = CliRunner().invoke(cli, ["vae-research-manifest", "--help"])

    assert result.exit_code == 0
    assert "--target [local|modal]" in result.output


def _write_research_fixture_manifest(tmp_path: Path) -> Path:
    samples: list[SatelliteSample] = []
    class_ids = tuple(range(1, len(OPENEARTHMAP_CLASSES)))
    for index, class_id in enumerate(class_ids):
        prepared_path = tmp_path / "prepared" / f"sample-{index}.pt"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        image = torch.full((8, 8, 3), 64 + class_id, dtype=torch.uint8)
        mask = torch.full((8, 8), class_id, dtype=torch.uint8)
        torch.save(
            {
                "version": PREPARED_DATA_VERSION,
                "image": image,
                "mask": mask,
                "size": (8, 8),
                "coordinates": None,
            },
            prepared_path,
        )
        samples.append(
            SatelliteSample(
                image_path=tmp_path / "images" / f"region-{index % 4}" / f"sample-{index}.png",
                mask_path=tmp_path / "masks" / f"region-{index % 4}" / f"sample-{index}.png",
                prepared_path=prepared_path,
                split="train",
                source="fixture",
            )
        )

    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(samples, manifest_path)
    return manifest_path
