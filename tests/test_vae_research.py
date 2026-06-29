from __future__ import annotations

from pathlib import Path

import torch

from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.vae.research import (
    VAEResearchManifestConfig,
    build_vae_research_manifest,
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
    assert output_manifest_path.exists()
    assert report_path.exists()
    assert len(read_manifest(output_manifest_path)) == 8
    for class_name in OPENEARTHMAP_CLASSES[1:]:
        assert result.class_hits[class_name] >= 1


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
