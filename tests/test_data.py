from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import build_samples, prepare_manifest
from sim_satellite.data import (
    SatelliteDataCache,
    SatelliteDataLoader,
    SatelliteDataset,
    SatelliteSample,
    cache_root,
    read_manifest,
    write_manifest,
)


def test_cache_defaults_to_data_and_can_be_overridden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SIM_SATELLITE_DATA_DIR", raising=False)
    assert cache_root() == Path("data")

    cache = SatelliteDataCache(tmp_path / "cache").ensure()

    assert cache.root == tmp_path / "cache"
    assert cache.raw_dir.is_dir()
    assert cache.prepared_dir.is_dir()
    assert cache.manifests_dir.is_dir()


def test_manifest_round_trip_normalizes_sample_paths(tmp_path: Path) -> None:
    cache = SatelliteDataCache(tmp_path / "cache").ensure()
    image = cache.prepared_dir / "source-a" / "tile.png"
    mask = cache.prepared_dir / "source-a" / "tile-mask.png"
    image.parent.mkdir(parents=True)
    image.touch()
    mask.touch()

    sample = SatelliteSample(
        image_path=image,
        mask_path=mask,
        source="source-a",
        split="train",
        classes=("house", "road"),
        metadata={"tile_id": "001"},
    )

    manifest_path = write_manifest([sample], cache.manifest_path(), root=cache.root)
    raw_record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    loaded = read_manifest(manifest_path, root=cache.root)

    assert raw_record["image_path"] == "prepared/source-a/tile.png"
    assert raw_record["mask_path"] == "prepared/source-a/tile-mask.png"
    assert loaded == [sample]


def test_dataset_filters_and_orders_deterministically(tmp_path: Path) -> None:
    cache = SatelliteDataCache(tmp_path / "cache").ensure()
    samples = [
        SatelliteSample(cache.prepared_dir / "b.png", source="z", split="val"),
        SatelliteSample(cache.prepared_dir / "a.png", source="z", split="train"),
        SatelliteSample(cache.prepared_dir / "c.png", source="a", split="train"),
    ]
    write_manifest(samples, cache.manifest_path(), root=cache.root)

    dataset = SatelliteDataset(cache_dir=cache.root, split="train")

    assert [sample.image_path.name for sample in dataset] == ["c.png", "a.png"]


def test_dataset_seeded_shuffle_is_repeatable(tmp_path: Path) -> None:
    cache = SatelliteDataCache(tmp_path / "cache").ensure()
    samples = [
        SatelliteSample(cache.prepared_dir / f"{index}.png", source="source", split="train")
        for index in range(8)
    ]
    write_manifest(samples, cache.manifest_path(), root=cache.root)

    first = SatelliteDataset(cache_dir=cache.root, shuffle=True, seed=7)
    second = SatelliteDataset(cache_dir=cache.root, shuffle=True, seed=7)
    different_seed = SatelliteDataset(cache_dir=cache.root, shuffle=True, seed=8)

    assert [sample.image_path for sample in first] == [sample.image_path for sample in second]
    assert [sample.image_path for sample in first] != [sample.image_path for sample in different_seed]


def test_dataloader_batches_without_dropping_tail() -> None:
    samples = [
        SatelliteSample(Path(f"{index}.png"), source="source", split="train")
        for index in range(5)
    ]
    loader = SatelliteDataLoader(samples, batch_size=2)

    assert [[sample.image_path.name for sample in batch] for batch in loader] == [
        ["0.png", "1.png"],
        ["2.png", "3.png"],
        ["4.png"],
    ]
    assert len(loader) == 3


def test_prepare_manifest_discovers_images_masks_and_classes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    cache = tmp_path / "cache"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "b.png").touch()
    (image_dir / "a.jpg").touch()
    (image_dir / "notes.txt").touch()
    (mask_dir / "a.png").touch()

    samples = build_samples(
        image_dir,
        source="fixture",
        split="train",
        mask_dir=mask_dir,
        classes=("building",),
    )
    manifest_path = prepare_manifest(
        image_dir,
        cache_dir=cache,
        source="fixture",
        split="train",
        mask_dir=mask_dir,
        classes=("building",),
    )
    loaded = read_manifest(manifest_path, root=cache)

    assert [sample.image_path.name for sample in samples] == ["a.jpg", "b.png"]
    assert samples[0].mask_path == mask_dir / "a.png"
    assert samples[1].mask_path is None
    assert [sample.image_path.name for sample in loaded] == ["a.jpg", "b.png"]
    assert loaded[0].classes == ("building",)
