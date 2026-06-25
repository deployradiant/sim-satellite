# LandCover.ai Dataset Import PR Proposal

Status: proposal for a separate future PR.

## Intent

Add LandCover.ai as a proof-of-concept, non-commercial class-specific
augmentation source. This PR should be separate from LoveDA because the labels
cover fewer classes and should be evaluated differently.

LandCover.ai provides high-quality labels for objects that are important to the
current mask-conditioned generator:

| LandCover.ai label | Target label |
| --- | --- |
| background | unlabeled |
| building | building |
| woodland | tree |
| water | water |
| road | road |

The background class should not be mapped to `bareland`, `rangeland`,
`developed space`, or `agriculture land`. It is mixed unknown land cover and
should stay `unlabeled` unless a later teacher model assigns a class with high
confidence.

## License Gate

LandCover.ai is licensed under CC BY-NC-SA 4.0. It is suitable for this
repository only as proof-of-concept/non-commercial data.

The PR must:

- keep LandCover.ai data under `data/landcoverai/` as generated local state;
- not commit downloaded images, masks, archives, tiles, or prepared tensors;
- include citation and license text in importer output or generated metadata;
- preserve the README's proof-of-concept/non-commercial framing.

Source:

- https://landcover.ai.linuxpolska.com/

## Expected Data Delta

Current local baseline measured on 2026-06-24:

- `data/openearthmap/manifest.jsonl`: 2,687 labeled samples

LandCover.ai has 41 large RGB orthophotos. The dataset's standard prepared
layout, also reflected by TorchGeo, splits those scenes into 10,674 512x512
tiles.

Expected delta if the PR follows the 512x512 tile layout:

- `+10,674` partially labeled samples
- combined labeled manifest: approximately `13,361` samples
- increase over current local baseline: approximately `+397%`

The PR must report actual observed counts from the importer, including skipped
tiles and class-empty tiles. Because this is a partial-label dataset, the PR
description should report both total tiles and tiles containing at least one
supervised class pixel.

## Implementation Scope

Expected files:

- `src/sim_satellite/data/landcoverai.py`
- tests in `tests/test_landcoverai_data.py`
- a CLI command, for example `uv run sim-satellite landcoverai`
- optional combined-manifest helper only if it is needed for smoke training

The importer should:

- download or accept a local LandCover.ai v1 archive;
- extract under `data/landcoverai/`;
- tile large orthophotos deterministically, preferably using the dataset's
  512x512 split convention;
- remap masks into the existing OpenEarthMap class IDs;
- write prepared tensors and `data/landcoverai/manifest.jsonl`;
- emit a summary with split counts, skipped tiles, supervised-class tile counts,
  and per-class pixel counts.

## Required PR Evidence

Attach the exact command output for:

```bash
uv run sim-satellite landcoverai --force-prepared
wc -l data/openearthmap/manifest.jsonl data/landcoverai/manifest.jsonl
uv run pytest -q tests/test_landcoverai_data.py tests/test_training_data.py
```

Attach one model-consumption smoke run. The minimum acceptable run is a short
CUDA or Modal train that proves the new manifest can feed the model end to end:

```json
{
  "steps": 8,
  "batch_size": 4,
  "image_size": 64,
  "hidden_channels": 16,
  "manifest_path": "data/landcoverai/manifest.jsonl",
  "checkpoint_path": "artifacts/landcoverai-smoke.pt",
  "metrics_path": "artifacts/landcoverai-smoke-metrics.json",
  "max_train_samples": 64,
  "log_every": 1
}
```

Example command:

```bash
uv run sim-satellite train --target modal train.landcoverai-smoke.json
```

The PR should include the final loss, total tile count, supervised-class tile
count, class-pixel summary, and the checkpoint/metrics paths in its description.

## SAM Comparison Note

LandCover.ai is especially useful for evaluating a SAM-assisted labeling path:
SAM should help with crisp object boundaries, while the importer gives known
labels for `building`, `tree`, `water`, and `road`. This still does not replace
a closed-set segmentation teacher, because SAM masks need a reliable classifier
before they can become OpenEarthMap-style semantic masks.
