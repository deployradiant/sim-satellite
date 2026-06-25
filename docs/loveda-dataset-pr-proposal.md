# LoveDA Dataset Import PR Proposal

Status: proposal for a separate future PR.

## Intent

Add LoveDA as a proof-of-concept, non-commercial training-data source that can
be remapped into the existing OpenEarthMap-style mask classes. This PR should be
kept separate from LandCover.ai and from SAM fine-tuning so its data delta and
model-smoke result are easy to evaluate.

LoveDA is a good first expansion because its classes map closely to this
project's current 9-channel mask format:

| LoveDA label | Target label |
| --- | --- |
| no-data | unlabeled |
| background | unlabeled |
| building | building |
| road | road |
| water | water |
| barren | bareland |
| forest | tree |
| agriculture | agriculture land |

`rangeland` and `developed space` are not represented directly and should not
be guessed.

## License Gate

LoveDA is suitable for this repository only as non-commercial proof-of-concept
data. Its Zenodo record describes the dataset as CC BY-NC-SA 4.0 and states
that use is academic-only and commercial use is prohibited. The images also need
to respect Google Earth terms.

The PR must:

- keep LoveDA data under `data/loveda/` as generated local state;
- not commit downloaded images, masks, archives, or prepared tensors;
- include citation and license text in importer output or generated metadata;
- preserve the README's proof-of-concept/non-commercial framing.

Sources:

- https://zenodo.org/records/5706578
- https://github.com/Junjue-Wang/LoveDA

## Expected Data Delta

Current local baseline measured on 2026-06-24:

- `data/openearthmap/manifest.jsonl`: 2,687 labeled samples

LoveDA has 5,987 total 1024x1024 RGB images. The PR should count only splits
with labels as supervised training data. Train plus validation are the expected
development splits, so the expected supervised delta is approximately:

- `+4,191` labeled LoveDA samples
- combined labeled manifest: approximately `6,878` samples
- increase over current local baseline: approximately `+156%`

The PR must report actual observed counts from the importer, including skipped
or unlabeled-only samples. If the importer finds that additional labeled splits
are legally and technically available, it must report that separately rather
than folding them silently into the training set.

## Implementation Scope

Expected files:

- `src/sim_satellite/data/loveda.py`
- tests in `tests/test_loveda_data.py`
- a CLI command, for example `uv run sim-satellite loveda`
- optional combined-manifest helper only if it is needed for smoke training

The importer should:

- download or accept local `Train.zip` and `Val.zip`;
- extract under `data/loveda/`;
- remap masks into the existing OpenEarthMap class IDs;
- write prepared tensors and `data/loveda/manifest.jsonl`;
- emit a summary with split counts, skipped samples, and per-class pixel counts.

## Required PR Evidence

Attach the exact command output for:

```bash
uv run sim-satellite loveda --force-prepared
wc -l data/openearthmap/manifest.jsonl data/loveda/manifest.jsonl
uv run pytest -q tests/test_loveda_data.py tests/test_training_data.py
```

Attach one model-consumption smoke run. The minimum acceptable run is a short
CUDA or Modal train that proves the new manifest can feed the model end to end:

```json
{
  "steps": 8,
  "batch_size": 4,
  "image_size": 64,
  "hidden_channels": 16,
  "manifest_path": "data/loveda/manifest.jsonl",
  "checkpoint_path": "artifacts/loveda-smoke.pt",
  "metrics_path": "artifacts/loveda-smoke-metrics.json",
  "max_train_samples": 64,
  "log_every": 1
}
```

Example command:

```bash
uv run sim-satellite train --target modal train.loveda-smoke.json
```

The PR should include the final loss, sample count, class-pixel summary, and the
checkpoint/metrics paths in its description.

## SAM Comparison Note

This PR should not decide whether fine-tuned SAM is better than a closed-set
segmentation teacher. Its role is to add clean remapped supervised labels. A
follow-up experiment can then compare:

- a closed-set segmentation teacher trained on OpenEarthMap plus LoveDA; and
- a SAM/SAM2 fine-tuning pipeline that proposes masks and needs a classifier or
  teacher layer to assign OpenEarthMap classes.
