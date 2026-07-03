# Flow-Matching Research Harness

The harness turns "are we getting closer to a competitive satellite
flow-matching model?" into one command per experiment and one command for the
answer. It is built for many parallel subagents working in isolated worktrees:
every scored checkpoint produces exactly one small JSON run record, records
are committed to git, and the progress report is derived from those records.

Reference point: [DiffusionSat](https://arxiv.org/abs/2312.03606)-class
mask-faithful satellite generation. The committed spec
`benchmarks/flow-oem-v1.json` encodes the current competitiveness bar as
explicit metric targets.

## The three commands

```bash
# 1. What must training exclude? (frozen held-out city groups)
uv run sim-satellite benchmark --holdout-only benchmarks/flow-oem-v1.json

# 2. Score a checkpoint (writes benchmarks/runs/<run-id>.json locally)
uv run sim-satellite benchmark --target modal \
  --checkpoint artifacts/my-experiment.pt \
  --experiment my-experiment \
  --notes "what changed and why" \
  benchmarks/flow-oem-v1.json

# 3. Where do we stand? (leaderboard + remaining gap to each target)
uv run sim-satellite benchmark-progress benchmarks/flow-oem-v1.json \
  --markdown artifacts/research/progress.md
```

`benchmark-progress` needs no GPU and no dataset; it only reads
`benchmarks/runs/`.

## What a run measures

- `generation.*`: the existing paired generation metrics (MSE, PSNR, SSIM,
  per-class RGB mean MAE, gradient MAE, boundary gradient delta) computed on
  held-out masks with fixed seeds.
- `mask_adherence.*`: the frozen segmentation checkpoint named in the spec is
  run on the generated images and its prediction is compared to the
  conditioning mask (`generated_mean_iou`). The same score on the real
  held-out images (`real_mean_iou`) is the judge's own ceiling, and
  `iou_ratio = generated / real` normalizes out judge weakness. This is the
  primary metric: it directly measures "the model draws what the mask asked
  for", and it improves through both better models and better labeling.
- `data.manifest_samples_by_source`: sample counts per dataset source at run
  time, so data-scaling experiments (including auto-labeled sources) show up
  next to model experiments in the same leaderboard.

## Protocol rules (what keeps runs comparable)

- Only the eval-defining protocol fields — `manifest_path`, `split`, `seed`,
  `samples`, `image_size`, `segmentation_checkpoint_path` — are hashed into
  the eval hash, and `benchmark-progress` only compares runs with the spec's
  exact eval hash. Never edit those fields in a committed spec — copy the
  file to `flow-oem-v2.json` and start a new leaderboard.
- The sampler fields — `generation_steps`, `guidance_scale`, `precision`,
  `batch_size` — configure the system under test, not the measurement. Runs
  may vary them freely and stay on the same leaderboard: what we care about is
  performance, however the model achieves it. Each run's sampler settings are
  recorded and shown in a leaderboard column (with `wall_seconds`), so a win
  bought with 10x more sampling steps is visible rather than hidden.
- Generation noise is seeded per eval sample, not per batch, so `batch_size`
  is purely a throughput knob and can never change scores.
- The eval set is chosen deterministically: whole city groups are hash-ranked
  by the protocol seed and held out, then `samples` masks are drawn with the
  same seed. Each run records a `sample_digest`; if digests diverge between
  runs (the dataset grew), the progress report warns that deltas are no longer
  a pure model comparison.
- Training must exclude the benchmark holdout. Copy the
  `validation_holdout_patterns` string from `--holdout-only` (or from any run
  record) into the training config field of the same name.
- Checkpoints must match the protocol's `image_size` and the OpenEarthMap
  class count; incompatible checkpoints are rejected instead of producing an
  incomparable number.

## Contract for parallel subagents

Follow the AutoResearch skill (`.agents/skills/autoresearch.md`) for the
overall loop. Harness-specific rules:

- Work in an isolated worktree; never write to the shared checkout.
- Use a unique `--experiment` name per hypothesis and a unique
  `SIM_SATELLITE_MODAL_APP_NAME` per parallel Modal run.
- One hypothesis, one training run, one `benchmark` call, one committed run
  record. Run records are append-only files named by run id, so parallel
  branches merge without conflicts.
- Do not delete or edit other agents' run records. A bad run is documented by
  a better run, not by rewriting history.
- Commit the run record together with the code/config change that produced
  the checkpoint, so `git_commit` in the record points at reproducible state.
- Negative results are results: record them and say so in `--notes`.

## The two improvement axes

Model axis: change architecture, loss, sampler, or training schedule; train
with the benchmark holdout excluded; score the checkpoint. The leaderboard
delta against the current best is the result.

Data axis: grow the training manifest, then retrain the reference config and
score it. For auto-labeled data, follow the segmentation workflow: predict
masks with `sim-satellite segmentation-predict --confidence-threshold 0.85`
(low-confidence pixels become `unlabeled`), give generated labels their own
manifest `source` name, and keep the benchmark eval set untouched — it only
ever contains human-labeled OpenEarthMap groups from the frozen holdout.

## Cached artifacts

Anything deterministic under a protocol is computed once and reused:

- The judge's real-image scores are identical for every run under one eval
  hash (same frozen eval samples, same crops, same judge weights — sampler
  settings do not affect them). The first scored run computes them and the
  CLI writes `benchmarks/cache/real-adherence-<eval-hash>.json`; commit it
  with your run record. Every later run — local or Modal, by any agent, with
  any sampler settings — sends the cache along and only segments its
  *generated* images, which halves judge inference and makes every run's
  `iou_ratio` denominator exactly identical.
- The cache self-invalidates: it is keyed on the eval hash, the eval
  `sample_digest`, and the sha256 of the judge checkpoint file, so a changed
  eval protocol, dataset, or judge silently falls back to recomputing (the
  run record shows `mask_adherence.real_scores_from_cache`).
- Datasets and prepared tensors are already cached in the Modal data volume
  (`_ensure_modal_training_data`); benchmark runs reuse them and never
  re-download.

Do not hand-edit cache files; delete them to force recomputation.

## Judge checkpoint availability

`flow-oem-v1` uses the committed segmentation study checkpoint
`artifacts/segmentation-study-2026-06-28-200k/deeplab_unet_vae200k_fused_noaug_lr3e-4_alldata_s200000_val256.pt`
as the frozen judge. Modal runs read it from the artifact volume; if it is
missing there, upload it (and the custom VAE it references) once:

```bash
uv run modal volume put sim-satellite-artifacts \
  artifacts/segmentation-study-2026-06-28-200k \
  segmentation-study-2026-06-28-200k
uv run modal volume put sim-satellite-artifacts \
  artifacts/openearthmap-custom-vae-f4-c24-h32-train88-val300-200k.pt \
  openearthmap-custom-vae-f4-c24-h32-train88-val300-200k.pt
```

If the judge cannot be loaded the run fails loudly. `--no-mask-adherence`
records a run without the judge, but such runs miss the primary metric and
sort to the bottom of the leaderboard.
