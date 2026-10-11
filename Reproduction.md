# Reproduction

Setup, experiment observations, and run history for [Sim Satellite](Readme.md).
Run commands from the repository root. Historical results use different data
splits and objectives; compare only runs with matching evaluation protocols.

- [Saved 200k runs](#saved-200k-runs)
- [Run the project](#run-the-project)
- [Training and VAE reconstruction](#training-and-vae-reconstruction)
- [VAE research pools](#vae-research-pools)
- [Flow-matching benchmark harness](#flow-matching-benchmark-harness)
- [Local debugging and Modal web app](#local-debugging-and-modal-web-app)
- [Training log](#training-log)
- [Local xBD and segmentation experiments](#local-xbd-and-segmentation-experiments)
- [Further research notes](#further-research-notes)

## Saved 200k runs

These are the saved 200,000-step reconstruction and segmentation evaluations,
not samples generated from noise. They predate the July xBD experiments below.

### VAE reconstruction

The factor-4, 24-channel VAE trained on 2,387 images and evaluated on 300
city-held-out images at 256 px. The recorded PSNR is 45.63 dB and SSIM is
0.9983. Full settings, seed, history, and latent statistics are in the
[metrics JSON](artifacts/openearthmap-custom-vae-f4-c24-h32-train88-val300-200k-metrics.json).

![Original images, custom VAE reconstructions, and absolute error amplified four times](artifacts/openearthmap-custom-vae-f4-c24-h32-train88-val300-200k-grid.png)

### Segmentation comparison

The saved 256-image evaluation reports mIoU 0.4161 for the lite model and
0.6266 for the VAE-fused model; pixel accuracy is 62.6% and 77.5%, respectively.
See the [lite metrics](artifacts/segmentation-study-2026-06-28-200k/lite_200k_validation256_same_previous.json)
and [VAE-fused metrics](artifacts/segmentation-study-2026-06-28-200k/vae200k_fused_200k_validation256_same_previous.json).
These are historical comparison results, not the later official 500-image gate.

![Image, reference label, lite 200k prediction, and VAE-fused 200k prediction](artifacts/segmentation-study-2026-06-28-200k/head-to-head-labelled-lite200k-vs-vae200k-fused200k.png)

[Additional comparison on unlabelled images](artifacts/segmentation-study-2026-06-28-200k/head-to-head-unlabelled-lite200k-vs-vae200k-fused200k.png).
Figures use OpenEarthMap imagery and labels (Xia et al., WACV 2023);
see [attribution and artifact terms](NOTICE.md).

## Run the project

```bash
uv sync
uv run sim-satellite --help
uv run sim-satellite openearthmap
cat > train.json <<'JSON'
{
  "steps": 8,
  "batch_size": 4,
  "checkpoint_path": "artifacts/tiny-sim-satellite.pt"
}
JSON
uv run sim-satellite train --target modal train.json
uv run sim-satellite infer --target modal
```

## Development checks

```bash
git config core.hooksPath .githooks
uv run ruff format .
uv run ruff check .
```

Commits require `uv run ruff format --check .` through the tracked pre-commit
hook.

## Training and VAE reconstruction

Training reads run settings from a JSON file. Omitted fields use
`TrainingConfig` defaults.
Final mask-conditioned generation metrics can be enabled with
`generation_metrics_samples`, `generation_metrics_steps`, and
`generation_metrics_guidance_scale`. They require either `validation_split` or
`validation_holdout_patterns` so the generated images are compared against
held-out real images. The training result and metrics JSON record MSE, PSNR,
and SSIM.

Native satellite VAE training is available as a separate reconstruction audit
step before latent flow-matching experiments. The 50% OpenEarthMap config trains
a 4x, 24-channel convolutional VAE on half of the prepared training images and
evaluates on images from held-out city groups:

```bash
uv run sim-satellite train-vae train.openearthmap-vae-f4-c24-half.json
```

For longer VAE runs, use the Modal/H100 target. The 20k candidate config trains
on roughly 88% of the prepared training images and validates on 320 images from
held-out city groups:

```bash
uv run sim-satellite train-vae --target modal train.openearthmap-vae-f4-c24-h32-train88-val320-20k.json
```

This writes a custom VAE checkpoint, metrics JSON, and reconstruction grid under
`artifacts/`. The checkpoint and metrics JSON include per-channel train-latent
mean/std values for the encoder `mu` distribution, which is the normalization
contract expected by later latent flow-matching experiments.
Local sweeps on the prepared OpenEarthMap cache used 1,343 train images and up
to 96 held-out validation images. The current config uses 24 latent channels,
32 hidden channels, `kl_weight=1e-6`,
`reconstruction_exponential_weight=0.5`, `reconstruction_exponential_scale=12.0`,
and `learning_rate=1.5e-4`. Treat this as support for the custom VAE direction,
not as a final production codec.
`reconstruction_exponential_weight` and `reconstruction_exponential_scale`
control a nonlinear reconstruction term that makes 99% reconstructions worth
disproportionately more than merely acceptable reconstructions, without adding a
separate hand-tuned sharpness loss.

## VAE research pools

For fast VAE research loops, build a fixed balanced OpenEarthMap pool in the
Modal data volume before launching many seed sweeps:

```bash
uv run sim-satellite vae-research-manifest --target modal vae-research-fast-pool-512.modal.json
uv run sim-satellite vae-research-sweep --target modal vae-research-fast-pool-sweep.modal.json
```

Use `vae-research-fast-pool-wide-sweep.modal.json` as the paired old-objective
baseline and `vae-research-fast-pool-3k.modal.json` as the longer class-aware
fast-pool run. The 2026-06-30 results are recorded in
`docs/vae_fast_pool_findings_2026_06_30.md`.

After validating the 512-entry fast pool, use
`vae-research-balanced-pool-2048.modal.json` and
`vae-research-balanced-pool-2048-3k.modal.json` to run the broader cached
research-pool check. Use `vae-research-balanced-pool-2048-5k.modal.json` for
the current strongest 2048-pool VAE baseline.

To train the downstream VAE-fused segmentation classifier from that baseline,
use `segmentation-vae-pool2048-5k-smoke.modal.json` for a fast Modal wiring
check and `segmentation-vae-pool2048-5k.modal.json` for the longer run. Mask
prediction writes class-id masks, color masks, and confidence masks by default;
use `--confidence-threshold 0.85` on `segmentation-predict` to set low-confidence
pixels to `unlabeled` before using predictions as generated labels.

## Flow-matching benchmark harness

Progress toward a competitive mask-conditioned flow-matching model is measured
with the frozen benchmark spec `benchmarks/flow-oem-v1.json`. The spec pins a
deterministic held-out eval set, fixed generation seeds/steps, a frozen
segmentation judge for mask-adherence mIoU, and explicit competitiveness
targets. Every scored checkpoint writes one committed run record under
`benchmarks/runs/`, and the leaderboard plus remaining target gaps come from:

```bash
uv run sim-satellite benchmark --holdout-only benchmarks/flow-oem-v1.json
uv run sim-satellite benchmark --target modal \
  --checkpoint artifacts/my-experiment.pt \
  --experiment my-experiment benchmarks/flow-oem-v1.json
uv run sim-satellite benchmark-progress benchmarks/flow-oem-v1.json
```

Training runs must exclude the benchmark holdout by copying the
`validation_holdout_patterns` value from `--holdout-only` into the training
config. The spec pins exact `eval_sample_identities`; those identities and the
other eval-defining protocol fields (data path, split, seed, sample count, crop
size, judge) gate comparability through the eval hash and are immutable once
runs exist. `benchmark-progress` refuses to rank a run whose recorded
`sample_digest` does not match the pinned spec digest. Sampler settings (steps,
guidance, precision, batch size) belong to the system under test, are recorded
per run, and never fragment the leaderboard. Vary those sampler settings with
the benchmark CLI overrides (`--generation-steps`, `--guidance-scale`,
`--precision`, `--batch-size`) for exploratory runs. Adopt a new committed
sampler default only after recording a benchmark run that justifies it.
`guidance_scale` is the classifier-free guidance multiplier: `1.0` uses the
plain mask-conditioned prediction, while larger values push samples harder
toward the conditioning mask and can trade image realism for mask adherence.
Deterministic per-eval artifacts are cached and committed under
`benchmarks/cache/` — the judge's real-image scores are computed by the first
run and reused by every later run. See `docs/research-harness.md` for the
parallel-subagent contract and the data-growth workflow.

The fast pool config writes
`/data/openearthmap/research/openearthmap-vae-fast-balanced-512-v1.jsonl` and a
balance report under `/artifacts/vae-research/`. It balances useful land-cover
classes, region groups, and region plus tile-id location buckets. OpenEarthMap
does not provide true latitude/longitude coordinates in the prepared manifest,
so the location buckets are an explicit path-derived proxy rather than real
geospatial coordinates.

## Local debugging and Modal web app

For local CUDA PyTorch debugging runs, put `"steps": 1` in the training JSON:

```bash
uv sync
cat > train.local.json <<'JSON'
{
  "steps": 1,
  "device": "cuda"
}
JSON
uv run sim-satellite train --target local train.local.json
uv run sim-satellite infer --target local --steps 1
```

For the remote Modal web app, authenticate once, make sure the web checkpoint is
available in the Modal artifact volume, and start the hot-reloaded web endpoint:

```bash
uv run modal setup
uv run modal volume create sim-satellite-artifacts
uv run modal volume put --force sim-satellite-artifacts artifacts/tiny-sim-satellite-modal.pt tiny-sim-satellite-modal.pt
uv run modal serve -m sim_satellite.modal_app
```

Modal GPU defaults are intentionally split by workload. Training uses
`SIM_SATELLITE_MODAL_TRAINING_GPU`, defaulting to `H100!`. Batch inference uses
`SIM_SATELLITE_MODAL_INFERENCE_GPU`, defaulting to `T4`. The web endpoint uses
`SIM_SATELLITE_MODAL_WEB_GPU`, defaulting to the inference GPU. Keep
inference-only web endpoints on `T4` unless they fail memory or latency checks;
try `L4` or `A10` before using training-class GPUs.

If `sim-satellite-artifacts` already exists, skip the `modal volume create`
command. Modal prints the temporary `web_remote` URL when `modal serve` starts.
For a persistent URL, deploy the same app instead:

```bash
uv run modal deploy -m sim_satellite.modal_app
```

## Data

Data is generated locally under `data/` by default and is not committed. Model
checkpoints and other generated artifacts are written under `artifacts/` by
default. Most are ignored; selected historical run artifacts are tracked.
OpenEarthMap should normally be fully downloaded, extracted, and prepared once
under `data/openearthmap/` for local inspection and debugging. Modal runs prepare
and reuse their copy under `/data/openearthmap` in the shared data volume. Prepared
samples track the source dataset, split, image path, mask path, and preparation
metadata.

Dataset licenses vary by source and may restrict commercial use. OpenEarthMap
label data follows the license of each source image, with some source regions
under non-commercial Creative Commons terms. LandCover.ai is CC BY-NC-SA 4.0.
LoveDA is academic/non-commercial only. Any future dataset import should
document the source license, preserve attribution metadata, and be treated as
proof-of-concept/non-commercial unless every included source is explicitly
compatible with broader use.

See [NOTICE.md](NOTICE.md) for the code and research-artifact licensing boundary.
The repository includes selected historical checkpoints and evaluation figures;
raw dataset archives and prepared tensor caches remain local or in Modal volumes.

## OpenEarthMap
It has 5,000 high-resolution aerial and satellite images with manually annotated land-cover masks across 97 regions in 44 countries. Enough for a small end-to-end mask-conditioned diffusion run and for proving the data/model/inference loop.

## Long term goal - Our own labeling pipeline

Broader semi-automated labeling with SAM, SAM 3, or a mask classifier.

## Model Architecture

Custom DDPM is easier to understand and debug, but a library-backed implementation is likely faster for useful first results. The important boundary is that conditioning enters through semantic masks, so the architecture can evolve without changing the dataset format.

## References

- [OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping](https://open-earth-map.org/overview.html) (Xia et al., WACV 2023).
- [DiffusionSat: A Generative Foundation Model for Satellite Imagery](https://arxiv.org/abs/2312.03606) (Khanna et al., ICLR 2024)..

## Milestones


### Fast end-to-end run:

- download a small public satellite dataset into a local cache
- normalize image tiles and semantic masks into a simple manifest
- train a small mask-conditioned diffusion model
- run inference from a mask input


## Training log

- 2026-06-13 03:23 PDT: OpenEarthMap cache has 767 paired image/mask samples,
  329 images skipped because the downloaded archive has no matching mask, and 9 mask
  channels including `unlabeled`.
- 2026-06-13 03:23 PDT: Modal GPU run completed with
  `uv run sim-satellite train --target modal train.json`, where `train.json`
  set `steps=8`, `batch_size=4`, and
  `checkpoint_path=artifacts/tiny-sim-satellite.pt`.
  Result: `device=cuda`, `final_loss=0.9386278390884399`, checkpoint written to
  Modal volume path `/artifacts/tiny-sim-satellite.pt`.
- 2026-06-13 03:23 PDT: Pulled the Modal checkpoint locally with
  `uv run modal volume get --force sim-satellite-artifacts tiny-sim-satellite.pt artifacts/tiny-sim-satellite-modal.pt`.
  Verified metadata: `training_device=cuda`, `mask_channels=9`.
- 2026-06-13 03:23 PDT: Verified the web app at `http://127.0.0.1:8765`
  against `artifacts/tiny-sim-satellite-modal.pt`; browser generation produced a
  256px PNG from an OpenEarthMap mask.
- 2026-06-13 17:21 PDT: Replaced uniform flow timestep sampling with
  logit-normal sampling, `t = sigmoid(N(0, 1))`, for both training and validation.
  Research rationale: rectified-flow work such as Stable Diffusion 3 biases
  training toward perceptually useful intermediate noise levels while keeping the
  linear flow path and velocity target. The old uniform sampler was removed
  rather than kept as a compatibility mode.
- 2026-06-14 PDT: AutoResearch pass for faster progress on OpenEarthMap used
  fixed-seed 1000-step H100 runs with `batch_size=10`, `image_size=256`,
  `precision=bf16`, CUDA random tensors, and default sample preloading.
  Baseline current code (`logs/autoresearch-dualattn-current-1000.log`) ended at
  loss `0.032427478581666946`, `705.3s`, `14.18 img/s`, with sampled
  `load_seconds=0.433` at step 1000. Replacing Python tuple/list tensor
  conversion with byte-buffer conversion for RGB images and grayscale class-id
  masks (`logs/autoresearch-byte-tensors-1000.log`) produced the same final loss
  in `298.5s`, `33.50 img/s`, with sampled `load_seconds=0.036` at step 1000.
  Decision: keep the byte-buffer tensor path as a pure speed win. Next best
  hypotheses: remove or move 64x64 deep attention, try channels-last tensors,
  and separately test mask dropout `0.0` for mask-conditioned quality.
- 2026-06-14 PDT: A wider AutoResearch batch tested the next hypotheses with
  isolated worktrees after an initial shared-checkout batch was rejected as
  confounded. Clean 1000-step results: no-deep-attention
  (`logs/autoresearch-no-deep-clean-1000.log`) reached loss `0.031935956329107285`
  in `252.1s`; fixed-width time MLP (`logs/autoresearch-time-mlp-clean-1000.log`)
  reached loss `0.031136244535446167` in `311.3s`; channels-last
  (`logs/autoresearch-channels-last-clean-1000.log`) reached loss
  `0.03235258907079697` in `365.4s`; LR `2e-4`
  (`logs/autoresearch-lr2e-4-clean-1000.log`) reached loss
  `0.030154123902320862` in `313.8s`; mask dropout `0.0`
  (`logs/autoresearch-maskdrop0-clean-1000.log`) reached loss
  `0.04803936555981636`; hidden channels `96`
  (`logs/autoresearch-hidden96-clean-1000.log`) reached loss
  `0.03410714119672775` in `254.2s`. The merged direction removes the extra
  64x64 down-path attention, keeps the byte-buffer tensor path, and updates the
  main 20k config to LR `2e-4`. The combined validation
  (`logs/autoresearch-nodeep-lr2e4-combined-1000.log`) reached loss
  `0.029987284913659096` in `263.0s`, `38.02 img/s`, with `93,109,891`
  parameters. Future parallel Modal experiments should set a unique
  `SIM_SATELLITE_MODAL_APP_NAME` to avoid web endpoint label collisions.
- 2026-06-14 PDT: AutoResearch follow-up focused on H100 wall-clock efficiency
  without intentionally weakening the model. Batch-size probes showed that
  `batch_size=13` reached `39.56 img/s` but worsened loss to
  `0.04314757138490677`, `batch_size=20` reached `42.88 img/s` but worsened loss
  to `0.05172574892640114`, and `batch_size=32/40/59` OOMed on H100 80GB.
  Decision: do not update the production config from the batch-size sweep alone.
  The data-loading boundary was changed instead: manifests now require prepared
  sample data under a `prepared` field, and `sim-satellite openearthmap` writes
  decoded RGB `uint8` plus compact class-id mask `uint8` tensors so training
  crops from prepared data rather than re-decoding PIL images. Local validation:
  `uv run pytest -q` and `uv run ruff check` passed. Local data prep is blocked
  from reproducing the old 767-sample benchmark manifest because this checkout
  has only a partial extracted tree and no raw archive; full archive prep found
  2,687 pairs, which should be treated as a new dataset baseline if used.
  Next best hypotheses: benchmark prepared-data loading on Modal with the same
  767-sample manifest, add `torch.compile`/fused Adam/SDPA experiments, and only
  then retune batch size or LR.
- 2026-06-21 PDT: First AutoResearch follow-up implemented native mask-aware
  generation metrics. Final generation metrics now include per-class RGB mean
  MAE, RGB histogram JS distance, gradient MAE, and boundary gradient deltas
  when semantic masks are available.
- 2026-06-26 PDT: AutoResearch VAE follow-up kept the exponential reconstruction
  loss fixed and tested training mechanics/capacity on 1,343 train and 96
  validation OpenEarthMap samples for 1,500 local steps. Baseline c16/h32 reached
  MSE `0.002373`, PSNR `27.117`, SSIM `0.8529`, edge MAE `0.04639`, and RGB MAE
  `0.00817`. Deterministic mean-latent training and `weight_decay=0.0` were
  rejected because they worsened the fixed-seed validation metrics. Wider
  c24/h48 improved SSIM/edge but introduced color drift. Latent-bandwidth-only
  c24/h32 improved every tracked metric on the same seed: MSE `0.002359`, PSNR
  `27.262`, SSIM `0.8560`, edge MAE `0.04587`, and RGB MAE `0.00738`. Decision:
  update the custom VAE config/default to 24 latent channels while keeping hidden
  width and the loss unchanged. Next best hypotheses: run the c24/h32 config
  longer on H100, then retest c32/h32 and learning-rate schedule only if c24/h32
  plateaus blurry.
- 2026-06-26 PDT: The c24/h32 VAE was then trained for the longer 5,000-step
  Modal/H100 config with
  `SIM_SATELLITE_MODAL_APP_NAME=sim-satellite-vae-c24-long-20260626 uv run sim-satellite train-vae --target modal train.openearthmap-vae-f4-c24-half.json`.
  The run prepared 2,687 OpenEarthMap samples on remote `/data/openearthmap`
  scratch and trained on 1,343 of them. Final 64-sample validation metrics were
  MSE `0.001522`, PSNR `29.512`, SSIM `0.9211`, edge MAE `0.03701`, and RGB MAE
  `0.00932`. The checkpoint, metrics, and grid were written to the Modal
  artifact volume as `/artifacts/openearthmap-custom-vae-f4-c24-half.*`.
  Decision: keep c24/h32 for the custom VAE direction; the next risk to inspect
  is mild color drift, not structural capacity.
- 2026-06-26 PDT: A longer c24/h32 Modal/H100 run trained for 20,000 steps with
  `train_fraction=0.88`, 2,364 train samples, and 320 validation samples:
  `SIM_SATELLITE_MODAL_APP_NAME=sim-satellite-vae-c24-20k-20260626 uv run sim-satellite train-vae --target modal train.openearthmap-vae-f4-c24-h32-train88-val320-20k.json`.
  Final validation metrics improved to MSE `0.000245`, PSNR `37.456`, SSIM
  `0.9864`, edge MAE `0.01709`, and RGB MAE `0.00453`. The checkpoint, metrics,
  and grid were written to `/artifacts/openearthmap-custom-vae-f4-c24-h32-train88-val320-20k.*`.
  Decision: this is the current best validated custom VAE checkpoint. The next
  comparison should be latent flow matching with this encoder versus pixel flow
  matching, not more sharpness-loss tuning.
- 2026-07-03 PDT: Added the flow-matching benchmark harness on branch
  `claude/flow-matching-diffusion-eval-kdfr1r`. `sim-satellite benchmark`
  scores any diffusion checkpoint against the frozen spec
  `benchmarks/flow-oem-v1.json` (deterministic city-group holdout, fixed
  generation seeds, mask-adherence mIoU judged by the frozen 2026-06-28
  segmentation checkpoint) and writes one committed run record per run under
  `benchmarks/runs/`. `sim-satellite benchmark-progress` aggregates records
  into a leaderboard with remaining gaps to explicit competitiveness targets.
  No leaderboard entries exist yet; the next step is scoring the current best
  flow U-Net checkpoint as the baseline run, then dispatching parallel model
  and data experiments against it.
- 2026-07-04 UTC: AutoResearch benchmark follow-up on branch
  `codex/benchmark-improvement-flow` scored the existing
  `/artifacts/openearthmap-from-dataset-mix-landcoverai50-h100-b10-8k.pt`
  checkpoint on `flow-oem-v1`. Baseline 32-step, guidance-1.0 sampling reached
  mask mIoU `0.3716`, iou ratio `0.5968`, SSIM `0.1812`, RGB macro MAE
  `0.0496`, and boundary-gradient delta `0.1115`. A 32-step, guidance-1.5
  sampler cleared the two mask targets with mIoU `0.5532` and iou ratio
  `0.8885`, improved SSIM to `0.2139`, and improved RGB macro MAE to `0.0472`,
  while worsening boundary-gradient delta to `0.2183`. Decision: keep the
  benchmark default at 32 steps and update the default guidance scale to `1.5`;
  next best hypothesis is improving boundary realism/sharpness without giving
  back mask-adherence gains.

## Local xBD and segmentation experiments

The following notes were preserved from the local June–July 2026 research
checkout. Its xBD preparation scripts, SMP model implementations, inference
presets, and run configs are **not included in the published main branch**.
Commands in this section document that working environment and require those
unpublished changes; artifact paths refer to local or Modal run outputs unless
explicitly linked. These observations are not a reproducibility claim for the
published checkout.

The OpenEarthMap data command downloads, extracts, and prepares the dataset
locally under `data/openearthmap/`. It writes `manifest.jsonl` and prepared
tensor samples under `data/openearthmap/prepared/`. The public OpenEarthMap zip
contains labels for the xBD subset plus `xbd_files.csv`, but the xBD RGB images
must be downloaded separately from xBD/Maxar. The reliable source is the
authenticated xView2 download page at <https://xview2.org/dataset>; download the
signed Challenge train, tier3, test, and holdout archives, or the full geotiff
archive parts, and place the archives or extracted image tree under `data/xbd/`.
Mapped xBD pre-disaster images may be stored as PNG or TIFF files; preparation
normalizes source filenames by stem against `xbd_files.csv` and writes compiled
OpenEarthMap RGB PNGs into the cache.
Then compile the mapped images into the OpenEarthMap cache with:

```bash
uv run sim-satellite openearthmap --xbd-dir data/xbd --force-prepared
```

When `--xbd-dir` is provided, preparation fails if any RGB image listed in
`xbd_files.csv` is missing, instead of silently training on a partial xBD merge.
If archives are present, only the mapped OpenEarthMap images are extracted into
`data/xbd/.openearthmap-xbd-cache/`.
Modal training prepares and reuses data in the shared Modal data volume under
`/data/openearthmap`. If the xBD tree or archives are uploaded to `/data/xbd`,
Modal will compile them automatically before training:

```bash
uv run modal volume put --force sim-satellite-openearthmap-cache data/xbd xbd
uv run sim-satellite train --target modal train.json
```

Set `SIM_SATELLITE_OPENEARTHMAP_XBD_DIR` when the xBD tree is mounted somewhere
other than `/data/xbd`.
To explicitly rebuild and commit the shared Modal OpenEarthMap cache after
uploading xBD images or xView2 archives:

```bash
uv run python scripts/stage_openearthmap_xbd.py
uv run python scripts/audit_openearthmap_xbd_kaggle_sources.py
uv run python scripts/download_openearthmap_xbd_kaggle.py --dry-run
uv run modal volume put --force sim-satellite-openearthmap-cache data/xbd xbd
uv run modal run scripts/audit_openearthmap_modal.py
uv run modal run scripts/prepare_openearthmap_xbd_modal.py
uv run modal run scripts/audit_openearthmap_modal.py
```

`scripts/stage_openearthmap_xbd.py` writes a local report under
`artifacts/data-audits/` and refuses to upload with `--upload` unless all mapped
xBD images are present. `scripts/audit_openearthmap_xbd_kaggle_sources.py`
indexes candidate Kaggle mirrors and reports how many currently missing mapped
images they expose by event, suffix, and file prefix before any download is
attempted. `scripts/download_openearthmap_xbd_kaggle.py` can fetch only mapped
xBD pre-disaster RGB images from Kaggle xBD/xView2 mirrors using the pinned
`kaggle==1.6.17` CLI; use `--event` to constrain retries to specific disaster
groups. Use `scripts/stage_openearthmap_xbd.py --cache-archives` after large
archive downloads to extract only mapped images into
`data/xbd/.openearthmap-xbd-cache/`; top-level source archives can then be
deleted as generated state after `archive_available_images=0` and
`missing_images=0` are confirmed. Kaggle mirrors are best-effort seed sources,
not the primary data path. As of 2026-07-01, the locally staged Kaggle mirror
mix in this checkout covers all 1,162 OpenEarthMap-mapped xBD images.
Kaggle mirror license metadata varies, and the default broad xBD mirror reports
an unknown license; keep the upstream xBD/Maxar usage terms in mind before using
the resulting model outside local research. The final Modal audit should report
`xbd_missing_images=0` and manifest split counts matching the official train/val
coverage before starting the next segmentation training run.

The post-xBD segmentation config enforces that coverage through
`required_split_counts`, so it fails fast if the shared cache still contains
only the no-xBD manifest:

```bash
uv run sim-satellite segmentation-train train.openearthmap-segmentation-smp-unet-efficientnet-b4-scse-official-xbd-full-polish-v1.json
```

The current full-coverage 9-class OpenEarthMap validation gate is below
production quality, but calibrated operating points are exposed as inference
presets. They use the current xBD checkpoint, validated class-logit biases, and
multiscale feathered tiles. The current dense 9-class full-coverage gate is
0.807 labeled-pixel accuracy and 0.668 mIoU on the official 500-image
validation split. The Modal prepared-mask audit in
`artifacts/data-audits/openearthmap-modal-prepared-mask-audit-20260702.json`
loaded all 3,500 prepared tensors with no missing files and no invalid labels,
so the remaining gap is model/taxonomy ambiguity rather than a corrupt prepared
tensor cache.
The consolidated operating-point gate in
`artifacts/modal-metrics/openearth-operating-points-20260702.json` marks the
production mask presets as passing and the dense 9-class full-coverage gate as
failing.

Prediction presets default to `artifacts/segmentation-predictions/<preset>`
when `--output-dir` is omitted. Each prediction manifest includes the class-id
mask, optional RGB mask, and a grayscale `confidence_map` PNG by default; pass
`--no-confidence-maps` only when the per-pixel confidence artifact is not needed.

For a full-coverage mask that clears 90% labeled-pixel accuracy on the full
500-image official OpenEarthMap validation gate, use `coarse-openearth-v1`; it
remaps fine classes to natural, built, water, and unlabeled by aggregating
fine-class logits before the coarse argmax. This coarse mask selected all pixels
with 93.7% labeled-pixel accuracy and 0.851 mIoU, recorded in
`artifacts/modal-metrics/xbd-coarse-openearth-v1-official-val500.json`:

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset coarse-openearth-v1 \
  --images-dir path/to/unseen/images \
  --output-dir artifacts/segmentation-predictions/coarse-openearth-v1
```

For a richer full-coverage land-cover mask, use `landcover-openearth-v1`; it
keeps vegetation, bare, built, water, and unlabeled classes, with roads merged
into built. On the same official 500-image validation gate, direct remapped
logit inference reached 93.3% labeled-pixel accuracy and 0.747 mIoU, recorded
in
`artifacts/modal-metrics/xbd-landcover-builtroad-openearth-v1-official-val500.json`:

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset landcover-openearth-v1 \
  --images-dir path/to/unseen/images \
  --output-dir artifacts/segmentation-predictions/landcover-openearth-v1
```

Use `landcover-road-openearth-v1` when a separate road class is worth the lower
full-coverage mIoU; it reached 91.1% labeled-pixel accuracy and 0.716 mIoU in
`artifacts/modal-metrics/xbd-landcover-openearth-v1-official-val500.json`.

For the highest-coverage fine-class preset that clears 90% selected
labeled-pixel accuracy on the same validation gate, use
`coverage-openearth-v1`; it abstains on uncertain pixels by writing them as
unlabeled, using a global 0.74 threshold. It selected 68.7% of labeled pixels
with 91.0% selected labeled-pixel accuracy and 0.806 selected mIoU, recorded in
`artifacts/modal-metrics/coverage-openearth-v1-official-val500.json`:

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset coverage-openearth-v1 \
  --images-dir path/to/unseen/images \
  --output-dir artifacts/segmentation-predictions/coverage-openearth-v1
```

A top1/top2 probability-margin sweep was also run on the same checkpoint and
validation split. It did not improve the reliable frontier over max-confidence
thresholding; the comparison is recorded in
`artifacts/modal-metrics/xbd-full-official-val500-scales075-1-125-bias-confidence-margin-sweep.json`.
An official-val class-specific confidence-threshold coordinate search also did
not materially improve the reliable frontier. It started from the global 0.74
threshold operating point, then reached 69.0% selected labeled-pixel coverage
with 90.8% selected labeled-pixel accuracy and 0.800 selected mIoU by loosening
only `unlabeled` and `bareland`; the stronger promoted preset remains the
global 0.74 threshold at 68.7% coverage, 91.0% accuracy, and 0.806 mIoU. The
comparison is recorded in
`artifacts/modal-metrics/xbd-full-official-val500-class-threshold-coordinate-search-v1.json`.
An official-val coordinate search for class-logit biases with dense labeled
accuracy as the objective also failed to move the full-coverage gate materially:
the best candidate reached 0.8076 labeled-pixel accuracy and 0.668 mIoU in
`artifacts/modal-metrics/xbd-full-official-val500-scales075-1-125-lacc-bias-calibrated-stride4-eval.json`.
A conservative official-xBD generalization fine-tune with label smoothing, EMA,
domain augmentation, and a coarse auxiliary loss also failed to improve the
dense fine-class gate. Its best full official-val checkpoint reached 0.795
labeled-pixel accuracy and 0.650 mIoU, while the coarse remapped metric remained
high at 0.934 labeled-pixel accuracy and 0.844 mIoU. The training metrics are in
`artifacts/segmentation-openearthmap-smp-unet-efficientnet-b4-scse-official-xbd-generalize-v1/smp-unet-efficientnet-b4-scse-official-xbd-generalize-s900.json`.
A direct five-class land-cover head trained from the dense xBD checkpoint was
also checked on the full official validation split and stopped after the first
validation pass because it was worse than the remapped dense-logit production
preset: step 250 reached 0.890 labeled-pixel accuracy and 0.415 mIoU. The
checkpoint with embedded metrics is
`artifacts/segmentation-openearthmap-smp-unet-efficientnet-b4-scse-official-xbd-landcover-direct-fullval-v1/smp-unet-efficientnet-b4-scse-official-xbd-landcover-direct-fullval-s1000.pt`.
The Modal confidence and bias-calibration helpers now default to the current
xBD checkpoint on the official `val` split with the 500-image gate, multiscale
tiles, and feather blending. Use the `SIM_SATELLITE_*_TRAINING_VALIDATION_SPLIT`
environment flags only for exploratory holdout checks, not for promotion.

For the default production fine-class mask on unseen images, use
`production-openearth-v1`. It is an alias for the validated
`reliable-openearth-v1` operating point; on the same full validation gate it
selected 62.0% of labeled pixels with 92.6% selected labeled-pixel accuracy and
0.833 selected mIoU, recorded in
`artifacts/modal-metrics/reliable-openearth-v1-official-val500.json`:

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset production-openearth-v1 \
  --images-dir path/to/unseen/images \
  --output-dir artifacts/segmentation-predictions/production-openearth-v1
```

If the unseen images are already staged in the shared Modal data volume, run the
same preset remotely and write masks to the artifact volume:

```bash
uv run sim-satellite segmentation-predict \
  --target modal \
  --inference-preset production-openearth-v1 \
  --images-dir data/unseen/images \
  --output-dir artifacts/segmentation-predictions/production-openearth-v1
```

For stricter masks with higher selected-pixel mIoU and lower coverage, use
`strict-openearth-v1`. On the official 500-image OpenEarthMap validation gate,
this preset selected 34.8% of labeled pixels with 97.1% selected
labeled-pixel accuracy and 0.917 selected mIoU; unselected pixels are written as
unlabeled.

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset strict-openearth-v1 \
  --images-dir path/to/unseen/images \
  --output-dir artifacts/segmentation-predictions/strict-openearth-v1
```

To recheck a preset on the official OpenEarthMap validation split:

```bash
uv run sim-satellite segmentation-evaluate \
  --target modal \
  --inference-preset coarse-openearth-v1 \
  --manifest data/openearthmap/manifest.jsonl \
  --split val \
  --manifest-split \
  --max-samples 500
```

The command can also be called directly as a package script:

```bash
uv run python -m sim_satellite.data.openearthmap
```

To inspect the current cache without downloading, extracting, or rewriting
prepared tensors:

```bash
uv run sim-satellite openearthmap --audit
uv run sim-satellite openearthmap --audit --xbd-dir data/xbd
uv run modal run scripts/audit_openearthmap_modal.py
```

The audit reports manifest split counts, official split coverage, whether
`xbd_files.csv` is present, and how many mapped xBD RGB images are available
from the configured xBD directory or archives.

Prepared tensors can be reused across training runs as long as the preparation
code, OpenEarthMap class mapping, and tensor format are unchanged. After
changing preparation behavior, rebuild them with:

```bash
uv run sim-satellite openearthmap --force-prepared
```


## Further research notes

- [Frozen flow-matching benchmark](docs/research-harness.md)
- [VAE fast-pool findings](docs/vae_fast_pool_findings_2026_06_30.md)
- [Scaling-law findings](docs/scaling-law-findings-2026-06-22.md)
- [Black Forest latent VAE learnings](docs/black_forest_latent_vae_learnings.md)
- [LandCover.ai dataset proposal](docs/landcover-ai-dataset-pr-proposal.md)
- [LoveDA dataset proposal](docs/loveda-dataset-pr-proposal.md)
