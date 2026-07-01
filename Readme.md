# Sim Satellite

This project is intended to be an end-to-end exploration of diffusion models and steering.
It is a proof-of-concept research project, not a commercial product or commercial
dataset pipeline.
The repository contains MIT-licensed code only; it does not redistribute
third-party datasets, prepared tensors, trained checkpoints, or model weights.

The output is a simple diffusion model that has been trained on satellite images. The satellite images have been labelled to indicate where a house, tree, road etc is. This allows us to create fully fictious satellite landscapes from the diffusion model.

The user can interact with the model through a simple webapp. They can either generate a new image from noise or draw what they would like to see.

### Implementation

The project is written in Python with uv. It uses Modal by default for training and inference. The web app is a static HTML page that talks to a minimal FastAPI server for masks and generation.

### License

The source code in this repository is licensed under the MIT License. See
[LICENSE](LICENSE).

Third-party satellite datasets, generated dataset derivatives, prepared tensor
data, trained checkpoints, model weights, and evaluation artifacts are not
covered by the MIT code license. This repository does not redistribute those
materials.

Experiments that use non-commercial/share-alike datasets such as OpenEarthMap,
LandCover.ai, or LoveDA should be treated as non-commercial research work. If
dataset derivatives, manifests, checkpoints, or model artifacts are published
separately, release them under CC BY-NC-SA 4.0 with dataset attribution and a
clear non-commercial-use restriction. See [NOTICE.md](NOTICE.md).

### Run the project

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

### Development checks

This repository accepts small documentation-only commits for local Git workflow
verification.

```bash
git config core.hooksPath .githooks
uv run ruff format .
uv run ruff check .
```

Commits require `uv run ruff format --check .` through the tracked pre-commit
hook.

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

The OpenEarthMap data command downloads, extracts, and prepares the dataset
locally under `data/openearthmap/`. It writes `manifest.jsonl` and prepared
tensor samples under `data/openearthmap/prepared/`. Modal training uses the
persistent Modal data volume mounted at `/data`; OpenEarthMap lives under
`/data/openearthmap` and is reused between runs instead of being redownloaded
into throwaway scratch space.
The command can also be called directly as a package script:

```bash
uv run python -m sim_satellite.data.openearthmap
```

Prepared tensors can be reused across training runs as long as the preparation
code, OpenEarthMap class mapping, and tensor format are unchanged. After
changing preparation behavior, rebuild them with:

```bash
uv run sim-satellite openearthmap --force-prepared
```

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

The fast pool config writes
`/data/openearthmap/research/openearthmap-vae-fast-balanced-512-v1.jsonl` and a
balance report under `/artifacts/vae-research/`. It balances useful land-cover
classes, region groups, and region plus tile-id location buckets. OpenEarthMap
does not provide true latitude/longitude coordinates in the prepared manifest,
so the location buckets are an explicit path-derived proxy rather than real
geospatial coordinates.

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

### Data

Data is generated locally under `data/` by default and is not committed. Model
checkpoints and other generated artifacts are written under `artifacts/` by
default and are also not committed.
OpenEarthMap should normally be fully downloaded, extracted, and prepared once
under `data/openearthmap/` for local inspection and debugging. Modal runs prepare
their own copy under `/data/openearthmap` on remote local scratch disk. Prepared
samples track the source dataset, split, image path, mask path, and preparation
metadata.

Dataset licenses vary by source and may restrict commercial use. OpenEarthMap
label data follows the license of each source image, with some source regions
under non-commercial Creative Commons terms. LandCover.ai is CC BY-NC-SA 4.0.
LoveDA is academic/non-commercial only. Any future dataset import should
document the source license, preserve attribution metadata, and be treated as
proof-of-concept/non-commercial unless every included source is explicitly
compatible with broader use.

The intended public release shape for dataset-backed work is a reproducible
research repository plus paper or technical write-up. Do not commit or
redistribute raw datasets, extracted tiles, prepared tensor data, trained
checkpoints, or model weights in this repository.

Before any public release, create a clean release copy of the repository rather
than publishing this working checkout directly. The release copy should include
only code, configs, synthetic test fixtures, documentation, and reproduction
instructions. Exclude local data caches, prepared tensors, logs, checkpoints,
model weights, generated images, and any other dataset-derived artifacts.

### OpenEarthMap
It has 5,000 high-resolution aerial and satellite images with manually annotated land-cover masks across 97 regions in 44 countries. Enough for a small end-to-end mask-conditioned diffusion run and for proving the data/model/inference loop.

### Long term goal - Our own labeling pipeline

Broader semi-automated labeling with SAM, SAM 3, or a mask classifier.

### Model Architecture

Custom DDPM is easier to understand and debug, but a library-backed implementation is likely faster for useful first results. The important boundary is that conditioning enters through semantic masks, so the architecture can evolve without changing the dataset format.

### References

- [OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping](https://open-earth-map.org/overview.html) (Xia et al., WACV 2023).
- [DiffusionSat: A Generative Foundation Model for Satellite Imagery](https://arxiv.org/abs/2312.03606) (Khanna et al., ICLR 2024)..

### Milestones


#### Fast end-to-end run:

- download a small public satellite dataset into a local cache
- normalize image tiles and semantic masks into a simple manifest
- train a small mask-conditioned diffusion model
- run inference from a mask input


### Training log

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
