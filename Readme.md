# Sim Satellite

This project is intended to be an end-to-end exploration of diffusion models and steering.

The output is a simple diffusion model that has been trained on satellite images. The satellite images have been labelled to indicate where a house, tree, road etc is. This allows us to create fully fictious satellite landscapes from the diffusion model.

The user can interact with the model through a simple webapp. They can either generate a new image from noise or draw what they would like to see.

### Implementation

The project is written in Python with uv. It uses Modal by default for training and inference. The web app is a static HTML page that talks to a minimal FastAPI server for masks and generation.

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

Training reads run settings from a JSON file. Omitted fields use
`TrainingConfig` defaults.

For checkpoints intended for an inpainting or infinite-canvas workflow, train
the same model with a small inpaint batch mix:

```json
{
  "image_size": 256,
  "inpaint_probability": 0.2,
  "inpaint_extension_probability": 0.5,
  "inpaint_min_size_ratio": 0.125,
  "inpaint_max_size_ratio": 0.5,
  "validation_holdout_patterns": "abancay",
  "validation_every": 1000
}
```

When `inpaint_probability` is greater than zero, the model gets one extra
conditioning channel where `1` marks pixels to generate and `0` marks known
context. Base batches use the normal full-image flow objective with an all-ones
inpaint channel. Inpaint batches sample interior rectangles, multiple
disconnected rectangles, brush-like strokes, round masks, blobs, and edge
strips. Known context is clamped onto its noised path, and loss is computed
only over the inpaint region. Inpaint batches are selected with a deterministic
schedule that tracks the configured long-run fraction, so `0.2` makes every
fifth batch a inpaint batch. A 256px context window with inpaint ratios from
0.125 to 0.5 trains roughly 32px to 128px inpaint areas without changing the
current model scale.

The OpenEarthMap data command downloads, extracts, and prepares the dataset
locally under `data/openearthmap/`. It writes `manifest.jsonl` and prepared
tensor samples under `data/openearthmap/prepared/`. Modal training downloads,
extracts, and prepares the same dataset on the remote box's local
`/data/openearthmap` scratch disk before training starts, so the full local data
tree does not need to be uploaded into a shared Modal data volume.
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

Data is generated locally under `data/` by default and is not committed.
OpenEarthMap should normally be fully downloaded, extracted, and prepared once
under `data/openearthmap/` for local inspection and debugging. Modal runs prepare
their own copy under `/data/openearthmap` on remote local scratch disk. Prepared
samples track the source dataset, split, image path, mask path, and preparation
metadata.

### OpenEarthMap
It has 5,000 high-resolution aerial and satellite images with manually annotated land-cover masks across 97 regions in 44 countries. Enough for a small end-to-end mask-conditioned diffusion run and for proving the data/model/inference loop.

### Long term goal - Our own labeling pipeline

Broader semi-automated labeling with SAM, SAM 3, or a mask classifier.

### Model Architecture

Custom DDPM is easier to understand and debug, but a PyTorch/diffusers-based implementation is likely faster for useful first results. The important boundary is that conditioning enters through semantic masks, so the architecture can evolve without changing the dataset format.

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
