# VAE Fast-Pool Findings - 2026-06-30

## Context

The fast balanced OpenEarthMap pool is intended to make VAE research loops fast
enough to run repeated seed and sample checks without overfitting to one fixed
selection. The pool manifest is persisted in the Modal data volume:

- `/data/openearthmap/research/openearthmap-vae-fast-balanced-512-v1.jsonl`

The pool has 512 entries, covers all useful classes, spans 75 region groups,
and caps location buckets at two samples per bucket.

## Compared Runs

All runs used the same 512-entry pool, 256 selected samples per trial, six
paired dataset/training seeds, city validation split, 192 px images, factor 4,
32 latent channels, and 48 hidden channels.

| Run | Config | Objective | Steps |
| --- | --- | --- | --- |
| Wide 1k | `vae-research-fast-pool-wide-sweep.modal.json` | Previous reconstruction objective | 1000 |
| Class-aware 1k | `vae-research-fast-pool-sweep.modal.json` | Class-balanced L1, edge loss, class crop sampling | 1000 |
| Class-aware 3k | `vae-research-fast-pool-3k.modal.json` | Same class-aware objective | 3000 |

## Result

The class-aware 3k run is the first fast-pool baseline that looks useful for
the downstream classifier path. It improved every global metric over both 1k
baselines on all six paired seeds.

| Metric | Wide 1k mean | Class-aware 1k mean | Class-aware 3k mean |
| --- | ---: | ---: | ---: |
| SSIM | 0.86652534 | 0.87633989 | 0.93415443 |
| MSE | 0.00227345 | 0.00216358 | 0.00113155 |
| PSNR | 27.73738575 | 27.87374942 | 31.07351112 |
| Edge MAE | 0.04455637 | 0.04311933 | 0.03390769 |
| Mean RGB MAE | 0.00870482 | 0.01031739 | 0.00567766 |

Against class-aware 1k, class-aware 3k won 6/6 paired seeds on SSIM, MSE,
PSNR, edge MAE, and mean RGB MAE.

## Class Error

Class-aware 3k reduced mean class MAE for every useful class compared with both
1k baselines.

| Class | Class-aware 1k MAE | Class-aware 3k MAE | Delta |
| --- | ---: | ---: | ---: |
| bareland | 0.02989804 | 0.02240251 | -0.00749552 |
| rangeland | 0.02889231 | 0.02004340 | -0.00884891 |
| developed space | 0.03321795 | 0.02354561 | -0.00967234 |
| road | 0.02392453 | 0.01706344 | -0.00686109 |
| tree | 0.03418133 | 0.02468432 | -0.00949701 |
| water | 0.02362814 | 0.01725320 | -0.00637493 |
| agriculture land | 0.02376424 | 0.01752498 | -0.00623926 |
| building | 0.04052028 | 0.02823908 | -0.01228120 |

Class-aware 3k also reduced mean class edge MAE for every useful class. The
largest edge gains versus class-aware 1k were building (-0.01333211), tree
(-0.01042277), and developed space (-0.01022644).

## Artifacts

Best class-aware 3k trial:

- Trial index: `1`
- Checkpoint:
  `/artifacts/vae-research/balanced-vae-fast-pool-f4-c32-h48-3k-t01-data20271639-train919177.pt`
- Grid:
  `/artifacts/vae-research/balanced-vae-fast-pool-f4-c32-h48-3k-t01-data20271639-train919177-grid.png`
- Summary:
  `/artifacts/vae-research/balanced-vae-fast-pool-f4-c32-h48-3k-sweep-summary.json`

## Interpretation

The class-aware objective was directionally correct at 1k, but the model was
undertrained. Extending the same loop to 3k steps produced a large, seed-stable
gain and visually better reconstructions. This validates the fast-pool workflow
as a research loop.

This is not yet a final OpenEarthMap-scale VAE. The next step should be a
larger selected-pool or full-dataset follow-up using the same objective, with
continued seed variation and class/location balance checks.

## 2048-Pool Follow-Up

The next scale-up used a larger fixed balanced pool:

- Pool config: `vae-research-balanced-pool-2048.modal.json`
- Sweep config: `vae-research-balanced-pool-2048-3k.modal.json`
- Pool manifest:
  `/data/openearthmap/research/openearthmap-vae-balanced-2048-v1.jsonl`
- Pool report:
  `/artifacts/vae-research/openearthmap-vae-balanced-2048-v1.json`

The 2048-entry pool covered all useful classes, all 75 region groups, and all
297 location buckets. The largest region group had 34 samples and the largest
location bucket had 10 samples.

Each sweep trial selected 512 samples from the 2048-entry pool. Every trial
covered all 75 groups and all 297 location buckets, with at most 7 samples per
group and at most 2 samples per location bucket.

The H100 launch did not schedule a remote function, so the sweep was run on L4:

```bash
SIM_SATELLITE_MODAL_APP_NAME=sim-satellite-vae-pool2048-3k-l4 \
SIM_SATELLITE_MODAL_TRAINING_GPU=L4 \
uv run sim-satellite vae-research-sweep --target modal \
  vae-research-balanced-pool-2048-3k.modal.json
```

### 2048-Pool Metrics

| Metric | Fast 512 class-aware 3k mean | 2048-pool class-aware 3k mean | 2048-pool stdev |
| --- | ---: | ---: | ---: |
| SSIM | 0.93415443 | 0.92656055 | 0.00345857 |
| MSE | 0.00113155 | 0.00124463 | 0.00015969 |
| PSNR | 31.07351112 | 30.71814442 | 0.54051311 |
| Edge MAE | 0.03390769 | 0.03694961 | 0.00226572 |
| Mean RGB MAE | 0.00567766 | 0.00543460 | 0.00159614 |

The larger pool is harder than the 512-pool loop, but the run remained stable:
SSIM ranged from 0.92171597 to 0.93062192 across six seeds. The best and worst
grids both retained coherent roads, roofs, fields, bare ground, and dense
settlement structure.

### 2048-Pool Class Error

| Class | 2048-pool MAE mean | 2048-pool MAE stdev | 2048-pool edge MAE mean |
| --- | ---: | ---: | ---: |
| bareland | 0.01895016 | 0.00439065 | 0.03022265 |
| rangeland | 0.02168292 | 0.00180207 | 0.03516635 |
| developed space | 0.02432272 | 0.00339402 | 0.03971926 |
| road | 0.01761492 | 0.00168390 | 0.02690030 |
| tree | 0.02686892 | 0.00335886 | 0.04573643 |
| water | 0.01417257 | 0.00351677 | 0.02087757 |
| agriculture land | 0.01632885 | 0.00344994 | 0.02421504 |
| building | 0.02802968 | 0.00221212 | 0.04458218 |

Best 2048-pool trial:

- Trial index: `4`
- Checkpoint:
  `/artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-3k-t04-data20274737-train1006705.pt`
- Grid:
  `/artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-3k-t04-data20274737-train1006705-grid.png`
- Summary:
  `/artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-3k-sweep-summary.json`

Worst-SSIM 2048-pool trial:

- Trial index: `5`
- SSIM: `0.92171597`
- Checkpoint:
  `/artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-3k-t05-data20275746-train1015881.pt`
- Grid:
  `/artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-3k-t05-data20275746-train1015881-grid.png`

### 2048-Pool Interpretation

The 2048-pool result is a better research signal than the 512-pool result. The
absolute metrics are slightly lower, but the model remains stable across seeds
while seeing much broader sample diversity. That makes the class-aware 3k
recipe credible enough for the next step: either a 5k run on the 2048 pool to
see whether the residual edge/class softness is undertraining, or a 1024-sample
per-trial sweep from the full OpenEarthMap manifest.
