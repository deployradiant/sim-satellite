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
