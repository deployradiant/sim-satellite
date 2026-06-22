# Scaling Law Findings - 2026-06-22

## Question

We tested whether OpenEarthMap quality is currently limited more by model width
or by the small dataset size. The working hypothesis was that the model might be
too small, or that fewer than 5,000 available samples could already be the main
constraint.

## Setup

- Dataset: Modal prepared 2,687 paired OpenEarthMap samples and skipped 1,151
  images with missing masks.
- Validation: Zanzibar holdout, leaving roughly 2,652 training samples and about
  35 validation samples.
- Training recipe: 256px tiles, `batch_size=10`, `learning_rate=0.0002`,
  `precision=bf16`, coordinates enabled, and `mask_drop_probability=0.1`.
- Generation metrics: 16 held-out samples, 16 generation steps,
  `guidance_scale=1.0`.

## Short Sweep

The 2k-step sweep compared model width and small data caps.

| run | params | final train loss | final val loss | gen MSE | PSNR | SSIM | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h64 full data | 25.36M | 0.15514 | 0.13059 | 0.03531 | 14.62 | 0.1623 | 88.64 img/s |
| h128 full data | 100.69M | 0.14928 | 0.12776 | 0.03782 | 14.32 | 0.1542 | 44.44 img/s |
| h160 full data | 157.09M | 0.15074 | 0.12919 | 0.04028 | 14.04 | 0.1496 | 32.54 img/s |
| h128, 128 samples | 100.69M | 0.16107 | 0.12434 | 0.04275 | 13.74 | 0.1600 | 44.95 img/s |
| h128, 384 samples | 100.69M | 0.16841 | 0.12776 | 0.02889 | 15.49 | 0.1807 | 44.66 img/s |

Takeaway: the wider models did not produce a clean validation or generation
quality win at 2k steps. h160 was slower and worse than h64/h128 on the measured
generation metrics. The data-cap runs were noisy because the validation set is
small, but they did not show a simple "more width fixes it" story either.

## 20k Validation

The focused 20k-step run compared h64 and h128 with the same full-data recipe.

| run | params | final train loss | final val loss | gen MSE | PSNR | SSIM | elapsed | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h64 full data | 25.36M | 0.07519 | 0.11319 | 0.03407 | 14.86 | 0.1583 | 36.63 min | 91.00 img/s |
| h128 full data | 100.69M | 0.07527 | 0.11301 | 0.03350 | 14.93 | 0.1753 | 74.87 min | 44.52 img/s |

The h128 run was only marginally better:

- validation loss improved by 0.00018, about 0.16 percent relative to h64
- generation MSE improved by about 1.7 percent
- PSNR improved by 0.07 dB
- SSIM improved by 0.017
- cost increased to about 4x parameters and 2x wall time

Both validation curves flattened around 12k to 15k steps. Extending training
inside the current architecture produced only small oscillating gains after that.

## Conclusion

The model being too small is not the primary blocker. h64 is already close to
h128 on validation loss and generation metrics, and h160 did not improve the
short sweep. The current architecture appears close to its practical ceiling on
this dataset and training recipe.

The more likely bottlenecks are data scale, validation noise, and objective or
architecture quality. With roughly 2,687 paired samples and only about 35
Zanzibar validation samples, the validation signal is too small to justify
paying h128 or h160 costs for routine iteration.

## Decision

Use `hidden_channels=64` as the default model width for current OpenEarthMap
training runs. Keep h128 as an optional final-quality probe only when the small
SSIM lift is worth doubling runtime.

Practically, we are close to using the current model architecture to the best of
its abilities on this data. The next high-leverage work should be better data,
a stronger validation split, augmentation or objective changes, and qualitative
inspection of generated samples rather than more width.

## Run Hygiene

For future parallel Modal experiments, use a unique
`SIM_SATELLITE_MODAL_APP_NAME` and spawn detached/background work with
`.spawn()`. Avoid `.remote()` for detached training calls, because Modal can
cancel synchronous calls when the local caller disconnects.
