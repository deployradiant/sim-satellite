# production-openearth-v1

`production-openearth-v1` is a calibrated fine-class OpenEarthMap segmentation
operating point for unseen satellite images.

It uses the SMP U-Net EfficientNet-B4 SCSE checkpoint at:

```text
artifacts/segmentation-openearthmap-smp-unet-efficientnet-b4-scse-official-xbd-full-polish-v1/smp-unet-efficientnet-b4-scse-official-xbd-full-polish-s2500.pt
```

Inference averages overlapping tiled logits at scales `0.75`, `1.0`, and
`1.25`, uses feather blending, applies fixed class logit biases, then keeps only
pixels whose maximum softmax confidence is at least `0.8`. Lower-confidence
pixels are written as `unlabeled`.

On official OpenEarth validation, this selected-pixel operating point measured:

- labeled-pixel accuracy: `0.9263`
- selected mean IoU: `0.8327`
- selected labeled-pixel coverage: `0.6196`

The metric is intentionally not dense full-image accuracy. It labels about 62%
of labeled validation pixels and abstains on the rest to avoid auto-labeling
uncertain predictions.

```bash
uv run sim-satellite segmentation-predict \
  --inference-preset production-openearth-v1 \
  --images-dir data/unseen/images
```

## Training Practice

The production recipe is committed as:

```text
train.openearthmap-segmentation-production-openearth-v1.json
```

It fine-tunes the SMP U-Net EfficientNet-B4 SCSE model from the previous polish
checkpoint, using the official OpenEarth `train` split and validating on `val`.
The recipe uses `512` pixel crops, batch size `4`, cosine learning rate decay
from `2e-6` to `2e-7`, Dice plus class-weighted cross entropy, low unlabeled
weight, class-weight multipliers for hard built/road/vegetation classes, and
overlapping feathered validation tiles with stride `256`.

Run it on Modal:

```bash
uv run sim-satellite segmentation-train \
  train.openearthmap-segmentation-production-openearth-v1.json
```

Then evaluate the trained checkpoint with the same production inference
calibration:

```bash
uv run sim-satellite segmentation-evaluate \
  --target modal \
  --inference-preset production-openearth-v1 \
  --manifest data/openearthmap/manifest.jsonl \
  --split val \
  --max-samples 0 \
  --output artifacts/modal-metrics/production-openearth-v1-official-val.json
```
