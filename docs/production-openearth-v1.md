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
