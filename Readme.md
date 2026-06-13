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
uv run sim-satellite data openearthmap
uv run sim-satellite train --target modal
uv run sim-satellite infer --target modal
```

The OpenEarthMap data command writes a local `manifest.jsonl` plus `classes.json` under
`data/openearthmap/`. The command can also be called directly as a package script:

```bash
uv run python -m sim_satellite.data.openearthmap
```

For local PyTorch debugging runs, install the optional local extra:

```bash
uv sync --extra local
uv run sim-satellite train --target local --steps 1
uv run sim-satellite infer --target local --steps 1
```

### Data

Data is cached locally under `data/` by default. Prepared samples should track at least the source dataset, split, image path, optional mask path, semantic classes, and preparation metadata.

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
  `uv run sim-satellite train --target modal --steps 8 --batch-size 4 --checkpoint-path artifacts/tiny-sim-satellite.pt`.
  Result: `device=cuda`, `final_loss=0.9386278390884399`, checkpoint written to
  Modal volume path `/artifacts/tiny-sim-satellite.pt`.
- 2026-06-13 03:23 PDT: Pulled the Modal checkpoint locally with
  `uv run modal volume get --force sim-satellite-artifacts tiny-sim-satellite.pt artifacts/tiny-sim-satellite-modal.pt`.
  Verified metadata: `training_device=cuda`, `mask_channels=9`.
- 2026-06-13 03:23 PDT: Verified the web app at `http://127.0.0.1:8765`
  against `artifacts/tiny-sim-satellite-modal.pt`; browser generation produced a
  256px PNG from an OpenEarthMap mask.
