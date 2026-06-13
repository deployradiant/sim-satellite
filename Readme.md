# Sim Satellite

This project is intended to be an end-to-end exploration of diffusion models and steering.

The output is a simple diffusion model that has been trained on satellite images. The satellite images have been labelled to indicate where a house, tree, road etc is. This allows us to create fully fictious satellite landscapes from the diffusion model.

The user can interact with the model through a simple webapp. They can either generate a new image from noise or draw what they would like to see.

### Implementation

The project is written in Python with uv. It uses Modal by default for training and inference. The canvas is a simple HTML webapp using canvas to render images and enable the drawing functions.

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

The first dataset target is [OpenEarthMap](https://open-earth-map.org/overview.html). It has 5,000 high-resolution aerial and satellite images with manually annotated land-cover masks across 97 regions in 44 countries. That should be enough for a small end-to-end mask-conditioned diffusion run and for proving the data/model/inference loop. It is not enough to train a broad satellite foundation model from scratch, so later runs should either fine-tune a pretrained image diffusion backbone or add more compatible mask-labeled data.

The long-term goal is not just image generation from captions. The data pipeline should support semantic masks because the drawing workflow needs spatial control. A practical labeling pipeline will likely combine automatic segmentation with review:

1. tile source satellite imagery into fixed-size RGB images
2. generate candidate object/region masks with a segmentation model such as SAM
3. assign semantic labels to masks with a smaller vision-language or classification model
4. merge labels into a single class-index mask
5. keep confidence scores and route uncertain masks through human review

This keeps segmentation, labeling, and quality control separate enough to improve each stage without rewriting the model training code.

Broader semi-automated labeling with SAM, SAM 3, or a mask classifier is a later research track. For the first end-to-end run, use existing OpenEarthMap masks directly.

### Model Architecture

Start with the smallest architecture that can run end to end. A custom DDPM is easier to understand and debug, but a PyTorch/diffusers-based implementation is likely faster for useful first results. The important boundary is that conditioning enters through semantic masks, so the architecture can evolve without changing the dataset format.

### References

- [OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping](https://open-earth-map.org/overview.html) (Xia et al., WACV 2023).
- [DiffusionSat: A Generative Foundation Model for Satellite Imagery](https://arxiv.org/abs/2312.03606) (Khanna et al., ICLR 2024).
  This is useful prior work because it treats satellite imagery as a first-class diffusion domain and conditions on remote-sensing metadata such as geolocation. Our working hypothesis is different: metadata and text are not precise enough for the drawing workflow. The core conditioning input should be a semantic mask with labels such as roads, trees, houses, water, and fields so generation can follow a user-drawn layout.

### Milestones


#### Fast end-to-end run:

- download a small public satellite dataset into a local cache
- normalize image tiles and semantic masks into a simple manifest
- train a small mask-conditioned diffusion model
- run inference from a mask input


### Training log

[ please update here any entries about the model training ]


### TODO 

- [ ] Implement data loading
- [ ] Implement data labeling
- [ ] test different model architecture
- [ ] Run simple training run
- [ ] Implement evaluation of training
- [ ] Implement web canvas
