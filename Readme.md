# Sim Satellite

This project is intended to be an end-to-end exploration of diffusion models and steering.

The output is a simple diffusion model that has been trained on satellite images. The satellite images have been labelled to indicate where a house, tree, road etc is. This allows us to create fully fictious satellite landscapes from the diffusion model.

The user can interact with the model through a simple webapp. They can either generate a new image from noise or draw what they would like to see.

### Implementation

The project is written in Python with uv. It uses Modal by default for training and inference, with local execution as a fallback for small smoke tests. The canvas is a simple HTML webapp using canvas to render images and enable the drawing functions.



### Data

Data should be cached locally under `data/` by default. Prepared samples should track at least the source dataset, split, image path, optional mask path, semantic classes, and preparation metadata.

The long-term goal is not just image generation from captions. The data pipeline should support semantic masks because the drawing workflow needs spatial control. A practical labeling pipeline will likely combine automatic segmentation with review:

1. tile source satellite imagery into fixed-size RGB images
2. generate candidate object/region masks with a segmentation model such as SAM
3. assign semantic labels to masks with a smaller vision-language or classification model
4. merge labels into a single class-index mask
5. keep confidence scores and route uncertain masks through human review

This keeps segmentation, labeling, and quality control separate enough to improve each stage without rewriting the model training code.

### Model Architecture

Start with the smallest architecture that can run end to end. A custom DDPM is easier to understand and debug, but a PyTorch/diffusers-based implementation is likely faster for useful first results. The important boundary is that conditioning enters through semantic masks, so the architecture can evolve without changing the dataset format.

### References

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
