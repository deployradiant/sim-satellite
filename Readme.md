# Sim Satellite

This project is intended to be an end-to-end exploration of diffusion models and steering.

The output is a simple diffusion model that has been trained on satellite images. The satellite images have been labelled to indicate where a house, tree, road etc is. This allows us to create fully fictious satellite landscapes from the diffusion model.

The user can interact with the model through a simple webapp. They can either generate a new image from noise or draw what they would like to see.

### Implementation

The project is written in Python with uv. It uses modal for training of the model and inference. The canvas is a simple html webapp using canvas to render images and enable the drawing functions.

### Data

### Model Architecture

### Training log

[ please update here any entries about the model training ]


### TODO 

- [ ] Implement data loading
- [ ] Implement data labeling
- [ ] test different model architecture
- [ ] Run simple training run
- [ ] Implement evaluation of training
- [ ] Implement web canvas
