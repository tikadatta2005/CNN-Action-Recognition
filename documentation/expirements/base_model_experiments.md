# Base Model

The **Base Model** serves as the initial benchmark architecture for the project. Its implementation is located in `/architectures/BaseModel.py`.

This model follows a conventional Convolutional Neural Network (CNN) design consisting of repeated convolutional feature extraction blocks followed by a classification head:

`[Conv2D] → [ReLU] → [MaxPooling] → ... → [Fully Connected Layers]`

The architecture is designed to support a configurable number of convolutional blocks, allowing experimentation with different network depths while maintaining the same overall structure. This flexibility enables the model to be reused across multiple training configurations and serves as a foundation for evaluating more advanced architectures.

## Expirement with 64x64 Images and 4 Convolutional blocks
