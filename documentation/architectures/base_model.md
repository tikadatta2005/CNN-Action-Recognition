# BaseModel Architecture

## Overview

`BaseModel` is a configurable Convolutional Neural Network (CNN) designed for image classification tasks. The architecture dynamically constructs a stack of convolutional blocks based on the specified number of layers, followed by a manually implemented fully connected classification layer.

The model is intended as a simple baseline architecture for experimentation and learning purposes, while maintaining flexibility through adjustable depth and channel sizes.

---

## Architecture Structure

The network follows the pattern:

```
[Conv2D] → [ReLU] → [MaxPool2D]
                ↓
            Repeat N Times
                ↓
            Flatten
                ↓
         Linear Layer
                ↓
            Softmax*
```

* Softmax is internally handled by `CrossEntropyLoss` during training.

---

## Convolutional Feature Extractor

The feature extraction component is built dynamically using a sequence of convolutional blocks.

Each block consists of:

1. Convolution Layer

   * Kernel Size: 3×3
   * Stride: 1
   * Padding: 1

2. ReLU Activation

3. Max Pooling

   * Kernel Size: 2×2
   * Stride: 2

After each block:

* Spatial dimensions are reduced by half.
* Output channels are doubled.

Example configuration:

| Layer  | Input Channels | Output Channels |
| ------ | -------------- | --------------- |
| Conv 1 | 3              | 32              |
| Conv 2 | 32             | 64              |
| Conv 3 | 64             | 128             |

for:

```python
BaseModel(
    conv_layers=3,
    initial_output_channel=32,
    initial_image_size=224
)
```

---

## Flattening

After passing through all convolutional blocks, the resulting feature maps are flattened into a one-dimensional vector.

Flattened size is computed dynamically as:

```
Flattened Size =
Final Channels ×
Final Height ×
Final Width
```

where the height and width are reduced after every pooling operation.

---

## Classification Layer

Instead of using `nn.Linear`, the model implements the final classification layer manually using trainable parameters:

```python
self.W
self.b
```

Prediction is computed as:

```
y = xW + b
```

where:

* x = flattened feature vector
* W = weight matrix
* b = bias vector

The output dimension is fixed to 3 classes.

---

## Loss Function

The model uses Cross Entropy Loss:

```python
nn.CrossEntropyLoss()
```

This combines:

1. Softmax activation
2. Negative Log Likelihood Loss

into a single numerically stable operation.

---

## Manual Parameter Updates

Instead of using PyTorch optimizers such as SGD or Adam, the model performs gradient updates manually.

Training step:

1. Compute loss
2. Perform backpropagation

```python
loss.backward()
```

3. Update parameters

```python
param -= lr * param.grad
```

4. Reset gradients

```python
param.grad.zero_()
```

All trainable parameters returned by `self.parameters()` are updated, including:

* Convolution weights
* Convolution biases
* Classification weights (`W`)
* Classification bias (`b`)

This effectively implements basic Gradient Descent optimization.

## Intended Use

This architecture serves as a baseline CNN for:

* Image classification experiments
* Deep learning education
* Understanding backpropagation
* Understanding manual parameter optimization
* Comparing against more advanced CNN architectures
