# NormModel Architecture

## Overview

`NormModel` is a configurable Convolutional Neural Network (CNN) designed for image classification tasks. The architecture dynamically constructs a stack of convolutional blocks based on the specified number of layers, followed by a manually implemented fully connected classification layer.

The model is intended as a simple baseline architecture for experimentation and learning purposes, while maintaining flexibility through adjustable depth and channel sizes.

## Architecture Structure

The network follows the pattern:

```
[Conv2D] → [ReLU] → [BatchNorm2d] → [MaxPool2D] → [DropOut2D]
                ↓
            Repeat N Times
                ↓
            Flatten
                ↓
            Linear Layer
                ↓
            CrossEntropyLoss
```

## Convolutional Feature Extractor

The feature extraction component is built dynamically using a sequence of convolutional blocks.

Each block consists of:

1. <b>Convolution Layer</b>

   * Kernel Size: 3×3
   * Stride: 1
   * Padding: 1

2. <b>ReLU Activation</b>

3. <b>BatchNorm</b>
    
   * output 

4. <b>Max Pooling</b>

   * Kernel Size: 2×2
   * Stride: 2

5. <b>Dropout</b>
   
   * p: 0.3
    
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
NormModel(
    conv_layers=3,
    initial_output_channel=32,
    initial_image_size=224
)
```

## Flattening

After passing through all convolutional blocks, the resulting feature maps are flattened into a one-dimensional vector.

Flattened size is computed dynamically as:

``` text
Flattened Size =
Final Channels ×
Final Height ×
Final Width
```

where the height and width are reduced after every pooling operation.

## Classification Layer

The model implements the final classification layer manually using trainable parameters:

```python
self.W
self.b
```

Prediction is computed as:

``` math
y = xW + b
```

where:

* x = flattened feature vector
* W = weight matrix
* b = bias vector

## Loss Function

The model uses Cross Entropy Loss:

```python
nn.CrossEntropyLoss()
```

This combines:

1. Softmax activation
2. Negative Log Likelihood Loss

into a single numerically stable operation.

## Manual Parameter Updates

Training step:

1. <b>Compute loss</b>

   ```python
   nn.CrossEntropyLoss()
   ```

2. <b>Perform backpropagation</b>

   ```python
   loss.backward()
   ```

3. <b>Update parameters</b>

   ```python
   param -= lr * param.grad
   ```

4. <b>Reset gradients</b>

   ```python
   param.grad.zero_()
   ```

All trainable parameters returned by `self.parameters()` are updated, including:

* Convolution weights
* Convolution biases
* Classification weights (`W`)
* Classification bias (`b`)

This effectively implements basic Gradient Descent optimization.