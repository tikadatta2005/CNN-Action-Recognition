# WeightLossModel Architecture (Updated)

## Overview

`WeightLossModel` is a dynamic Convolutional Neural Network (CNN) designed for image classification tasks. The architecture dynamically constructs convolutional blocks based on user-defined configuration and supports class-weighted loss for handling class imbalance.

The model removes dependency on oversampling strategies and instead relies on loss-level weighting for stable and balanced learning.

## Architecture Structure

```
[Conv2D] -> [ReLU] -> [Optional BatchNorm2d] -> [Optional MaxPool2D] -> [Optional DropOut2D]
                |
            Repeat N Times
                |
            Flatten
                |
        Linear Classification Layer
                |
     CrossEntropyLoss (with optional class weights)

```
## Convolutional Feature Extractor

Each convolutional block contains:

1. <b>Convolution Layer</b>
- Kernel Size: 3x3
- Stride: 1
- Padding: 1

2. <b>ReLU Activation</b>

3. <b>Optional Batch Normalization</b>
- Controlled by normalizations array (boolean or int)
- Adds BatchNorm2d(out_channels) if enabled

4. <b>Optional Max Pooling</b>
- Controlled by poolings array (boolean or int)
- Kernel Size: 2x2
- Stride: 2
- Reduces spatial size by half

5. <b>Optional Dropout</b>
- Controlled by dropouts array (boolean or int)
- Uses Dropout2d(p=dropout_p)


After each block:
* Output channels are doubled
* Spatial dimensions are reduced if pooling is enabled

Example:
```
Conv1: 3 -> 32
Conv2: 32 -> 64
Conv3: 64 -> 128
```

### Example Configuration
``` python
WeightLossModel(
    conv_layers=3,
    initial_output_channel=32,
    initial_image_size=224,
    normalizations=[1, 1, 1],
    poolings=[1, 1, 1],
    dropouts=[0, 1, 1],
    dropout_p=0.2,
    class_weights=class_weights
)
```

## Flattening

``` text
Flattened Size = Final Channels * Final Height * Final Width
```

## Classification Layer

``` text
self.W
self.b
```
## Prediction:

``` math
y = xW + b
```

Where:
* x = flattened feature vector
* W = learnable weights
* b = learnable bias

## Loss Function

CrossEntropyLoss with optional class weights:

``` python
if class_weights is not None:
    nn.CrossEntropyLoss(weight=class_weights)
else:
    nn.CrossEntropyLoss()
```

## Training Procedure

1. Forward pass
2. Compute loss
3. Backpropagation (loss.backward())
4. Manual parameter update:
   param -= lr * param.grad
5. Zero gradients

## Trainable Parameters:
- Conv weights
- Conv biases
- Fully connected weights (W)
- Fully connected bias (b)