# DynamicModel Architecture

## Overview

`DynamicModel` is a configurable Convolutional Neural Network (CNN) designed for image classification tasks. The architecture dynamically constructs both convolutional and fully connected layers based on user-provided configurations, allowing experimentation with different network depths, normalization strategies, pooling operations, dropout regularization, and dense layer structures.

The model combines:

* Dynamic convolutional feature extraction
* Optional Batch Normalization
* Optional Max Pooling
* Optional Spatial Dropout
* Configurable Fully Connected Layers
* Manually implemented output classification layer
* Manual Gradient Descent parameter updates

The architecture is intended for educational purposes, experimentation, and understanding the internal mechanics of neural networks without relying on PyTorch optimizers.

---

## Architecture Structure

The network follows the general pattern:

```text
Input Image
      ↓
[Conv2D]
      ↓
[Optional BatchNorm2D]
      ↓
[ReLU]
      ↓
[Optional MaxPool2D]
      ↓
[Optional Dropout2D]
      ↓
      Repeat N Times
      ↓
Flatten
      ↓
[Linear + Activation]
      ↓
      Repeat M Times
      ↓
Manual Output Layer (W,b)
      ↓
Class Scores (Logits)
      ↓
CrossEntropyLoss
```

where:

* **N** = Number of convolutional blocks (`conv_layers`)
* **M** = Number of fully connected layers defined by `activations` and `activation_outputs`

---

## Input Parameters

### Constructor

```python
DynamicModel(
    conv_layers,
    normalizations,
    poolings,
    dropouts,
    initial_output_channel,
    initial_image_size,
    activations,
    activation_outputs,
    dropout_p=0
)
```

### Parameters

| Parameter                | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `conv_layers`            | Number of convolutional blocks to create                           |
| `normalizations`         | Array controlling BatchNorm2D insertion for each convolution block |
| `poolings`               | Array controlling MaxPool2D insertion for each block               |
| `dropouts`               | Array controlling Dropout2D insertion for each block               |
| `initial_output_channel` | Output channels of the first convolution layer                     |
| `initial_image_size`     | Input image height/width                                           |
| `activations`            | List of activation names for dense layers                          |
| `activation_outputs`     | Output dimensions for dense layers                                 |
| `dropout_p`              | Dropout probability used by Dropout2D                              |

---

## Validation Checks

The model validates the lengths of configuration arrays.

Required:

```python
len(normalizations) == conv_layers
len(poolings) == conv_layers
len(dropouts) == conv_layers
```

If the lengths do not match, warning messages are displayed.

Example:

```python
conv_layers = 3

normalizations = [1,1,1]
poolings = [1,0,1]
dropouts = [0,1,1]
```

Each index corresponds to one convolution block.

---

## Convolutional Feature Extractor

The convolutional portion of the network is constructed dynamically using a loop.

Each convolution block contains:

### 1. Convolution Layer

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1
)
```

Properties:

| Parameter   | Value |
| ----------- | ----- |
| Kernel Size | 3×3   |
| Stride      | 1     |
| Padding     | 1     |

Padding preserves spatial dimensions before pooling.

---

### 2. Optional Batch Normalization

Added when:

```python
normalizations[i] == 1
```

or

```python
normalizations[i] == True
```

Layer:

```python
nn.BatchNorm2d(out_channels)
```

Purpose:

* Stabilizes training
* Reduces internal covariate shift
* Allows larger learning rates
* Improves gradient flow

---

### 3. ReLU Activation

Every convolution block contains:

```python
nn.ReLU()
```

Function:

```text
f(x) = max(0,x)
```

Purpose:

* Introduces non-linearity
* Prevents linear feature extraction
* Improves convergence speed

---

### 4. Optional Max Pooling

Added when:

```python
poolings[i] == 1
```

or

```python
poolings[i] == True
```

Layer:

```python
nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

Effect:

```text
224 × 224
     ↓
112 × 112
```

The image dimensions are halved after each pooling operation.

---

### 5. Optional Dropout2D

Added when:

```python
dropouts[i] == 1
```

or

```python
dropouts[i] == True
```

Layer:

```python
nn.Dropout2d(p=dropout_p)
```

Purpose:

* Reduces overfitting
* Improves generalization
* Randomly drops feature maps during training

Example:

```python
dropout_p = 0.3
```

meaning 30% of feature maps are randomly disabled during training.

---

## Channel Expansion Strategy

The number of channels doubles after every convolution block.

Rule:

```python
out_channels = out_channels * 2
```

Example:

| Layer | Input Channels | Output Channels |
| ----- | -------------- | --------------- |
| Conv1 | 3              | 32              |
| Conv2 | 32             | 64              |
| Conv3 | 64             | 128             |
| Conv4 | 128            | 256             |

Example configuration:

```python
DynamicModel(
    conv_layers=4,
    initial_output_channel=32,
    ...
)
```

produces:

```text
3 → 32 → 64 → 128 → 256
```

channel progression.

---

## Flattening Layer

After all convolution blocks, feature maps are converted into a one-dimensional vector.

Operation:

```python
x = x.view(x.size(0), -1)
```

Shape transformation:

```text
(B, C, H, W)
       ↓
(B, C×H×W)
```

where:

* B = Batch Size
* C = Channels
* H = Height
* W = Width

---

## Flattened Size Calculation

The flattened feature size is calculated dynamically.

Formula:

```text
Flattened Size =
Final Channels ×
Final Height ×
Final Width
```

Implemented as:

```python
flattened_size =
    in_channels *
    self.image_size *
    self.image_size
```

The image size is updated whenever a pooling layer is applied.

Example:

```text
Input Image = 224 × 224

Pooling Layers = 3

224
 ↓
112
 ↓
56
 ↓
28
```

If final channels = 128:

```text
Flattened Size =
128 × 28 × 28
=
100352
```

---

## Fully Connected Layers

The dense section is also created dynamically.

Supported activations:

```python
{
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}
```

Each dense block consists of:

```text
Linear
   ↓
Activation
```

Construction:

```python
nn.Linear(
    current_features,
    output_val
)
```

followed by the specified activation.

---

### Example

```python
activations = [
    "relu",
    "relu",
    "sigmoid"
]

activation_outputs = [
    512,
    128,
    64
]
```

Produces:

```text
Flatten
   ↓
Linear(... → 512)
   ↓
ReLU
   ↓
Linear(512 → 128)
   ↓
ReLU
   ↓
Linear(128 → 64)
   ↓
Sigmoid
```

---

## Output Classification Layer

Instead of using:

```python
nn.Linear()
```

the final classifier is implemented manually.

Trainable parameters:

```python
self.W
self.b
```

Initialization:

```python
self.W =
torch.randn(
    current_features,
    3
) * 0.01

self.b =
torch.zeros(3)
```

where:

* 3 = Number of classes

---

### Forward Computation

Prediction is computed as:

```text
y = xW + b
```

Implemented as:

```python
x = x @ self.W + self.b
```

Output shape:

```text
(Batch Size, 3)
```

These outputs are raw logits.

---

## Forward Pass

The complete forward propagation sequence is:

```python
x = self.conv(x)

x = x.view(
    x.size(0),
    -1
)

x = self.fc(x)

x = x @ self.W + self.b
```

Flow:

```text
Input
 ↓
Convolution Blocks
 ↓
Flatten
 ↓
Dense Layers
 ↓
Output Layer
 ↓
Logits
```

---

## Loss Function

The model uses:

```python
nn.CrossEntropyLoss()
```

stored as:

```python
self.criterion
```

Loss calculation:

```python
loss =
self.criterion(
    y_pred,
    y
)
```

CrossEntropyLoss internally performs:

```text
Softmax
    +
Negative Log Likelihood
```

in a numerically stable manner.

---

## Manual Gradient Descent

The model does not use:

```python
torch.optim.SGD
```

or any other optimizer.

Instead, parameter updates are implemented manually.

---

### Step 1: Backpropagation

```python
loss.backward()
```

Computes gradients for all trainable parameters.

---

### Step 2: Parameter Update

```python
param -= lr * param.grad
```

for every parameter returned by:

```python
self.parameters()
```

This applies Gradient Descent:

```text
θ = θ - α∇J(θ)
```

where:

* θ = parameter
* α = learning rate
* ∇J(θ) = gradient

---

### Step 3: Gradient Reset

```python
param.grad.zero_()
```

Prevents gradient accumulation between iterations.

---

## Trainable Parameters

The following parameters are optimized:

### Convolution Layers

* Convolution weights
* Convolution biases

### Batch Normalization Layers

* Scale parameters (γ)
* Shift parameters (β)

### Fully Connected Layers

* Dense layer weights
* Dense layer biases

### Output Layer

* Classification weight matrix (`W`)
* Classification bias vector (`b`)

All trainable parameters are automatically included through:

```python
self.parameters()
```

during the manual update step.

---

## Example Configuration

```python
model = DynamicModel(
    conv_layers=3,
    normalizations=[1,1,1],
    poolings=[1,1,1],
    dropouts=[0,1,1],
    initial_output_channel=32,
    initial_image_size=224,
    activations=["relu", "relu"],
    activation_outputs=[512,128],
    dropout_p=0.3
)
```

Resulting architecture:

```text
Input
 ↓
Conv(3→32)
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool

 ↓
Conv(32→64)
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Dropout

 ↓
Conv(64→128)
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Dropout

 ↓
Flatten

 ↓
Linear(...→512)
 ↓
ReLU

 ↓
Linear(512→128)
 ↓
ReLU

 ↓
Manual Output Layer
 ↓
3-Class Prediction
```
