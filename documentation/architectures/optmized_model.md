# OptimizedModel Architecture

## Overview

`OptimizedModel` is a dynamic CNN designed for image classification tasks. The architecture constructs convolutional blocks based on user-defined configuration and supports class-weighted loss for handling imbalanced datasets.

Unlike traditional fixed training pipelines, this model supports an **external optimizer attachment system**, allowing flexible integration with modern optimization strategies such as adaptive optimizers and learning rate schedulers.

The model preserves a simple training interface:

```python
model.backward(loss)
```

while delegating optimization control through:

```python
model.set_optimizer(optimizer)
```

---

## Architecture Structure

```
[Conv2D] → [ReLU] → [Optional BatchNorm2d] → [Optional MaxPool2D] → [Optional Dropout2D]
                              |
                        Repeat N Times
                              |
                           Flatten
                              |
                 Linear Classification Layer
                              |
              CrossEntropyLoss (with optional class weights)
```

---

## Convolutional Feature Extractor

Each convolutional block contains:

### 1. Convolution Layer

| Parameter   | Value |
|-------------|-------|
| Kernel Size | 3×3   |
| Stride      | 1     |
| Padding     | 1     |

### 2. ReLU Activation

### 3. Optional Batch Normalization

- Controlled by `normalizations` array (boolean or integer)
- Applies `BatchNorm2d(out_channels)` when enabled

### 4. Optional Max Pooling

- Controlled by `poolings` array (boolean or integer)
- Kernel Size: 2×2, Stride: 2
- Reduces spatial dimensions by half when applied

### 5. Optional Dropout

- Controlled by `dropouts` array (boolean or integer)
- Applies `Dropout2d(p=dropout_p)` when enabled

---

## Channel Progression

Output channels are doubled after each convolutional block. Spatial dimensions are reduced only if pooling is applied.

```
Conv1: C   → 2C
Conv2: 2C  → 4C
Conv3: 4C  → 8C
```

---

## Example Configuration

```python
OptimizedModel(
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

---

## Flattening

```
Flattened Size = Final Channels × Final Height × Final Width
```

---

## Classification Layer

`self.fc` performs linear projection:

```
y = xW + b
```

| Symbol | Description              |
|--------|--------------------------|
| `x`    | Flattened feature vector |
| `W`    | Learnable weights        |
| `b`    | Learnable bias           |
| `y`    | Output logits            |

---

## Loss Function

```python
if class_weights is not None:
    nn.CrossEntropyLoss(weight=class_weights)
else:
    nn.CrossEntropyLoss()
```

---

## Optimizer System

Supports external optimizer attachment:

```python
model.set_optimizer(optimizer)
```

**Supported optimizers:**
- Adam
- SGD
- RMSProp
- OneCycleLR-compatible optimizers

---

## Training Procedure

1. Forward pass
2. Compute loss
3. Backpropagation via `model.backward(loss)`
4. Optimizer step
5. Zero gradients

### Usage

```python
preds = model(x)
loss = model.calculate_loss(preds, y)
model.backward(loss)
```

### `backward()` Internals

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
return loss.item()
```

---

## Trainable Parameters

- Convolution weights and biases
- BatchNorm parameters (if enabled)
- Fully connected weights and bias