# Base Model

## Overview

The Base Model is a configurable Convolutional Neural Network (CNN) designed for image classification. It serves as the baseline architecture for evaluating violence and alert-level detection performance.

The architecture allows the number of convolutional layers and initial channel size to be adjusted while maintaining a consistent design pattern.

## Architecture

The model follows the structure:

```text
Input
 ↓
[Conv2D → ReLU → MaxPool]
 ↓
[Repeat N Times]
 ↓
Flatten
 ↓
Linear Layer
 ↓
Output Classes
```

## Convolutional Layers

Each convolutional block consists of:

* Conv2D (3×3 kernel)
* ReLU activation
* MaxPool2D (2×2)

The number of blocks is determined by the `conv_layers` parameter.

## Feature Expansion

The model progressively increases feature channels after each convolutional block.

Example:

```text
3 → 16 → 32 → 64 → 128
```

This enables deeper layers to learn increasingly complex visual patterns.

## Spatial Reduction

Max pooling reduces the spatial dimensions of feature maps by half after every block.

Example for a 224×224 image:

```text
224×224
↓
112×112
↓
56×56
↓
28×28
```

This reduces computational cost while retaining important features.

## Classification Layer

After feature extraction, the feature maps are flattened and passed through a fully connected layer.

The output layer produces predictions for three classes:

* Alert
* High Alert
* Normal

## Loss Function

Cross Entropy Loss is used for multi-class classification.

## Purpose

The Base Model is intended to:

* Establish a performance baseline.
* Evaluate the effectiveness of a simple CNN architecture.
* Provide a foundation for comparison with deeper and more advanced architectures in subsequent experiments.
