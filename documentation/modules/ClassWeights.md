# Compute Class Weights Function (PyTorch Documentation)

## Overview

The compute_class_weights function is used to calculate class imbalance weights from a PyTorch DataLoader. These weights are used to improve model performance on imbalanced datasets by giving higher importance to underrepresented classes during training.

This approach helps reduce bias toward majority classes and improves recall and F1 score for minority classes.

---

## Purpose

In many real-world datasets, class distribution is uneven. Standard training causes the model to favor majority classes, leading to poor performance on rare classes.

Class weights solve this by:
- Increasing penalty for mistakes on rare classes
- Reducing penalty for frequent classes
- Improving overall balance in learning

---

## Input

The function takes a PyTorch DataLoader as input.

The DataLoader is expected to return batches in the form:
- inputs (images or features)
- labels (integer class indices)

---

## Process

The function performs the following steps:

### 1. Label Extraction
It iterates through the DataLoader and collects all labels from each batch.

---

### 2. Class Frequency Calculation
It counts how many samples exist for each class using frequency counting.

Example distribution:
- Class 0: 5000 samples
- Class 1: 2000 samples
- Class 2: 600 samples

---

### 3. Weight Computation
Class weights are computed using inverse frequency logic:

- Classes with more samples receive lower weights
- Classes with fewer samples receive higher weights

This ensures balanced contribution from all classes during training.

---

## Output

The function returns a tensor of class weights, where each index corresponds to a class label.

These weights can be directly used in loss functions such as CrossEntropyLoss.

---

## Usage

The computed class weights are typically passed into the loss function:

- CrossEntropyLoss uses these weights internally to adjust gradient contribution per class.

---

## Advantages

- Improves performance on imbalanced datasets
- Eliminates need for heavy oversampling
- Stabilizes training compared to repeated sampling
- Enhances minority class recall and F1 score

---

## Limitations

- Requires correct label formatting (integer encoded classes)
- Sensitive to extreme imbalance if not normalized
- Should be used carefully alongside other imbalance techniques to avoid overcompensation

---

## Recommended Use

This method is recommended when:
- Dataset is moderately or highly imbalanced
- Oversampling leads to instability
- Stable training behavior is preferred over data duplication

---

## Summary

The compute_class_weights function is a loss-level imbalance handling technique that improves model fairness by adjusting the contribution of each class during training based on its frequency in the dataset.