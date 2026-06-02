# Experiment: Base Model (64x64 Images, 4 Conv Blocks)

## Overview

This experiment uses the BaseModel architecture defined in:

```text
/architectures/BaseModel.py
```

The architecture consists of:

* 4 Convolution Layers
* ReLU Activations
* Max Pooling
* 1 Final Linear Layer

Input images are resized to:

```text
64 x 64
```

This experiment also introduces oversampling to help balance the dataset.

```python
sampler = oversampler(
    dataset=dataset,
    over_sample_rate=8
)
```

The learning rate was increased from:

```text
0.001 → 0.0075
```

compared to previous experiments.

---

## Evaluation

The graph:

```text
/documentation/resources/base_img64_c4_lr0.0075_os.png
```

shows training and validation metrics for:

* Loss
* Accuracy
* Precision
* Recall
* F1 Score

across 100 epochs.

---

## Results

### Loss

Training loss continuously decreases:

```text
~0.9 → ~0.02
```

while validation loss increases:

```text
~0.9 → ~2.3
```

This suggests that the model is fitting the training data very well but struggles to generalize to unseen validation samples.

---

### Accuracy

Training accuracy reaches:

```text
100%
```

while validation accuracy stabilizes around:

```text
72%
```

The large gap between training and validation accuracy indicates overfitting.

---

### Precision

Training precision reaches:

```text
1.00
```

while validation precision remains around:

```text
0.65
```

This means the model performs very well on training data but produces noticeably more incorrect predictions on validation data.

---

### Recall

Training recall reaches:

```text
1.00
```

while validation recall remains around:

```text
0.58
```

Recall is currently the weakest validation metric. This means a significant number of actual events are still being missed.

---

### F1 Score

Training F1 reaches:

```text
1.00
```

while validation F1 stabilizes around:

```text
0.60
```

The large difference again suggests strong overfitting.

---

## Observations

The model is clearly learning useful patterns from the dataset.

Validation accuracy reaches approximately:

```text
72%
```

which is considerably higher than random guessing for a 3-class problem.

However, the model begins to overfit relatively early in training. Training metrics continue improving while validation metrics improve only slightly after around Epoch 60–70.

The current model is able to learn the dataset but does not generalize well enough to unseen data.

---

## Next Steps

The next experiment will focus on reducing overfitting by introducing additional regularization.

### 1. Batch Normalization

Add:

```python
nn.BatchNorm2d()
```

after each convolution layer.

Reason:

* Stabilize training
* Improve feature learning
* Reduce overfitting
* Help the model generalize better

### 2. Dropout

Add:

```python
nn.Dropout()
```

before the final classifier layer.

Reason:

* Prevent memorization
* Reduce reliance on specific neurons
* Improve generalization performance

---

## Conclusion

The model successfully learns meaningful features from the dataset and achieves approximately 72% validation accuracy.

Despite the improvement over previous experiments, the model shows clear signs of overfitting, with training performance reaching nearly perfect scores while validation performance remains significantly lower.

The next experiment will introduce Batch Normalization and Dropout to investigate whether regularization can reduce overfitting and improve validation performance.
