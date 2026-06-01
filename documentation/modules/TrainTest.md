# PyTorch Training Utility Functions

This module provides lightweight utility functions for training, evaluating, and tracking PyTorch classification models. It is designed to simplify experimentation by standardizing metric computation, training loops, checkpoint management, and result collection.

The module does not define model architectures, optimizers, or datasets. Instead, it assumes that the provided model implements a custom training interface.

---

## Functional Overview

The module contains three primary functions:

* `train_model()` — Performs a single training epoch and computes training metrics.
* `test_model()` — Evaluates a model on validation or test data without gradient computation.
* `trainer()` — Executes multi-epoch training, optional evaluation, checkpoint saving, and metric logging.

---

## 1. `train_model(model, dataloader, lr=0.001)`

### Description

Performs one complete training epoch over the supplied dataset.

For each batch, the function:

1. Moves data to GPU.
2. Executes forward propagation.
3. Computes loss.
4. Performs backpropagation and parameter updates.
5. Collects predictions and labels for metric computation.

### Parameters

#### `model`

Type: `torch.nn.Module`

Model to be trained.

The model must implement:

* `forward(x)`
* `calculate_loss(logits, labels)`
* `backward(loss, lr)`

#### `dataloader`

Type: `torch.utils.data.DataLoader`

Training dataset iterator.

#### `lr`

Type: `float`

Default: `0.001`

Learning rate used during optimization.

### Returns

Dictionary containing:

| Metric             | Description                   |
| ------------------ | ----------------------------- |
| training_loss      | Average loss over all batches |
| training_accuracy  | Classification accuracy       |
| training_precision | Macro-averaged precision      |
| training_recall    | Macro-averaged recall         |
| training_f1        | Macro-averaged F1 score       |

---

## 2. `test_model(model, dataloader)`

### Description

Evaluates model performance on validation or test data.

The model is automatically switched to evaluation mode and gradients are disabled during execution.

### Parameters

#### `model`

Type: `torch.nn.Module`

Trained model to evaluate.

#### `dataloader`

Type: `torch.utils.data.DataLoader`

Validation or testing dataset iterator.

### Returns

Dictionary containing:

| Metric    | Description              |
| --------- | ------------------------ |
| loss      | Average evaluation loss  |
| accuracy  | Classification accuracy  |
| precision | Macro-averaged precision |
| recall    | Macro-averaged recall    |
| f1        | Macro-averaged F1 score  |

---

## 3. `trainer(...)`

### Description

High-level training loop responsible for:

* Multi-epoch training
* Optional validation/testing
* Progress logging
* Checkpoint saving
* Final model saving
* Metric history collection

### Signature

```python
trainer(
    model,
    train_dataloader,
    test_dataloader=None,
    epoch=1,
    lr=0.001,
    print_on=10,
    save_dir=None,
    save_checkpoints=None
)
```

### Parameters

#### `model`

Type: `torch.nn.Module`

Model to train.

#### `train_dataloader`

Type: `torch.utils.data.DataLoader`

Training dataset.

#### `test_dataloader`

Type: `torch.utils.data.DataLoader`, optional

Validation or testing dataset.

If omitted, only training metrics are collected.

#### `epoch`

Type: `int`

Default: `1`

Number of training epochs.

#### `lr`

Type: `float`

Default: `0.001`

Learning rate.

#### `print_on`

Type: `int`

Default: `10`

Prints training progress every `print_on` epochs.

#### `save_dir`

Type: `str` or `Path`, optional

Directory where model checkpoints and final weights are stored.

#### `save_checkpoints`

Type: `int`, optional

Checkpoint interval.

For example:

```python
save_checkpoints=50
```

saves a checkpoint every 50 epochs.

---

## Checkpointing

Periodic checkpoints are saved as:

```text
checkpoint_epoch_50.pth
checkpoint_epoch_100.pth
checkpoint_epoch_150.pth
...
```

The final model is always saved at the end of training as:

```text
final_model.pth
```

---

## Return Value

Returns a list containing metrics for every epoch.

Example:

```python
[
    {
        "epoch": 1,
        "training_loss": 0.523,
        "training_accuracy": 0.81,
        "training_precision": 0.80,
        "training_recall": 0.82,
        "training_f1": 0.81,
        "test_loss": 0.612,
        "test_accuracy": 0.77,
        "test_precision": 0.76,
        "test_recall": 0.78,
        "test_f1": 0.77
    },
    ...
]
```

This structure can be directly converted into a Pandas DataFrame for visualization, analysis, or experiment tracking.

---

## Assumptions

* CUDA is available and used as the execution device.
* Classification tasks use integer-encoded class labels.
* Metrics are computed using macro averaging.
* The model implements a custom optimization method through:

```python
model.backward(loss, lr)
```