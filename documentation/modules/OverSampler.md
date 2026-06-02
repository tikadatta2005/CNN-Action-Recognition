# Oversampler

## What It Does

The `oversampler()` function creates a PyTorch `WeightedRandomSampler` to help balance imbalanced datasets during training.

Classes with fewer samples are selected more often, while classes with many samples are selected less often.

---

## How It Works

1. Reads all labels from `dataset.targets`.
2. Counts how many samples each class has.
3. Assigns higher weights to samples from smaller classes.
4. Creates a `WeightedRandomSampler` using those weights.

The sampler does not create new images. It simply changes how often samples are selected during training.

---

## Oversample Rate

`oversample_rate` controls how many samples are drawn per epoch.

Example:

- Dataset size = 100
- `oversample_rate=1` → 100 samples per epoch
- `oversample_rate=2` → 200 samples per epoch
- `oversample_rate=3` → 300 samples per epoch

---

## Requirements

The dataset must contain a `targets` attribute (e.g., `torchvision.datasets.ImageFolder`).