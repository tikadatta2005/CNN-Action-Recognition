from collections import Counter
from torch.utils.data import WeightedRandomSampler
import torch


def oversampler(dataset, oversample_rate=1):

    targets = dataset.targets

    class_counts = Counter(targets)

    sample_weights = [
        1.0 / class_counts[label]
        for label in targets
    ]

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset) * oversample_rate,
        replacement=True
    )

    return sampler