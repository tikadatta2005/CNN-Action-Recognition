import torch
from collections import Counter

def compute_class_weights(dataloader):
    """
    Computes class weights from a PyTorch DataLoader.

    Args:
        dataloader: PyTorch DataLoader (or dataset wrapped in DataLoader)

    Returns:
        torch.FloatTensor: class weights tensor
    """

    # collect all labels
    all_labels = []

    for batch in dataloader:
        # case 1: (images, labels)
        if isinstance(batch, (list, tuple)):
            labels = batch[1]
        else:
            raise ValueError("Dataloader must return (inputs, labels)")

        all_labels.extend(labels.cpu().numpy())

    # count classes
    class_counts = Counter(all_labels)

    num_classes = len(class_counts)

    # compute weights (inverse frequency)
    class_weights = [
        len(all_labels) / class_counts[i]
        for i in range(num_classes)
    ]

    return torch.tensor(class_weights, dtype=torch.float)