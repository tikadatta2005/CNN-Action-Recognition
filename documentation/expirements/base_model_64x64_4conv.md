# Base Model: 64x64 & 4 Conv Blocks
This model uses the BaseModel architecture present in `/architectures/BaseModel.py`. The model uses images of 64x64 for training and 4 conv blocks.
`datasets` and `transforms` from `torchvision` is used for image loading and transformations. Here, `DataLoader` from `torch.utils.data` is imported for making batches, and loading overall data.

* Image is loaded and resized to (64x64)
* Image is transformed to Tensor
* Batches of size 32 is made using DataLoader

