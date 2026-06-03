# Enhanced Norm Model: 64x64 & 4 Conv Blocks
## Overview
This experiment uses the EnhancedNormModel architecture `/architectures/EnhancedNormModel.py`. The architecture is documented briefly: `/documentation/architectures/en_norm_model.md`.

The process has been changed in most of the parts of this experiment but it still uses almost all same modules as previous experiments.

This experiment uses lower learning rate:

``` math
0.002
```

This experiment uses 4 Convolutional Blocks followed with `BatchNorm2D` and `MaxPool2D` each block. The model uses the `Dropout` skipping one block so total 2 blocks uses `Dropout`. The experiment also increased initial_input_channel from 8 to 16 for better feature extraction: 

``` python
model = EnhancedNormModel(
    conv_layers=4, 
    normalizations = [1, 1, 1, 1],
    poolings = [1,1,1,1],
    dropouts = [0, 1, 0, 1],
    initial_output_channel=16, 
    initial_image_size=64,
    dropout_p = 0.2
    ).to("cuda")
```

The experiment has also added the additional Image Augmentations. The experiment uses following transformations:

``` python
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2)
])
```
This transformation created differences like flips, rotatopms. zitter, erase of elements and resize for model to better focus on different aspect of the image.

The experiment also uses less Ovesampling of `2` instead of `8`:

``` python
sampler = oversampler(
    dataset=train_dataset,
    oversample_rate= 2
)
```

The experiment very much outperformed previous experiments just because of these changes.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch seven.

### Evaluation From Metrics
The visualized graph `/documentation/resources/en_norm_img64_c4_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows smooth and stable loss drop for training set, but shows some small jumps and falls in validation. 
* The accuracy graph shows model quicky learning and making nice accuracy for training set, It is also giving a good enough though bumpy rise for validation set.
* The precision graph shows very fast learning for training set but slow and bumpy for validation set.
* The recall graph shows gradual learning and reaching cieling for training set but unstable and hills for validation set.
* The F1 graph shows gradual learning and reaching cieling for training set and slow bit unstable for validation score.

The experiment has given the best results by far. The model has given best results on `Epoch 26` if only seen validation set:
* Accuracy = 0.76
* Recall = 0.69
* Precison = 0.69
* F1 = 0.67

The model still shows unstablity for valid stills. It's because the validation set is too small. Also some possible problems can be:
* Less Convolutional Depth
* Low Input Size

### Next Steps
* Add one more conv Layer
* Add BatchNorm2D for each layer
* Keep the Dropout only for deeper layers