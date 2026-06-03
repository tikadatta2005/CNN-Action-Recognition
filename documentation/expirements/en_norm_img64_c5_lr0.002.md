# Enhanced Norm Model: 64x64 & 5 Conv Blocks
## Overview
This experiment uses the EnhancedNormModel architecture `/architectures/EnhancedNormModel.py`. The architecture is documented briefly: `/documentation/architectures/en_norm_model.md`.

The process is almost same as mentioned in `/documentation/expirements/en_norm_img64_c4_lr0.002.md`.

This experiment uses `5` Convolutional Blocks followed with `BatchNorm2D` and `MaxPool2D` each block.

The model uses the `Dropout` for 3rd and 5th Convolutional Blocks.

Overall the EnhancedNormModel looks like:

``` python
model = EnhancedNormModel(
    conv_layers=4, 
    normalizations = [1, 1, 1, 1],
    poolings = [1,1,1,1],
    dropouts = [0, 0, 1, 0, 1],
    initial_output_channel=16, 
    initial_image_size=64,
    dropout_p = 0.2
    ).to("cuda")
```
The experiment has implemented stricter early_stopping_patience. It was 20 but now is only 6. It looks like:

``` python
metrics = trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    epoch=30,
    lr=0.002,
    print_on=1,
    save_dir="../models/en_norm_img64_c5_lr_0.002_os/",
    save_checkpoints=1,
    checkpoint_name="en_norm_64x64_train_0_",
    early_stop_patience = 6,
    min_delta = 0.01
)
```
This early stopping did a great job stopping the model that was not performing any better. The trainer stopped traing at epoch 16.

The experiment very much outperformed previous experiments just because of these changes.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch seven.

### Evaluation From Metrics
The visualized graph `/documentation/resources/en_norm_img64_c5_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows smooth and stable loss drop for training set, but shows some small jumps and falls in validation but better stablization previous experiments.
* The accuracy graph shows model quicky learning and making nice accuracy for training set, It is also shows a good and smoother learning and improving behaviour.
* The precision graph shows very fast learning for training set but slow for validation set.
* The recall graph shows gradual learning and reaching cieling for training set but unstable and hills for validation set.
* The F1 graph shows gradual learning and reaching cieling for training set and slow for validation score.

The model has given best results on `Epoch 10` if only seen validation set:
* Accuracy = 0.73
* Recall = 0.65
* Precison = 0.63
* F1 = 0.64

This experiment has not given the best outputs but shows stable learning of model. The model is not moving beyond is likely due to following reasons:
* Very Small Input Size (64x64)

- The 5 Blocks reduces images almost to 4x4 from 64x64.

### Next Steps
* Increase Image/Input Size to (112 x 112)