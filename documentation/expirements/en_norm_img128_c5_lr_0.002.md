# Enhanced Norm Model: 128x128 & 5 Conv Blocks
## Overview
This experiment uses the EnhancedNormModel architecture `/architectures/EnhancedNormModel.py`. The architecture is documented briefly: `/documentation/architectures/en_norm_model.md`.

The process is almost same as mentioned in `/documentation/expirements/en_norm_img64_c5_lr0.002.md`.

The only change made was increasing input size from 64x64 to 128x128.

Overall the EnhancedNormModel looks like:

``` python
model = EnhancedNormModel(
    conv_layers=4, 
    normalizations = [1, 1, 1, 1],
    poolings = [1,1,1,1],
    dropouts = [0, 0, 1, 0, 1],
    initial_output_channel=16, 
    initial_image_size=128,
    dropout_p = 0.2
    ).to("cuda")
```
The experiment has implemented less strict early_stopping_patience of 8.

``` python
metrics = trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    epoch=30,
    lr=0.002,
    print_on=1,
    save_dir="../models/en_norm_img128_c5_lr_0.002_os/",
    save_checkpoints=1,
    checkpoint_name="en_norm_128x128_train_0_",
    early_stop_patience = 8,
    min_delta = 0.01
)
```
This early stopping did a great job stopping the model that was not performing any better. The trainer stopped traing at epoch 20.

The experiment very much outperformed previous experiments just because of change in input size changes.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch seven.

### Evaluation From Metrics
The visualized graph `/documentation/resources/en_norm_img128_c5_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows smooth and stable loss drop for training set, but shows some small jumps and falls in validation but better stablization than previous experiments.
* The accuracy graph shows model shows both lines increasing in similar manner. Still the validation line remains a bit unstable.
* The precision graph shows very fast learning for training set but slow for validation set almost sitting on same precison score.
* The recall graph shows gradual learning and reaching cieling for training set but unstable and hills & plateus for validation set.
* The F1 graph shows gradual learning and reaching cieling for training set and slow for validation score with some instability.

The model has given best results on `Epoch 20` if only seen validation set:
* Accuracy = 0.77
* Recall = 0.66
* Precison = 0.65
* F1 = 0.66

This experiment has given the best outputs and shows better stability in learning of model. It indicates the next model may be able to perform better on higher resolution:
* Very Small Input Size (64x64)

The `early_early_patience` is stopping the model a bit too fast may be in higher epochs the model may perform better. so for next experiment the early patience will be less stricter.

### Next Steps
* Increase Image/Input Size to (224 x 224)