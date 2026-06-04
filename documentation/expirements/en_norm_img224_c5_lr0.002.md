# Enhanced Norm Model: 224x224 & 5 Conv Blocks
## Overview
This experiment uses the EnhancedNormModel architecture `/architectures/EnhancedNormModel.py`. The architecture is documented briefly: `/documentation/architectures/en_norm_model.md`.

The process is almost same as mentioned in `/documentation/expirements/en_norm_img128_c5_lr0.002.md`.

The only change made was increasing input size from 128x128 to 224x224.

Overall the EnhancedNormModel looks like:

``` python
model = EnhancedNormModel(
    conv_layers=4, 
    normalizations = [1, 1, 1, 1],
    poolings = [1,1,1,1],
    dropouts = [0, 0, 1, 0, 1],
    initial_output_channel=16, 
    initial_image_size=224,
    dropout_p = 0.2
    ).to("cuda")
```
The experiment has implemented less strict early_stopping_patience of 15.

``` python
metrics = trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    epoch=30,
    lr=0.002,
    print_on=1,
    save_dir="../models/en_norm_img224_c5_lr_0.002_os/",
    save_checkpoints=1,
    checkpoint_name="en_norm_224x224_train_0_",
    early_stop_patience = 15,
    min_delta = 0.001
)
```
This early stopping did a great job stopping the model that was not performing any better. The trainer stopped traing at epoch 30.

The experiment very much outperformed previous experiments just because of change in input size changes.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch seven.

### Evaluation From Metrics
The visualized graph `/documentation/resources/en_norm_img224_c5_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows train line falling fast till 2 epochs then gradually drop till epoch 30. The valid line rises very high reaching `2.0` then suddenly drops to `1.0` then gradually lowers till 0.65 with some bumps and remain there.
* The accuracy graph shows very fast learning for training set. Here the valid also rises with train making outcome better even though there are some instabilities.
* The precision graph gradual learning for training set but slow for validation set reaching only upto `0.7`
* The recall graph shows gradual learning and reaching cieling for training set but unstable and hills & plateus for validation set.
* The F1 graph shows gradual learning and reaching cieling for training set and slow for validation score with some instability only going upto 0.65.

The model has given best results on `Epoch 21` if only seen validation set:
* Accuracy = 0.847
* Recall = 0.678
* Precison = 0.699
* F1 = 0.654

This experiment has given the best outputs and shows better stability in learning of model. 

Over all the model has performed pretty good with 224x224 input size. Looking at the table of metrics, the data shows model is properly identifying features.

The valid curve still shows some instability.
### Next Steps
* Change in architecture to accept dynamic Activation layers