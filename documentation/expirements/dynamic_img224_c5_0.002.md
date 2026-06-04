# Enhanced Norm Model: 224x224 & 5 Conv Blocks
## Overview
This experiment uses the EnhancedNormModel architecture `/architectures/DynamicModel.py`. The architecture is documented briefly: `/documentation/architectures/dynamic_model.md`.

The process is similar as mentioned in `/documentation/expirements/en_norm_img224_c5_lr0.002.md`.

New changes include:
* New Architecture `DynamicModel` used
* decreased `initial_output_channels` to `12`
* removed pooling from the first conv block
* 1 `activation` using `tanh`
* 1 `activation` with `8` outputs

Overall the DynamicModel looks like:

``` python
model = DynamicModel(
    conv_layers=5, 
    normalizations = [1, 1, 1, 1, 1],
    poolings = [0, 1, 1, 1, 1],
    dropouts = [0, 0, 1, 0, 1],
    initial_output_channel=12, 
    initial_image_size=224,
    dropout_p = 0.3,
    activations=["tanh"],
    activation_outputs=[8]
    ).to("cuda")
```
The experiment has implemented strict early_stopping_patience of 8.

``` python
metrics = trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    epoch=60,
    lr=0.002,
    print_on=5,
    save_dir="../models/dynamic_img224_c5_lr_0.002_a_1/",
    save_checkpoints=1,
    checkpoint_name="dynamic_224x224_a1_train_0_",
    early_stop_patience = 10,
    min_delta = 0.001
)
```
This early stopping did a great job stopping the model that was not performing any better. The trainer stopped traing at epoch 30.

The experiment showed some fluctuations in the valid set.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch 30.

### Evaluation From Metrics
The visualized graph `/documentation/resources/dynamic_img224_c5_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows train line gradually drop till epoch 30. The valid line drops but with some epoch having high peaks.
* The accuracy graph shows good learning for training set. Here the valid also rises with train but has some extreme valleys and peaks
* The precision graph gradual learning for training set but slow for validation set reaching only upto `0.7`
* The recall graph shows gradual learning and reaching cieling for training set but unstable and peaks & valleys occur in validation set.
* The F1 graph shows gradual learning and reaching cieling for training set and slow for validation score with high instability.

The model has given best results on `Epoch 21` if only seen validation set:
* Accuracy = 0.78
* Recall = 0.665
* Precison = 0.668
* F1 = 0.666. 

Over all the model performance dropped slightly and is more unstable compared to previous experiment `/documentation/expirements/en_norm_img224_c5_lr0.002.md`

The valid curve still shows some instability.
### Next Steps
* Switch back to previous architecture
* set `monitor` to `valid_loss`
* Increasing `RandomErasing` in `transforms`
* Keep dropout to last conv block