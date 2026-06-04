# Enhanced Norm Model: 224x224 & 5 Conv Blocks
## Overview
This experiment uses the `WeightLossModel` architecture `/architectures/WeightLossModel.py`. The architecture is documented briefly: `/documentation/architectures/weight_loss_model.md`.

The model accepts addional parameter `class_weights` on top of `EnhancedNormModel`. It does not prefer using `sampler`.

Overall the WeightLossModel looks like:

``` python
model = WeightLossModel(
    conv_layers=5, 
    normalizations = [1, 1, 1, 1, 1],
    poolings = [1, 1, 1, 1, 1],
    dropouts = [0, 0, 1, 0, 1],
    initial_output_channel=16, 
    initial_image_size=224,
    dropout_p = 0.2,
    class_weights = class_weights
    ).to("cuda")
```
The experiment has implemented less strict early_stopping_patience of 20.

``` python
metrics = trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    epoch=60,
    lr=0.002,
    print_on=5,
    save_dir="../models/wl_img224_lr_0.002/",
    save_checkpoints=1,
    checkpoint_name="wl_224x224_train_0_",
    early_stop_patience = 20,
    min_delta = 0.001
)
```
This early stopping did a great job stopping the model that was not performing any better. The trainer stopped traing at epoch 23.

The experiment show instability in validation throughout the epochs.

As always the model returns a metric data which is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. 

### Evaluation From Metrics
The visualized graph `/documentation/resources/en_norm_img224_c5_lr0.002.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* Training loss decreases smoothly, but validation loss is highly unstable with large spikes → clear overfitting
* Training accuracy steadily improves, while validation accuracy is noisy and inconsistent → poor generalization
* Training precision increases steadily, validation precision fluctuates with only slight improvement → unstable class performance
* Training recall improves smoothly, but validation recall is very unstable → minority classes not handled well
* Training F1 improves steadily, but validation F1 is highly unstable → overall weak validation balance

This run is not good. The model is clearly overfitting and failing to generalize on validation data.

The previous experiment:
`/documentation/experiment/en_norm_img224_c5_lr0.002.md`
was more stable and better performing overall, so it still stands as the better model.

### Next Steps
* Add WeightedRandomSampler (1–2× strength) to fix class imbalance in batching
* Introduce learning rate decay One Cycle Lr