# Norm Model: 64x64 & 4 Conv Blocks
## Overview
This experiment uses the BaseModel architecture `/architectures/BaseModel.py`. The architecture is defined and documented briefly : `/documentation/architectures/base_model.md`. 

The process is almost same as in `/documentation/expirements/base_img64_c4_lr0.001.md`. But due to it's limitations:
* Slow Learining
* Class imbalance

The expirement has decreased learning rate:
``` math
0.005
```

### Flow of using modules:
* <b>transforms: </b> For applying transformations such as:
    - Resize to 64x64
    - Convert to Tensors

    ``` python
    from torchvision import transforms

    transform_function = transforms.Compose([
        transforms.Resize((64,64))
        transforms.ToTensor()
    ])
    ```
* <b>datasets: </b> For accesing data from root directory. Loads data using: 

    ``` python
    from torchvision import datasets

    dataset = datasets.ImageFolder(
        root = "dataset_folder_path",
        transform = transform_function 
    )
    ```

* <b>oversampler: </b> For oversampling the data.

    ``` python
    from modules.OverSampler import oversampler

    sample = oversampler(
        dataset = dataset,
        oversample_rate = 5,
        early_stop_patience = 5,
    )
    ```
* <b>DataLoader: </b> For loading the data from dataset and making batches.
    - It uses batch_size as 32
    - It shuffels the data

    ``` python
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        sample, 
        batch_size=32, 
        shuffle=False
    )
    ```
* <b>More modules for visualization</b>
    - `matplotlib.pyplot` as `plt`
    - `seaborn` as `sns`
    - `pandas` as `pd`

### Custom Imports
* <b>NormModel: </b> It is the architecture for the model. This is used for model training. It is documented in `/documentation/architectures/norm_model.md`. Used as:

    ``` python
    from architectures.NormModel import NormModel
    
    model = NormModel(
        conv_layers=4, 
        initial_output_channel=8, 
        initial_image_size=64
        ).to("cuda")
    ```
* <b>trainer: </b> It is a custom function that accepts multiple parameters to train, validate and return metrics with loss. It's documentation is present at `/documentation/modules/TrainTest.md`. It's use case:

    ``` python
    trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=valid_dataloader,
        epoch=100,
        lr=0.0075,
        print_on=10,
        save_dir="../models/base_64x64_4conv_0.0075_os/",
        save_checkpoints=15,
        # new updated parameters
        min_delta = 0.01,
        monitor = "valid_loss" # Default value
    )
    ```
## Model Performance and Evaluation
### Train and Valid Comparison
The `trainer(...)` module returns an array of metrics in array dict format storing following for each epoch:
``` python
{
    "training_loss": float,
    "training_accuracy": float,
    "training_precision": float,
    "training_recall": float,
    "training_f1": float,
    "valid_loss": float,
    "valid_accuracy": float,
    "valid_precision": float,
    "valid_recall": float,
    "valid_f1": float
}
```

This data is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions new features allowed model training to stop after very less or no improvements. The training was stopped at epoch seven.

### Evaluations from Metrics
The visualized graph `/documentation/resources/norm_do_img64_c4_lr0.005_os.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>

* The loss graph shows smooth and stable learning for the training but the validation loss shows very less imporvements with bumpy lines causing the training to break in middle.
* The accuracy graph shows the training line gradually increasing to 0.80 and higher. But the valid line gradually increases till around 0.65 and stay almost constant. The valid line is not stable and has some small bumps.
* Precision graph shows train climbing fast then slows gradually reaching above 0.80 but the validation graph is barely moving upwards reaching steadly till 0.6.
* Recall graph shows train line climbing fast then slows gradually reaching above 0.80. On the other hand valid line barely moves remaining almost constant then slightly move even lower.
* F1 graphj shows training line almost same as recall and precision. But the valid line is very slowly moving upwards till 0.55 in 3 epochs and constant

### Limitations
* <b>Overfitting:</b> Validation loss decrease is very slow and eventually tends on growing while training loss continues decreasing.
* <b>Validation saturation:</b> Validation metrics improve early but then gradually flatten, showing limited further gains.
* <b>Train–validation divergence:</b> Training metrics keep improving while validation improvements slow significantly.
* <b>Generalization limit:</b> The model reaches a performance ceiling on validation data despite continued training improvements.

### Possible Problems
* Heavy Oversampling
* Heavy Dropouts
* Less Data Augmentation
* Too small Input size of 64x64

### Next Steps
* Decrease Oversampling
* Decrease Learning Rate to 0.002
* Update Architecture for Custom Dropouts after each layer.
* Multiple Transformations for Data Augmentation using `transforms` module

Not increasing input size for immediate next experiment to test if the new experiment reveals something.