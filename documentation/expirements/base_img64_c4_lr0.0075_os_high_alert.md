# Base Model: 64x64 & 4 Conv Blocks
## Overview
This model uses the BaseModel architecture `/architectures/BaseModel.py`. The architecture is defined and documented briefly : `/documentation/architectures/base_model.md`. 

The process is almost same as in `/documentation/expirements/base_img64_c4_lr0.001.md`. But due to it's limitations:
* Slow Learining
* Class imbalance

The expirement has increased learning rate:
``` math
0.0075
```
and additional module `oversampler` was developed and used. It's documentation `/documentation/modules/OverSampler.md` gives the idea of how it is used.

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
        oversample_rate = 5
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
* <b>BaseModel: </b> It is the architecture for the model. This is used for model training. It is documented in `/documentation/architectures/base_model.md`. Used as:

    ``` python
    from architectures.BaseModel import BaseModel
    
    model = BaseModel(
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
        save_checkpoints=15
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
    "validation_loss": float,
    "validation_accuracy": float,
    "validation_precision": float,
    "validation_recall": float,
    "validation_f1": float
}
```

This data is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance.

### Evaluations from Metrics
The visualized graph `/documentation/resources/base_img64_c4_lr0.001.png` shows comparison of 
* loss
* accuracy
* precision
* recall
* f1
of both training and validation on each epoch.

<b>Breif Insights from visualizations</b>