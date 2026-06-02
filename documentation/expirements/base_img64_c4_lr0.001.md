# Base Model: 64x64 & 4 Conv Blocks
## Overview
This model uses the BaseModel architecture `/architectures/BaseModel.py`. The architecture is defined and documented briefly : `/documentation/architectures/base_model.md`. 
The experiment uses different modules including custom modules. Modules used in the experiment are as follows with their purpose:

### Existing Modules:
* <b>datasets: </b> For accesing data from root directory. Loads data using: 

    ``` python
    from torchvision import datasets

    datasets.ImageFolder(
        root = "dataset_folder_path",
        transform = transform_function 
    )
    ```
* <b>transforms: </b> For applying transformations such as:
    - Resize to 64x64
    - Convert to Tensors

    ``` python
    from torchvision import transforms

    transforms.Compose([
        transforms.Resize((64,64))
        transforms.ToTensor()
    ])
    ```
* <b>DataLoader: </b> For loading the data from dataset and making batches.
    - It uses batch_size as 32
    - It shuffels the data

    ``` python
    from torch.utils.data import DataLoader

    DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    ```
* <b>More modules for visualization</b>
    - `matplotlib.pyplot` as `plt`
    - `seaborn` as `sns`
    - `pandas` as `pd`

### Custom Imports
* <b>BaseModel: </b> It is the architecture for our model. This is used for model training. It is documented in `/documentation/architectures/base_model.md`. Used as:

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
        lr=0.001,
        print_on=10,
        save_dir="../models/base_64x64_4conv/",
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

The loss graph shows that the model learns quickly in the beginning, but after some epochs it becomes almost flat, meaning the model stops improving much. Train and validation loss are very close, so the model is not overfitting.

Accuracy shows that training is quite stable, while validation accuracy has small ups and downs because the validation dataset is small. Overall it slowly improves but not very strongly.

Precision and recall show that the model first struggles to correctly detect the rare class (high-alert). Recall starts very low but improves later, meaning the model slowly learns to identify those cases. F1 score stays low to medium, showing the balance between precision and recall is not very strong yet.

### Limitations
The main problem is class imbalance. High-alert has only 600 samples compared to ~5000 in other classes, so the model is biased toward normal and alert classes. This causes low recall early and unstable performance on validation. Also, the model reaches a learning limit early and stops improving much after that.

### Next Steps
* Increase learning rate (try 0.0075) to see if model can learn faster and escape early saturation.
* Oversample the high-alert class to make it closer to ~5000 samples so the model learns it better and reduces bias.