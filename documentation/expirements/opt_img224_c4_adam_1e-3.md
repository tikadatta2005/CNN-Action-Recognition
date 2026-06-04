# Norm Model: 64x64 & 4 Conv Blocks
## Overview
This experiment uses the OptimizedModel architecture `/architectures/OptimizedModel.py`. The architecture is defined and documented briefly : `/documentation/architectures/optimized_model.md`. 

The model has some new changes:
* Use of Optimizer Adam
* Learning rate = 1e-3 or 0.001
* 4 conv blocks
* initial output channel = 32 

### Flow of using modules:
* <b>transforms: </b> For applying transformations such as:
    - Resize to 224x224
    - Convert to Tensors
    - Adds random horizontal flips
    - Random Rotations upto 8 deg
    - Color jitter
    - Random Erasing upto 20%

    ``` python
    from torchvision import transforms

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3)
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
        oversample_rate = 1,
    )
    ```
* <b>DataLoader: </b> For loading the data from dataset and making batches.
    - It uses batch_size as 32
    - uses sample

    ``` python
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        sample, 
        batch_size=32, 
        shuffle=False,
        sample = sample
    )
    ```

* <b>ClassWeight</b> For computing class weight loss.
    
    ``` python
    class_weight = compute_class_weights(train_dataloader)
    ```

* <b>OptimizedModel: </b> It is the architecture for the model. This is used for model training. It is documented in `/documentation/architectures/optimized_model.md`. Used as:

    ``` python
    from architectures.OptimizedModel import OptimizedModel
    
    model = OptimizedModel(
    conv_layers=4,
    normalizations = [1,1,1,1],
    poolings = [1,1,1,1],
    dropouts = [0,1,0,1],
    initial_output_channel = 32,
    initial_image_size = 224,
    class_weights = class_weight,
    num_classes = 3,
    dropout_p = 0.2
    ).to("cuda")
    ```

* <b>Adam Optimizer: </b> The main optimizer function used for controlling learning rates. 

    ``` python
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    model.set_optimizer(optimizer)
    ```

* <b>trainer: </b> It is a custom function that accepts multiple parameters to train, validate and return metrics with loss. It's documentation is present at `/documentation/modules/TrainTest.md`. It's use case:

    ``` python
    metrics = trainer(
    model = model,
    train_dataloader = train_dataloader,
    test_dataloader = valid_dataloader,
    epoch = 100,
    lr = 1e-3,
    print_on=2,
    save_dir = "../models/opt_img224_c4_adam_1e-3_b32_v2",
    save_checkpoints=1,
    checkpoint_name = "opt_img224_c4_b32_",
    )
    ```
## Model Performance and Evaluation

As always the metric data is converted to `pandas DataFrame` with added `epoch` columns.

The tabular format of data is to used for uderstanding the behaviour of the model. Visualization techinique is also used for better gimplse of model's performance. The trainer functions  features allowed model training to stop after very less or no improvements. The training was stopped at epoch 20. But it was still showing possibilities of growing as visualizations showed much stable graphs and values then previous. So the model was restarted by disabling early stops. Evaluations after 100 epochs are as follows.

### Evaluations from Metrics
The visualized graph `/documentation/resources/opt_img224_c4_adam_1e-3.png` shows comparison of 
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