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

* Training loss decreases smoothly throughout training, while validation loss improves initially before becoming noisy and gradually increasing after around 10 epochs, indicating moderate overfitting as the model continues fitting the training data more than the validation data.
* Training accuracy steadily rises to approximately 96%, whereas validation accuracy improves rapidly during the first 10–15 epochs before plateauing around 78–81%, showing that most generalizable learning occurs early in training.
*Training precision consistently increases to about 95%, while validation precision stabilizes between 65–73%, indicating that the model becomes more confident in its predictions while maintaining reasonable precision on unseen data.
* Training recall gradually improves to around 95%, while validation recall remains relatively stable between 63–68%, suggesting that the model captures a reasonable proportion of samples from each class but still struggles with some challenging examples.
* Training F1 score steadily increases to approximately 95%, while validation F1 quickly improves and then plateaus around 65–69%, with a best value of roughly 0.687, indicating the strongest balance between precision and recall achieved among recent experiments despite dataset imbalance.

### Limitations
* <b>Moderate Overfitting:</b> Training metrics continue improving while validation metrics plateau.
* <b>Validation Loss Instability:</b> Several spikes in validation loss suggest occasional overconfident predictions.
* <b>Limited Recall:</b> Validation recall remains around 65–68%, meaning some samples are still missed.
* <b>Performance Plateau:</b> Metrics stop improving significantly after roughly 20–30 epochs.
* <b>Class Imblance Sensitivity: </b> Despite improvements, minority classes may still be harder to classify accurately than majority classes.
* <b>Generalization Gap Remains: </b>Approximately 14–15% gap between training and validation performance still exists.

The best-performing checkpoint of the final model was obtained at `Epoch 41`, where the model achieved
``` text 
validation:
accuracy = 81.49%, 
precision = 73.48%, 
recall = 66.57%, 
F1-score = 68.69%.
```
At this point, the training metrics were 
``` text
94.78% accuracy, 
94.77% precision, 
94.78% recall, 
94.77% F1-score
```
indicating that the model had learned strong feature representations while maintaining reasonable generalization to unseen data. Although a train-validation gap remains, this epoch provided the best balance between precision and recall across classes, resulting in the highest validation F1-score observed during experimentation. Compared to previous models, this checkpoint demonstrated improved class-balanced performance and reduced overfitting, making it the selected final model.

The experiment may continue with another data later on.