# Data Preparation

A dataset containing fight, crime, violence, weaponized situations, and normal activities is required for the Human Action Recognition system. Simple human postures such as sitting, standing, or walking were not considered as separate classes because the primary objective of this project is violence and crime detection, with the possibility of extending it to a broader surveillance alert system in future work.

To gather sufficient data, multiple publicly available datasets were combined. The datasets used in this project were collected from different Kaggle providers. The original sources are listed below:

* https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
* https://www.kaggle.com/datasets/mdusmanhanif/normal-vs-abnormal-action-analysis-datasets
* https://www.kaggle.com/datasets/adityapatil8668/fighting

Most of the collected datasets are distributed in video format, while a few are provided as images. Each source contains different classifications, formats, and directory structures. Therefore, only the necessary and most relevant data were extracted and organized into the `/dataset/raw_data` directory.

The raw dataset consists of the following categories:

* **violent:** Contains synthetic violence videos. Although the videos are generally clean and visually clear, they lack environmental diversity, location variation, and detailed classifications. The original dataset contains only violent and non-violent folders.
* **SCVD_converted:** A high-quality dataset containing multiple camera angles, locations, and real-world scenarios. However, it only provides three classes: Normal, Violence, and Weaponized. The dataset also contains predefined training and testing splits.
* **reallife-camera-dataset:** Contains real-world surveillance footage classified into Fight and NoFight categories.
* **real-life-violence-non-violence-dataset:** Another high-quality dataset containing Violence and Non-Violence classes.
* **name-framed:** A small custom dataset consisting of six videos. The files were renamed using timestamps indicating when violent actions begin and end. For example, a filename such as `20-40.mp4` indicates that only the segment between 20 and 40 seconds should be extracted for processing.
* **weapon-dataset:** An image dataset containing individuals carrying weapons. This dataset is intended to represent high-alert situations.

Although the raw data provided valuable information, several challenges were identified:

* CNN-based image classification models cannot process videos directly.
* Different datasets use inconsistent class labels and directory structures.
* Some datasets provide predefined train-test splits while others do not.
* Some sources contain videos whereas others contain images.
* The weapon dataset contains both normal and weaponized images within the same directory, requiring additional filtering based on file naming conventions.

To address these issues, a unified data preparation pipeline was developed. The original dataset labels were mapped into three common categories:

* **No Alert**
* **Alert**
* **High Alert**

Violence-related activities were mapped to the **Alert** class, weaponized activities were mapped to **High Alert**, and normal activities were mapped to **No Alert**.

Initially, videos were converted into image frames and stored in `/dataset/cleaned-dataset`. Each extracted frame was resized to **224 × 224 pixels** to maintain a consistent input size for model training.

To keep the preprocessing pipeline modular and reusable, the implementation was organized into separate modules located in `/dataset/modules`.

The following modules were used:

* `VideoExtractor.py` – Responsible for extracting frames from videos.
* `ImageClassifier.py` – Responsible for processing and organizing image datasets.

The `VideoExtractor.py` module provides an `extract_frames()` function that accepts the parameters `input_directory`, `output_directory`, `size`, and `VIDEO_EXTENSIONS`. The default image size is `(224, 224)`, and the module uses OpenCV to read videos and save extracted frames.

The `ImageClassifier.py` module provides an `extract_image()` function that accepts `input_directory`, `output_directory`, `size`, `includes`, and `excludes`. The default image size is `(224, 224)`, while both `includes` and `excludes` default to `None`. The module uses PIL to load, resize, and save images.

The preprocessing workflow utilizing these modules is implemented in `/dataset/clean.ipynb`.

During experimentation, an important issue was identified when creating training, validation, and testing datasets after frame extraction. Consecutive frames extracted from the same video are often highly similar. If frame-level splitting is performed after extraction, nearly identical frames from a single video may be distributed across the training, validation, and testing sets. This can introduce data leakage and lead to overly optimistic evaluation results.

To address this problem, dataset splitting was performed at the **video level before frame extraction**. Videos were first divided into training, validation, and testing subsets. Frames were then extracted separately from each split using the existing preprocessing modules. This approach ensures that frames originating from the same video remain within a single dataset partition, resulting in a more realistic and reliable evaluation process.

The final dataset structure generated by the module contains:

* `train`
* `valid`
* `test`

The resulting `/dataset/final-dataset` directory is used throughout all experiments conducted in this research-oriented project.
