# Data Prepearation

The dataset of fight, crime, voilence and normal situation is required for human action Recognition system. Simple postures like sit, stand is not selected in this dataset as this project intends to be used as voilence detection, crime detection in future model.

For the such data, multiple sources were used for gathering. The data used in this project is gathered from different kaggle providers. Following are the direct kaggle reference of the dataset:
* https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset
* https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
* https://www.kaggle.com/datasets/mdusmanhanif/normal-vs-abnormal-action-analysis-datasets
* https://www.kaggle.com/datasets/tathagatbanerjee/ucf-violent-activity-frames-npy
* https://www.kaggle.com/datasets/adityapatil8668/fighting
* https://www.kaggle.com/datasets/turkikhalidalshuaibi/haris-weapon-detection-dataset-curated

Most of the dataset here are in video format. Each source has multiple type of voilence and non-voilence dataset. Some of the source provide training and testing set while some have not labeled.

Only necessary and most relevant data is extracted and stored in `/datasets/raw_data` folder. This folder contains 5 types of data:
* <b>voilent: </b> This folder has videos of synthetic data of some voilence. Though this data has the most clean and clear videos, it lacks location and human variation. It also lacks proper classifications. It only has a folder that says voilent and non voilent. Out of which non-voilent dataset is not included. 
* <b>SCVD_converted:</b> This is a very good dataset that has different angles, location and real-world making it one of the perfect dataset, but it still has some problems such as file type. Also the dataset's classification is not good. It only classifies as Normal, Voilence, Weaponized. This data contains Train and Test Splits.
* <b>reallife-camera-dataset:</b> This is another good dataset. But it still lacks the classifications. It only has two classification fight and noFight.
* <b>real-life-voilence-non-voilence-dataset:</b> This is another outstanding dataset but this still does have only two classes which is voilence and non-voilence.
* <b>name-framed:</b> This dataset is a very tiny dataset which includes only 6 videos which is of 40s average. This dataset was customized by renaming it with the exact time of when voilence or assult starts and ends. This time "20-40.mp4" this time will allow video to be splitted from 20 to 40 and frames to be extracted from this exact time frame.
* <b>weapon-dataset:</b> This is the image dataset that contains multiple images of armed person. This will be used for classifying high alert.

A raw data was ready so the data must be giving valuable information. But there are two major flaw:
* CNN doesnot understand videos directly. 
* There is in-consitent classifcations.
* Some contain train-test split while other dont.
* Some directory contains video where some contains images
* The data at `/dataset/raw_data/weaponized/` includes images of both normal and weaponized in same directory, but includes "normal" in the file name.

The optimal solutions for flaw:
* Extract Frames from video and store images.
* Classify voilence into alert, weaponized into high-alert and normal into no-alert.
* Blend the existing train-test to a single dataset then later use the final dataset for extracting 80-20 Train-Test split.
* Use different techniques for extracting data from videos and images
* Check filename and decide to classify the image

The frames extracted from the videos are stored in `/dataset/cleaned-dataset/..`. Each frame is of (224x224) size. This was done to meet the normal standard.

There are multple functions used in data cleaning. To structure the functions cleanly, each function is made as a reusable-module inside `/dataset/modules`. Following modules will be used:
* `VideoExtractor.py` for extracting frames
* `ImageClassifier.py` for classifying images

`VideoExtractor.py` module provides a function `extract_frames` that accepts parameters: input_directory, output_directory, size and VIDEO_EXTENSIONS. The default value for size is (224,224) and VIDEO_EXTENSIONS is {".mp4", ".av1"}. This module uses opencv-python for extracting frames from video and save in output_directory.

`ImageClassifier.py` module provides a function `extract_image` that accepts input_directory, output_directory, size, includes, and excludes. The default value for size is (224, 224), includes and excludes is None. This module use PIL.Image for opening, resizing and saving in output directory.

The code using both modules is in `/dataset/clean.ipynb`. 