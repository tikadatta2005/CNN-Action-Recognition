from pathlib import Path
from shutil import copy2
import random


def split_dataset(
    input_directory,
    output_directory,
    train_ratio=0.70,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    # set random seed
    random.seed(seed)

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    # process each class folder
    for class_directory in input_directory.iterdir():

        if not class_directory.is_dir():
            continue

        # collect images
        images = [
            path
            for path in class_directory.iterdir()
            if path.is_file()
        ]

        random.shuffle(images)

        total = len(images)

        train_size = int(total * train_ratio)
        valid_size = int(total * valid_ratio)

        # split images
        train_images = images[:train_size]
        valid_images = images[train_size:train_size + valid_size]
        test_images = images[train_size + valid_size:]

        splits = {
            "train": train_images,
            "valid": valid_images,
            "test": test_images
        }

        # create folders and copy files
        for split_name, files in splits.items():

            destination = (
                output_directory /
                split_name /
                class_directory.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for file in files:
                copy2(
                    file,
                    destination / file.name
                )