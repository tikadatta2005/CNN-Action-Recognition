from pathlib import Path
from shutil import copy2
import random


class DatasetSplitter:

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    VIDEO_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm"
    }

    def __init__(self, seed=42):
        self.seed = seed

    # --------------------
    # Public Methods
    # --------------------

    def split_images(
        self,
        input_directory,
        output_directory,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15
    ):
        self._split(
            input_directory=input_directory,
            output_directory=output_directory,
            extensions=self.IMAGE_EXTENSIONS,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )

    def split_videos(
        self,
        input_directory,
        output_directory,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15
    ):
        self._split(
            input_directory=input_directory,
            output_directory=output_directory,
            extensions=self.VIDEO_EXTENSIONS,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )

    # --------------------
    # Private Methods
    # --------------------

    def _split(
        self,
        input_directory,
        output_directory,
        extensions,
        train_ratio,
        valid_ratio,
        test_ratio
    ):
        random.seed(self.seed)

        input_directory = Path(input_directory)
        output_directory = Path(output_directory)

        for class_directory in input_directory.iterdir():

            if not class_directory.is_dir():
                continue

            files = self._collect_files(
                class_directory,
                extensions
            )

            train_files, valid_files, test_files = (
                self._create_splits(
                    files,
                    train_ratio,
                    valid_ratio,
                    test_ratio
                )
            )

            self._copy_split(
                train_files,
                output_directory,
                "train",
                class_directory.name
            )

            self._copy_split(
                valid_files,
                output_directory,
                "valid",
                class_directory.name
            )

            self._copy_split(
                test_files,
                output_directory,
                "test",
                class_directory.name
            )

    def _collect_files(
        self,
        directory,
        extensions
    ):
        files = [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in extensions
            )
        ]

        random.shuffle(files)

        return files

    def _create_splits(
        self,
        files,
        train_ratio,
        valid_ratio,
        test_ratio
    ):
        total = len(files)

        train_size = int(total * train_ratio)
        valid_size = int(total * valid_ratio)

        train_files = files[:train_size]

        valid_files = files[
            train_size:
            train_size + valid_size
        ]

        test_files = files[
            train_size + valid_size:
        ]

        return (
            train_files,
            valid_files,
            test_files
        )

    def _copy_split(
        self,
        files,
        output_directory,
        split_name,
        class_name
    ):
        destination = (
            Path(output_directory)
            / split_name
            / class_name
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