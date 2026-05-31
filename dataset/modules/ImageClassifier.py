from PIL import Image
from pathlib import Path
from uuid import uuid4

def extract_image(
    input_directory,
    output_directory,
    includes=None,
    excludes=None,
    size=(224, 224)
):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # normalize filters once
    includes = includes.lower() if includes else None
    excludes = excludes.lower() if excludes else None

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

    for image_path in input_directory.rglob("*"):

        # skip non-image files
        if image_path.suffix.lower() not in valid_ext:
            continue

        image_name = image_path.name.lower()

        # include filter
        if includes and includes not in image_name:
            continue

        # exclude filter
        if excludes and excludes in image_name:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(size)

            filename = f"{image_path.stem}-{uuid4().hex}.jpg"
            image.save(output_directory / filename)
            print("Image saved!")

        except Exception as e:
            print(f"Skipping {image_path}: {e}")