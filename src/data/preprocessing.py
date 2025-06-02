"""Contains functions and helpers for dataset pre-processing
"""
import shutil
import json
import random
import os
from pathlib import Path

import tensorflow as tf

DATA_DESCRIPTOR_FILE = "orinoquia_camera_traps.json"
DATA_SET_DIR = "/scratch/CS4232_wildlife_classification_project/wildlife_classification/src/cache/datasets/"


def _preprocess_data(
    min_count: int, max_count: int, dataset: Path, output_file: str = "labeled-data.json"
) -> Path:
    """Pre-processes and labels the images with their respective category names

    Parameters
    ----------
    min_count : int
        the minimum amount of the category required for model training
    output_file : str, optional
        the name of the json file to output the labeled data to, by default 'labeled-data.json'

    Returns
    -------
    Path
        path to the data output file

    Raises
    ------
    FileNotFoundError
        if the dataset descriptor json file is not found
    """
    data_set_descriptor = Path(DATA_SET_DIR).joinpath('orinoquia_dataset').joinpath(DATA_DESCRIPTOR_FILE)
    if not data_set_descriptor.exists():
        raise FileNotFoundError(
            f"Could not find the json dataset descriptor file on {str(data_set_descriptor)}"
        )
    descriptor_data = {}
    labeled_data = {}
    with data_set_descriptor.open("r") as infile:
        descriptor_data = json.load(infile)

    def category_filter(x) -> bool:
        return x.get('count') > min_count and 'human' not in x.get('name')

    filtered_categories = map(
        lambda x: {"id": x.get("id"), "name": x.get("name")},
        [x for x in descriptor_data["categories"] if category_filter(x)],
    )
    for category in filtered_categories:
        labeled_images = list(
            map(
                lambda x: x.get("image_id").replace("_", "/"),
                [
                    x
                    for x in descriptor_data["annotations"]
                    if x.get("category_id") == category["id"]
                ]
            )
        )
        labeled_images = [
            x
            for x in labeled_images
            if dataset.joinpath(x).exists()
        ]
        random.shuffle(labeled_images)
        if len(labeled_images) > max_count:
            labeled_images = labeled_images[:max_count]
        labeled_data[category["name"]] = labeled_images
    # write the labeled data to the json output
    output_file = Path(__file__).parent.joinpath(output_file)
    with output_file.open('w') as outfile:
        outfile.write(json.dumps(labeled_data, indent=2))
    return output_file


def _make_preprocessed_directory(
    dataset_parent: Path, labeled_file: Path, name: str
) -> None:
    os.umask(0)
    orinoquia_set = dataset_parent.joinpath("orinoquia_dataset")
    if not orinoquia_set.exists():
        raise FileNotFoundError(
            f"Could not find orinoquia dataset on path: {orinoquia_set}"
        )
    dataset_dir = dataset_parent.joinpath(name)
    # we should clear this when we re-run the data shuffling
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(mode = 0o770)
    with labeled_file.open("r") as infile:
        data = json.load(infile)
        for category, images in data.items():
            print(category)
            category_dir = dataset_dir.joinpath(category)
            if category_dir.exists():
                shutil.rmtree(category_dir)
            category_dir.mkdir(mode = 0o770)
            for image_path in images:
                shutil.copy(
                    orinoquia_set.joinpath("public").joinpath(image_path), category_dir
                )


def preprocess_image_data(
    dataset_directory: Path,
    *,
    preprocessed_data_name="preprocessed_data",
    data_name="orinoquia_dataset/public",
    min_category_img_count=1000,
    max_category_img_count=1000,
    validation_split=0.3
) -> tuple:
    preprocessed_data_dir = dataset_directory.joinpath(preprocessed_data_name)
    unprocessed_data = dataset_directory.joinpath(data_name)
    labeled_file = _preprocess_data(min_category_img_count, max_category_img_count, unprocessed_data)
    _make_preprocessed_directory(
        dataset_directory, labeled_file, name=preprocessed_data_name
    )
    return tf.keras.utils.image_dataset_from_directory(
        str(preprocessed_data_dir),
        shuffle=True,
        seed=42,
        validation_split=validation_split,
        subset="both"
    )


if __name__ == "__main__":
    print(preprocess_image_data(Path(DATA_SET_DIR)))
