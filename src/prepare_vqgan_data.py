# Copyright 2023 The LLM-Rec Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    default="./data",
    help="Directory containing the dataset",
)


def get_image_paths(articles_f, images_dir):
    with open(articles_f, "r") as f:
        articles = json.load(f)

    image_paths = []

    for article in articles:
        image_path = os.path.join(
            images_dir, f"0{str(article)[:2]}", f"0{article}.jpg"
        )
        if os.path.exists(image_path):
            image_paths.append(image_path)

    return image_paths


def make_splits(image_paths):
    image_paths.sort()
    random.shuffle(image_paths)

    dev_split = image_paths[: int(0.8 * len(image_paths))]
    train_split = dev_split[: int(0.8 * len(dev_split))]
    val_split = dev_split[int(0.8 * len(dev_split)) :]
    test_split = image_paths[int(0.8 * len(image_paths)) :]

    return train_split, val_split, test_split


def main():
    args = parser.parse_args()
    random.seed(42)

    articles_f = os.path.join(args.data_root, "articles.json")
    assert os.path.exists(articles_f), f"{articles_f} does not exist."
    assert os.path.isfile(articles_f), f"{articles_f} is not a file."

    images_dir = os.path.join(args.data_root, "images")
    assert os.path.exists(images_dir), f"{images_dir} does not exist."
    assert os.path.isdir(images_dir), f"{images_dir} is not a directory."

    image_paths = get_image_paths(articles_f, images_dir)

    train_split, val_split, test_split = make_splits(image_paths)

    print("Saving splits...")
    with open(os.path.join(args.data_root, "images_train.json"), "w") as f:
        json.dump(train_split, f)

    with open(os.path.join(args.data_root, "images_val.json"), "w") as f:
        json.dump(val_split, f)

    with open(os.path.join(args.data_root, "images_test.json"), "w") as f:
        json.dump(test_split, f)


if __name__ == "__main__":
    main()
