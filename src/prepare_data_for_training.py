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

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    default="./data",
    help="Directory containing the dataset",
)
parser.add_argument(
    "--max_items",
    type=int,
    default=4,
    help="Max number of history items to use to predict the next item",
)


def make_dataset(data_file, max_items):
    df = pd.read_csv(data_file)

    print("Creating groups...")
    groups = dict()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["customer_id"] not in groups:
            groups[row["customer_id"]] = []

        groups[row["customer_id"]].append(
            (row["article_id"], row["price"], row["sales_channel_id"])
        )

    print("Creating chunks...")
    dataset = []
    for k, v in tqdm(groups.items(), total=len(groups)):
        for i in range(0, len(v), max_items):
            item = v[i : i + max_items]
            if len(v) > 2:
                dataset.append({k: item})

    return dataset


def main():
    args = parser.parse_args()

    print("Processing training data...")
    train_data = make_dataset(
        os.path.join(args.data_root, "train.csv"), args.max_items
    )

    print("Processing validation data...")
    val_data = make_dataset(
        os.path.join(args.data_root, "val.csv"), args.max_items
    )

    print("Processing test data...")
    test_data = make_dataset(
        os.path.join(args.data_root, "test.csv"), args.max_items
    )

    print("Saving data...")
    with open(os.path.join(args.data_root, "train.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(args.data_root, "val.json"), "w") as f:
        json.dump(val_data, f)
    with open(os.path.join(args.data_root, "test.json"), "w") as f:
        json.dump(test_data, f)


if __name__ == "__main__":
    main()
