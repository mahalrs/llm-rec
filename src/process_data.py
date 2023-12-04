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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root", default="../data", help="Directory containing the dataset"
)


def process_articles(articles_f, images_dir):
    articles = pd.read_csv(articles_f)
    articles.dropna(inplace=True)

    def filter_fn(row):
        article_id = row["article_id"]
        path = os.path.join(
            images_dir, f"0{str(article_id)[:2]}", f"0{article_id}.jpg"
        )
        return os.path.exists(path)

    articles = articles[articles.apply(filter_fn, axis=1)]

    articles_dict = dict()
    for _, row in articles.iterrows():
        articles_dict[row["article_id"]] = row.drop(
            labels=["article_id"]
        ).to_dict()

    return articles_dict


def process_customers(customers_f):
    customers = pd.read_csv(customers_f)

    customers.drop(columns=["FN", "Active", "club_member_status"], inplace=True)
    customers.dropna(inplace=True)

    customers["fashion_news_frequency"] = customers[
        "fashion_news_frequency"
    ].map({"NONE": 0, "Regularly": 1})

    map_postal_code = dict()
    count = 0
    for code in customers["postal_code"]:
        if code not in map_postal_code:
            map_postal_code[code] = count
            count += 1

    customers["postal_code"] = customers["postal_code"].map(map_postal_code)

    customers_dict = dict()
    for _, row in customers.iterrows():
        customers_dict[row["customer_id"]] = [
            row["fashion_news_frequency"],
            row["age"],
            row["postal_code"],
        ]

    return customers_dict


def process_transactions(transactions_f, articles_dict, customers_dict):
    transactions = pd.read_csv(transactions_f)

    def filter_fn(row):
        if row["article_id"] not in articles_dict:
            return False

        if row["customer_id"] not in customers_dict:
            return False

        return True

    transactions = transactions[transactions.apply(filter_fn, axis=1)]
    transactions.sort_values(by=["t_dat"], inplace=True)
    transactions.reset_index(drop=True, inplace=True)

    return transactions


def create_splits(transactions_df):
    dev_df = transactions_df[transactions_df["t_dat"] < "2020-09-07"]
    dev_df.reset_index(drop=True, inplace=True)

    train_df = dev_df[dev_df["t_dat"] < "2020-08-23"]
    train_df.reset_index(drop=True, inplace=True)

    val_df = dev_df[dev_df["t_dat"] >= "2020-08-23"]
    val_df.reset_index(drop=True, inplace=True)

    test_df = transactions_df[transactions_df["t_dat"] >= "2020-09-07"]
    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def main():
    args = parser.parse_args()

    articles_f = os.path.join(args.data_root, "articles.csv")
    assert os.path.exists(articles_f), f"{articles_f} does not exist."
    assert os.path.isfile(articles_f), f"{articles_f} is not a file."

    customers_f = os.path.join(args.data_root, "customers.csv")
    assert os.path.exists(customers_f), f"{customers_f} does not exist."
    assert os.path.isfile(customers_f), f"{customers_f} is not a file."

    transactions_f = os.path.join(args.data_root, "transactions_train.csv")
    assert os.path.exists(transactions_f), f"{transactions_f} does not exist."
    assert os.path.isfile(transactions_f), f"{transactions_f} is not a file."

    images_dir = os.path.join(args.data_root, "images")
    assert os.path.exists(images_dir), f"{images_dir} does not exist."
    assert os.path.isdir(images_dir), f"{images_dir} is not a directory."

    print("Processing articles...")
    articles_dict = process_articles(articles_f, images_dir)

    print("Processing customers...")
    customers_dict = process_customers(customers_f)

    print("Processing transactions...")
    transactions_df = process_transactions(
        transactions_f, articles_dict, customers_dict
    )

    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(transactions_df)

    print("Saving processed data...")
    with open(os.path.join(args.data_root, "articles.json"), "w") as f:
        json.dump(articles_dict, f)

    with open(os.path.join(args.data_root, "customers.json"), "w") as f:
        json.dump(customers_dict, f)

    train_df.to_csv(os.path.join(args.data_root, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.data_root, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.data_root, "test.csv"), index=False)


if __name__ == "__main__":
    main()
