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

import json
import os

from torch.utils.data import Dataset


class HMDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer):
        self.tokenizer = tokenizer
        self._load(data_dir, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        customer_id = list(self.data[idx].keys())[0]
        items = self.data[idx][customer_id]

        inp = f"age: {self.customers[customer_id][1]} postal_code: {self.customers[customer_id][2]}"

        for item in items[:-1]:
            inp += " "
            inp += self._make_inp(item, self.articles[f"{item[0]}"])

        label = self._make_inp(items[-1], self.articles[f"{items[-1][0]}"])

        inputs = self.tokenizer(
            inp,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            label,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].view(-1),
            "attention_mask": inputs["attention_mask"].view(-1),
            "labels": labels["input_ids"].view(-1),
        }

    def _make_inp(self, item, article):
        return f"price: {item[1]} sales_channel: {item[2]} product_code: {article['product_code']} product_type_no: {article['product_type_no']} product_group_name: {article['product_group_name']} color_group_code: {article['colour_group_code']} department_no: {article['department_no']} index_code: {article['index_code']} index_group_no: {article['index_group_no']} section_no: {article['section_no']} garment_group_no {article['garment_group_no']}"

    def _load(self, data_dir, split):
        with open(os.path.join(data_dir, "articles.json"), "r") as f:
            self.articles = json.load(f)

        with open(os.path.join(data_dir, "customers.json"), "r") as f:
            self.customers = json.load(f)

        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            self.data = json.load(f)
