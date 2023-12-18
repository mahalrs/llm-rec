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

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from vqgan.model import VQModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    default="./data",
    help="Directory containing the dataset",
)
parser.add_argument(
    "--from_ckpt",
    default="./checkpoints/last.ckpt",
    help="Path to VQGAN checkpoint/pretrained model",
)
parser.add_argument(
    "--pretrained_lpips",
    default="./pretrained/vgg.pth",
    help="Path to pretrained LPIPS model",
)


class HMDatasetImages(Dataset):
    def __init__(self, data_dir, transform):
        self.image_paths = self.get_image_paths(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        article_id, image_path = self.image_paths[idx]

        try:
            img = Image.open(image_path).convert("RGB")
            return article_id, self.transform(img)
        except Exception as exc:
            print(exc)
            return None

    def get_image_paths(self, data_dir):
        articles_f = os.path.join(data_dir, "articles.json")
        images_dir = os.path.join(data_dir, "images")

        with open(articles_f, "r") as f:
            articles = json.load(f)

        image_paths = []
        for article in articles:
            image_path = os.path.join(
                images_dir, f"0{str(article)[:2]}", f"0{article}.jpg"
            )
            if os.path.exists(image_path):
                image_paths.append((article, image_path))

        print(f"Found {len(image_paths)} images")

        return image_paths


def encode_data(model, data_loader, device):
    torch.set_grad_enabled(False)
    model.eval()

    encoded_data = {}

    for ids, images in tqdm(data_loader, total=len(data_loader)):
        images = images.to(device)
        _, _, info = model.encode(images)
        tokens = info[2].reshape(-1, 256)
        tokens = tokens.cpu().numpy().tolist()

        for article_id, token in zip(ids, tokens):
            encoded_data[article_id] = token

    return encoded_data


def main():
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = HMDatasetImages(args.data_root, transform)
    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
    )

    hparams = {
        "n_embed": 16384,
        "embed_dim": 256,
        "learning_rate": 4.5e-06,
        "ddconfig": {
            "double_z": False,
            "z_channels": 256,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
        },
        "lossconfig": {
            "disc_conditional": False,
            "disc_in_channels": 3,
            "disc_start": 0,
            "disc_weight": 0.75,
            "disc_num_layers": 2,
            "codebook_weight": 1.0,
        },
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VQModel.load_from_checkpoint(args.from_ckpt, **hparams)
    model.init_lpips_from_pretrained(args.pretrained_lpips)
    model = model.to(device)

    encoded_data = encode_data(model, data_loader, device)

    print("Saving encoded data...")
    print(f"Encoded {len(encoded_data)} images")
    with open(os.path.join(args.data_root, "encoded_images.json"), "w") as f:
        json.dump(encoded_data, f)


if __name__ == "__main__":
    main()
