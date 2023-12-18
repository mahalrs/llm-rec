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
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HMDatasetImages
from vqgan.model import VQModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Directory containing the dataset",
)
parser.add_argument(
    "--log_dir", default="./logs", help="Directory to save logs"
)
parser.add_argument(
    "--ckpt_dir",
    default="./checkpoints",
    help="Directory to save checkpoints",
)
parser.add_argument(
    "--from_ckpt",
    default="./pretrained/vqgan.ckpt",
    help="Path to VQGAN checkpoint/pretrained model",
)
parser.add_argument(
    "--pretrained_lpips",
    default="./pretrained/vgg.pth",
    help="Path to pretrained LPIPS model",
)
parser.add_argument(
    "--train_batch",
    default=6,
    type=int,
    help="Train batch size",
)
parser.add_argument(
    "--val_batch",
    default=6,
    type=int,
    help="Validation batch size",
)
parser.add_argument(
    "--test_batch",
    default=6,
    type=int,
    help="Test batch size",
)
parser.add_argument(
    "--num_workers",
    default=8,
    type=int,
    help="Number of dataloader workers to use",
)
parser.add_argument(
    "--epochs", default=12, type=int, help="Number of epochs to train"
)
parser.add_argument(
    "--val_every_n_steps",
    default=500,
    type=int,
    help="Perform validation every n steps and save checkpoint",
)
parser.add_argument(
    "--log_every_n_steps",
    default=500,
    type=int,
    help="Log every n steps",
)
parser.add_argument(
    "--seed",
    default=123,
    type=int,
    help="Random seed to use",
)


def main():
    args = parser.parse_args()
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    # Define data processing and augmentation
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # Load dataset
    assert os.path.exists(args.data_dir), f"{args.data_dir} does not exist."
    assert os.path.isdir(args.data_dir), f"{args.data_dir} is not a directory."

    train_set = HMDatasetImages(args.data_dir, "train", transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_set = HMDatasetImages(args.data_dir, "val", transform)
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_set = HMDatasetImages(args.data_dir, "test", transform)
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # Define logger and checkpoint callback
    logger = TensorBoardLogger(args.log_dir, name="vqgan")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="{epoch:02d}-{step:06d}",
        save_last=True,
        save_top_k=-1,
    )
    # Load model
    assert os.path.exists(
        args.pretrained_lpips
    ), f"{args.pretrained_lpips} does not exist."
    assert os.path.isfile(
        args.pretrained_lpips
    ), f"{args.pretrained_lpips} is not a file."

    model = VQModel(**hparams)
    model.init_lpips_from_pretrained(args.pretrained_lpips)
    # Load trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        devices="auto",
        val_check_interval=args.val_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        callbacks=[ckpt_callback],
    )
    # Train model
    assert os.path.exists(args.from_ckpt), f"{args.from_ckpt} does not exist."
    assert os.path.isfile(args.from_ckpt), f"{args.from_ckpt} is not a file."

    trainer.fit(
        model,
        ckpt_path=args.from_ckpt,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    # Test model
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        num_nodes=1,
        logger=logger,
    )

    trainer.test(model, dataloaders=test_loader)
    # Save model
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "last.ckpt"))


if __name__ == "__main__":
    main()
