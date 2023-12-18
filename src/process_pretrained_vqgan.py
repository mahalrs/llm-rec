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

import lightning.pytorch as pl  # Do not remove this line. It is needed for loading the checkpoint.
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_vqgan",
    default="./pretrained/vqgan.ckpt",
    help="Path to pretrained VQGAN model",
)


def main():
    args = parser.parse_args()
    # Load checkpoint
    assert os.path.exists(
        args.pretrained_vqgan
    ), f"{args.pretrained_vqgan} does not exist."
    assert os.path.isfile(
        args.pretrained_vqgan
    ), f"{args.pretrained_vqgan} is not a file."

    ckpt = torch.load(args.pretrained_vqgan, map_location="cpu")
    ckpt["epoch"] = 0
    if "callbacks" in ckpt:
        del ckpt["callbacks"]
    # Save checkpoint
    torch.save(ckpt, args.pretrained_vqgan)


if __name__ == "__main__":
    main()
