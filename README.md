# Multimodal Product Recommendations via LLMs

This repository contains source code for various experiments to explore the feasibility of using a multi-modal approach to generate personalized product recommendations using LLMs. The model is trained on a dataset of customer purchase history with detailed metadata of purchased items with corresponding images.

## Install dependencies

```sh
pip install -r requirements.txt
```

## Download Data

Our project uses H&M Personalized Fashion Recommendations dataset, released as part of the Kaggle Competition by H&M. The dataset consists of customer purchase history over time with detailed metadata of purchased items, product images, and customer demographics. To read more, visit
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Download the dataset as follows:

Requirements:

- 100 GB free disk space
- 8 CPUs with 32 GB RAM

Before downloading, setup Kaggle API Credentials. Navigate to the Accounts page of Kaggle and go to the "API" section and select the "Create New API Token". This will trigger the download of kaggle.json file containing your API credentials. Place the file at `~/.kaggle/kaggle.json`.

```sh
# Download dataset
kaggle competitions download -c h-and-m-personalized-fashion-recommendations

# Create a directory to unzip data
mkdir ./data

# Unzip data
unzip h-and-m-personalized-fashion-recommendations.zip -d ./data
```

## Process Data

The downloaded dataset requires some preprocessing before we can use it. It involves creating train/val/test splits, removing purchased items with no corresponding images, etc.

Run the following commands to process the data.
Processing will take about 10 minutes.

```sh
cd src

# --data_root: directory containing articles.csv, customers.csv, images, etc.
python process_data.py --data_root ../data
```

## Prepare Data for Training

Now run the following commands to prepare the processed data for training.
Preprocessing will take about 30 minutes.

```sh
cd src

python prepare_data_for_training.py --data_root ../data
```

## Fine-tune VQGAN

Download VQGAN checkpoint pre-trained on ImageNet. See https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/ for more details.

Download LPIPS pre-trained model. See https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b for more details.

Preprocess the downloaded pre-trained model:

```sh
python process_pretrained_vqgan.py
python -m lightning.pytorch.utilities.upgrade_checkpoint ./pretrained/vqgan.ckpt
```

Prepare data for training:

```sh
python prepare_vqgan_data.py
```

Start fine-tuning:

```sh
python trainer_vqgan.py
```

To evaluate VQGAN model, run the notebook `notebooks/evaluate_vqgan.ipynb`.

Now we encode product images using VQGAN. This will take about an hour.

```sh
python encode_data.py
```

## Fine-tune BART

To start training:

```sh
cd src

python trainer_bart.py --data_dir ../data --output_dir ../logs
```

To evaluate BART model, run the notebook `notebooks/evaluate_bart.ipynb`.
