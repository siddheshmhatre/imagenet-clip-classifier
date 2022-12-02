# imagenet-clip-classifier

Train a simple MLP on clip embeddings to do imagenet classification

## Data

Download pre-computed CLIP embeddings from [here](https://drive.google.com/file/d/1Du1-f71iN-E-XhICl8AkTns-K0hOHx7u/view?usp=sharing)
The clip embeddings were created using the [open_clip](https://github.com/mlfoundations/open_clip). The CLIP model used is `ViT-H-14` and checkpoint is `laion2b_s32b_b79k`

## Setup environment

Setup using virtualenv

```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Run training

After setting up the environment and uncompressing the downloaded data run the following command

```
python train.py dataset.root_dir=/path/to/data
```