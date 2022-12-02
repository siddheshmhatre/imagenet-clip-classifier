import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet1kEmbeddings(Dataset):
    def __init__(self, features, targets, debug=False) -> None:
        super().__init__()

        self.debug = debug
        self.features = features
        self.targets = targets

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        if self.debug:
            return min(self.features.shape[0], 81920)
        else:
            return self.features.shape[0]


def load_datasets(root_dir, debug) -> dict:
    datasets_dict = {}
    train_features = torch.load(os.path.join(root_dir, "features_train.pt"))
    train_targets = torch.load(os.path.join(root_dir, "targets_train.pt"))
    datasets_dict["train"] = ImageNet1kEmbeddings(train_features, train_targets, debug)

    val_features = torch.load(os.path.join(root_dir, "features_val.pt"))
    val_targets = torch.load(os.path.join(root_dir, "targets_val.pt"))
    datasets_dict["validation"] = ImageNet1kEmbeddings(val_features, val_targets, debug)
    datasets_dict["test"] = None
    return datasets_dict
