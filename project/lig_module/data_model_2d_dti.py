import functools
import os
from utils.const import DATA_ROOT
import torch
from pathlib import Path
from monai import transforms
from typing import List, Optional, Tuple

import monai
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL
from monai.transforms import Compose
from utils.transforms import get_diffusion_preprocess, get_diffusion_label_preprocess
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, apply_transform
from torch.utils.data import DataLoader, Dataset


class Diffusion2DDataset(Dataset):
    def __init__(self, path: List[str], X_transform: Compose, type: str):
        self.path = path
        self.X_transform = X_transform
        self.y_transform = get_diffusion_label_preprocess()
        self.type = type

    def __len__(self):
        return int(len(self.path))

    def __getitem__(self, i):
        tmp = np.load(self.path[i])
        if self.type == "ADC":
            X_img, y_img = tmp["X"], tmp["ADC"]
        elif self.type == "FA":
            X_img, y_img = tmp["X"], tmp["FA"]

        X_img = np.transpose(X_img, (3, 0, 1, 2))
        y_img = np.squeeze(y_img)
        y_img = np.transpose(y_img, (2, 0, 1))

        return torch.from_numpy(X_img).float(), torch.from_numpy(y_img).float()


class DataModule2DDiffusion(pl.LightningDataModule):
    def __init__(self, type: str):
        super().__init__()
        self.type = type

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X = sorted(list(DATA_ROOT.glob("**/*.npz")))
        preprocess = get_diffusion_preprocess()

        self.train_dataset = Diffusion2DDataset(path=X[:-1] * 200, X_transform=preprocess, type=self.type)
        self.val_dataset = Diffusion2DDataset(
            path=[X[-1]] * 4, X_transform=preprocess, type=self.type
        )  # *4 in order to allocate on 4 GPUs

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(self.train_dataset, batch_size=1, num_workers=0, shuffle=True)

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)
