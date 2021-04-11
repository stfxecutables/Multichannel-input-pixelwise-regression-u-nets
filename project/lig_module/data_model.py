import functools
import os
from pathlib import Path
from typing import List, Optional, Tuple

import monai
import random
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.const import DATA_ROOT
from monai.transforms import Compose
from utils.transforms import get_train_img_transforms, get_val_img_transforms, get_label_transforms
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, Randomizable, apply_transform
from torch.utils.data import DataLoader, Dataset


class BraTSDataset(Dataset, Randomizable):
    def __init__(self, X_path: List[str], y_path: List[str], transform: Compose, using_flair: bool):
        self.X_path = X_path
        self.y_path = y_path
        self.X_transform = transform
        self.y_transform = get_label_transforms()
        self.using_flair = using_flair

    def __len__(self):
        return int(len(self.X_path))

    # What is this used for?
    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, i):
        self.randomize()
        loadnifti = LoadNifti()
        X_img, compatible_meta = loadnifti(self.X_path[i])
        if int(len(self.X_path)) < 1000 and i < 5:  # only print the val x_path
            print(f"No. {i} file, path: {self.X_path[i]}")
        y_img, compatible_meta = loadnifti(self.y_path[i])

        if isinstance(self.X_transform, Randomizable):
            self.X_transform.set_random_state(seed=self._seed)
            self.y_transform.set_random_state(seed=self._seed)
        X_img = apply_transform(self.X_transform, X_img)
        y_img = apply_transform(self.y_transform, y_img)

        if self.using_flair:
            X_path_str = str(self.X_path[i])
            if "t1" in X_path_str:
                X_fair_path = X_path_str.replace("t1", "flair")
            else:
                X_fair_path = X_path_str.replace("t2", "flair")
            X_fair, compatible_meta = loadnifti(Path(X_fair_path))
            X_fair_img = apply_transform(self.X_transform, X_fair)
            X_img = torch.cat((X_img, X_fair_img), 0)

        return X_img, y_img


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, X_image: str, y_image: str, using_flair: bool, fine_tune: bool):
        super().__init__()
        self.batch_size = batch_size
        self.X_image = X_image
        self.y_image = y_image
        self.using_flair = using_flair
        self.fine_tune = fine_tune

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X = sorted(list(DATA_ROOT.glob(f"**/*{self.X_image}.nii.gz")))
        y = sorted(list(DATA_ROOT.glob(f"**/*{self.y_image}.nii.gz")))

        random_state = random.randint(0, 100)

        if self.fine_tune:
            X = X[:300]
            y = y[:300]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

        train_transforms = get_train_img_transforms()
        val_transforms = get_val_img_transforms()
        self.train_dataset = BraTSDataset(
            X_path=X_train, y_path=y_train, transform=train_transforms, using_flair=self.using_flair
        )
        self.val_dataset = BraTSDataset(
            X_path=X_val, y_path=y_val, transform=val_transforms, using_flair=self.using_flair
        )

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)
