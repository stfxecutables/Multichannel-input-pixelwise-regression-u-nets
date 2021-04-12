import random
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from monai.transforms import Compose, apply_transform
from monai.utils import set_determinism
from torch.utils.data import DataLoader, Dataset

from utils.const import DATA_ROOT
from utils.transforms import get_diffusion_transform


class DiffusionDataset(Dataset):
    def __init__(self, path: List[str], type: str, transform: Optional[Compose] = None):
        self.path = path
        self.transform = transform
        self.type = type

    def __len__(self):
        return int(len(self.path))

    def __getitem__(self, i):
        tmp = np.load(self.path[i])
        if self.type == "ADC":
            X_img, y_img = tmp["X"], tmp["ADC"]
        elif self.type == "FA":
            X_img, y_img = tmp["X"], tmp["FA"]

        if self.transform is not None:
            data = {"image": X_img, "label": y_img}
            data = apply_transform(transform=self.transform, data=data)
        else:
            data = {
                "image": torch.from_numpy(X_img).float(),
                "label": torch.from_numpy(y_img).float(),
            }

        return data


class DataModuleDiffusion(pl.LightningDataModule):
    def __init__(self, batch_size: int, type: str, use_data_augmentation: bool, times: int):
        super().__init__()
        self.batch_size = batch_size
        self.type = type
        self.use_data_augmentation = use_data_augmentation
        self.times = times

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X = sorted(list(DATA_ROOT.glob("**/*.npz")))
        if self.use_data_augmentation:
            transform = get_diffusion_transform()
        else:
            transform = None

        print(f"validation image path: {X[-3:]}")

        self.train_dataset = DiffusionDataset(
            path=X[:-3] * self.times, transform=transform, type=self.type
        )
        self.val_dataset = DiffusionDataset(
            path=X[-3:] * 4, type=self.type
        )  # *4 in order to allocate on 4 GPUs

    def prepare_data(self, *args, **kwargs):
        # set deterministic training for reproducibility
        random_state = random.randint(0, 100)
        set_determinism(seed=random_state)
        return super().prepare_data(*args, **kwargs)

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)
