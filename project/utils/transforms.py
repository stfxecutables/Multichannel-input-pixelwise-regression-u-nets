from typing import List

import numpy as np
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandAffined,
    Resize,
    SpatialPad,
    ToTensor,
    ToTensord,
)
from monai.transforms.compose import Transform

from utils.const import IMAGESIZE
from utils.cropping import crop_to_nonzero


class Crop(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return crop_to_nonzero(img)


class Unsqueeze(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.expand_dims(img, axis=0)


class Squeeze(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.squeeze(img)


class Transpose(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (3, 0, 1, 2))


class Transpose2DInput(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (2, 3, 0, 1))


class Transpose2DLabel(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (2, 0, 1))


def get_diffusion_preprocess() -> Compose:
    return Compose(
        [
            NormalizeIntensity(nonzero=True),
            Transpose(),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
            ToTensor(),
        ]
    )


def get_diffusion_label_preprocess() -> Compose:
    return Compose(
        [
            NormalizeIntensity(nonzero=True),
            Unsqueeze(),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
            ToTensor(),
        ]
    )


def get_diffusion_transform() -> Compose:
    return Compose(
        [
            # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.2,
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_2D_diffusion_preprocess() -> Compose:
    return Compose([NormalizeIntensity(nonzero=True), Transpose2DInput()])


def get_2D_diffusion_label_preprocess() -> Compose:
    return Compose([NormalizeIntensity(nonzero=True), Transpose2DLabel()])


def get_preprocess(is_label: bool) -> List[Transform]:
    if not is_label:
        return [
            Crop(),
            NormalizeIntensity(nonzero=True),
            # Channel
            Unsqueeze(),
            SpatialPad(spatial_size=[193, 193, 193], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]
    else:
        return [
            Crop(),
            NormalizeIntensity(nonzero=True),
            Unsqueeze(),
            SpatialPad(spatial_size=[193, 193, 193], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]


def get_longitudinal_preprocess(is_label: bool) -> List[Transform]:
    # only without cropping, somehow, there is not much left to crop in this dataset...
    if not is_label:
        return [
            NormalizeIntensity(nonzero=True),
            Unsqueeze(),
            SpatialPad(spatial_size=[215, 215, 215], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]
    else:
        return [
            NormalizeIntensity(nonzero=True),
            Unsqueeze(),
            SpatialPad(spatial_size=[215, 215, 215], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]


def get_train_img_transforms() -> Compose:
    preprocess = get_preprocess(is_label=False)
    train_augmentation: List[Transform] = [ToTensor()]
    return Compose(preprocess + train_augmentation)


def get_val_img_transforms() -> Compose:
    preprocess = get_preprocess(is_label=False)
    return Compose(preprocess + [ToTensor()])


def get_label_transforms() -> Compose:
    preprocess = get_preprocess(is_label=True)
    return Compose(preprocess + [ToTensor()])


def get_longitudinal_train_img_transforms() -> Compose:
    preprocess = get_longitudinal_preprocess(is_label=False)
    train_augmentation: List[Transform] = [ToTensor()]
    return Compose(preprocess + train_augmentation)


def get_longitudinal_val_img_transforms() -> Compose:
    preprocess = get_longitudinal_preprocess(is_label=False)
    return Compose(preprocess + [ToTensor()])
    # return Compose(preprocess)


def get_longitudinal_label_transforms() -> Compose:
    preprocess = get_longitudinal_preprocess(is_label=True)
    return Compose(preprocess + [ToTensor()])
    # return Compose(preprocess)
