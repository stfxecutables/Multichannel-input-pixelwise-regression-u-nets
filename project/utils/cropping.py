"""
using Kmeans to make the threshold and do crop on the MR image
Some code are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/cropping.py
"""

import numpy as np
from typing import List


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes

    assert len(data.shape) == 3, "data must have shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data > 0
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask: np.ndarray, outside_value: int = 0) -> List[List[int]]:
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(img: np.ndarray, bbox: List[List[int]]) -> np.ndarray:
    assert len(img.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return img[resizer]


def crop_to_nonzero(img: np.ndarray) -> np.ndarray:
    nonzero_mask = create_nonzero_mask(img)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    img = crop_to_bbox(img, bbox)
    return img
