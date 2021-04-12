"""
Rather than re-invent the wheel here, we use heavily simplified versions of some of the automatic
cropping implementations available from the nnU-Net code for the paper:

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a
self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

The code below greatly compacts and simplifies the functions found in
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/cropping.py
"""

import numpy as np
from scipy.ndimage import binary_fill_holes


def crop_to_nonzero(img: np.ndarray) -> np.ndarray:
    assert len(img.shape) == 3, "Data must have shape (channels, width, height)"
    # fill holes to smooth masks
    mask = np.zeros(img.shape, dtype=bool)
    nonzero = img > 0
    mask = mask | nonzero
    filled = binary_fill_holes(mask)

    # crop to bounds of filled mask
    non_nulls = np.where(filled != 0)
    ch_min, ch_max = int(np.min(non_nulls[0])), int(np.max(non_nulls[0])) + 1
    x_min, x_max = int(np.min(non_nulls[1])), int(np.max(non_nulls[1])) + 1
    y_min, y_max = int(np.min(non_nulls[2])), int(np.max(non_nulls[2])) + 1
    slicer = (slice(ch_min, ch_max), slice(x_min, x_max), slice(y_min, y_max))
    return img[slicer]
