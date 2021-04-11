"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""

import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes, Figure
from matplotlib.text import Text
from numpy import ndarray
import numpy as np

from collections import OrderedDict
from numpy import ndarray
from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Optional
from pytorch_lightning.core.lightning import LightningModule
import matplotlib.gridspec as gridspec


"""
For TensorBoard logging usage, see:
https://www.tensorflow.org/api_docs/python/tf/summary
For Lightning documentation / examples, see:
https://pytorch-lightning.readthedocs.io/en/latest/experiment_logging.html#tensorboard
NOTE: The Lightning documentation here is not obvious to newcomers. However,
`self.logger` returns the Torch TensorBoardLogger object (generally quite
useless) and `self.logger.experiment` returns the actual TensorFlow
SummaryWriter object (e.g. with all the methods you actually care about)
For the Lightning methods to access the TensorBoard .summary() features, see
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.html#pytorch_lightning.loggers.TensorBoardLogger
**kwargs for SummaryWriter constructor defined at
https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer
^^ these args look largely like things we don't care about ^^
"""


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue."""
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    return scaled


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(str(logdir), name="unet")


# https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
class BrainSlices:
    def __init__(self, lightning: LightningModule, target: Tensor, prediction: Tensor):
        self.lightning = lightning
        self.target_img: ndarray = target.cpu().detach().numpy().squeeze() if torch.is_tensor(target) else target
        self.predict_img: ndarray = (
            prediction.cpu().detach().numpy().squeeze() if torch.is_tensor(prediction) else prediction
        )

        si, sj, sk = self.target_img.shape[:3]
        i = si // 2
        j = sj // 2
        k = sk // 2
        self.slices = [
            self.get_slice(self.target_img, i, j, k),
            self.get_slice(self.predict_img, i, j, k),
        ]

        self.shape = np.array(self.target_img.shape)

    def get_slice(self, input: np.ndarray, i: int, j: int, k: int):
        return [
            (input[i // 2, ...], input[i, ...], input[i + i // 2, ...]),
            (input[:, j // 2, ...], input[:, j, ...], input[:, j + j // 2, ...]),
            (input[:, :, k // 2, ...], input[:, :, k, ...], input[:, :, k + k // 2, ...]),
        ]

    # def get_slice(self, input: np.ndarray, i: int, j: int, k: int):
    #     return [input[i, ...], input[:, j, ...], input[:, :, k, ...]]

    def plot(self) -> Figure:
        nrows, ncols = 2, 3

        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if i == 0:
                    axis.set_title("target image")
                else:
                    axis.set_title("predict image")

        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            imgs = [img for img in slice_]
            imgs = np.concatenate(imgs, axis=1)

            axis.imshow(imgs, cmap="bone", alpha=0.8)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, state: str, fig: Figure, loss: float, batch_idx: int) -> None:
        logger = self.lightning.logger
        summary = f"{state}-Epoch:{self.lightning.current_epoch + 1}-batch:{batch_idx}-loss:{loss:0.5e}"
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_all_info(
    module: LightningModule,
    target: Union[Tensor, ndarray],
    preb: Union[Tensor, ndarray],
    loss: float,
    batch_idx: int,
    state: str,
) -> None:
    brainSlice = BrainSlices(module, target, preb)
    fig = brainSlice.plot()

    # fig.savefig("/home/jueqi/projects/def-jlevman/jueqi/rUnet/3/tmp.png")
    brainSlice.log(state, fig, loss, batch_idx)
