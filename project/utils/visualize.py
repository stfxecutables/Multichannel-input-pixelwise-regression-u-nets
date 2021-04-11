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
    def __init__(
        self,
        lightning: LightningModule,
        img: Optional[Tensor],
        target: Tensor,
        prediction: Tensor,
        input_img_type: str,
        target_img_type: str,
    ):
        self.lightning = lightning
        self.input_img: ndarray = img.cpu().detach().numpy().squeeze() if torch.is_tensor(img) else img
        self.target_img: ndarray = target.cpu().detach().numpy().squeeze() if torch.is_tensor(target) else target
        self.predict_img: ndarray = make_imgs(
            prediction.cpu().detach().numpy().squeeze() if torch.is_tensor(prediction) else prediction
        )
        self.input_img_type = input_img_type
        self.target_img_type = target_img_type

        if self.lightning.hparams.task == "t1t2":
            if len(self.input_img.shape) == 3:
                si, sj, sk = self.input_img.shape
                i = si // 2
                j = sj // 2
                k = sk // 2
                self.slices = [
                    self.get_slice(self.input_img, i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]
            else:
                si, sj, sk = self.input_img.shape[1:]
                i = si // 2
                j = sj // 2
                k = sk // 2
                self.slices = [
                    self.get_slice(self.input_img[0], i, j, k),
                    self.get_slice(self.input_img[1], i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]
        elif self.lightning.hparams.task == "longitudinal":
            if len(self.input_img.shape) == 3:
                si, sj, sk = self.input_img.shape
                i = si // 2
                j = sj // 2
                k = sk // 2
                self.slices = [
                    self.get_slice(self.input_img, i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]
            else:
                si, sj, sk = self.input_img.shape[1:]
                i = si // 2
                j = sj // 2
                k = sk // 2
                if self.input_img.shape[0] == 2:
                    self.slices = [
                        self.get_slice(self.input_img[0], i, j, k),
                        self.get_slice(self.input_img[1], i, j, k),
                        self.get_slice(self.target_img, i, j, k),
                        self.get_slice(self.predict_img, i, j, k),
                    ]
                elif self.input_img.shape[0] == 3:
                    self.slices = [
                        self.get_slice(self.input_img[0], i, j, k),
                        self.get_slice(self.input_img[1], i, j, k),
                        self.get_slice(self.input_img[2], i, j, k),
                        self.get_slice(self.target_img, i, j, k),
                        self.get_slice(self.predict_img, i, j, k),
                    ]
        self.title = ["input image: M12", "input image: M06", "input image: SC", "target image: M24", "predict image"]
        self.shape = np.array(self.input_img.shape)

    def get_slice(self, input: np.ndarray, i: int, j: int, k: int):
        return [
            (input[i // 2, ...], input[i, ...], input[i + i // 2, ...]),
            (input[:, j // 2, ...], input[:, j, ...], input[:, j + j // 2, ...]),
            (input[:, :, k // 2, ...], input[:, :, k, ...], input[:, :, k + k // 2, ...]),
        ]

    # def get_slice(self, input: np.ndarray, i: int, j: int, k: int):
    #     return [input[i, ...], input[:, j, ...], input[:, :, k, ...]]

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 3  # one row for each slice position

        if nrows == 5:
            fig = plt.figure(figsize=(14, 10))
        else:
            fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if self.lightning.hparams.task == "t1t2":
                    if i == 0:
                        axis.set_title(f"input image: {self.input_img_type}")
                        continue
                    if len(self.slices) == 4:
                        if i == 1:
                            axis.set_title(f"input image: flair")
                        elif i == 2:
                            axis.set_title(f"target image: {self.target_img_type}")
                        else:
                            axis.set_title(f"predict image")
                    else:
                        if i == 1:
                            axis.set_title(f"target image: {self.target_img_type}")
                        else:
                            axis.set_title(f"predict image")
                elif self.lightning.hparams.task == "longitudinal":
                    if i == 0:
                        axis.set_title(self.title[0])
                    elif i == (len(self.slices) - 1):
                        axis.set_title(self.title[-1])
                    elif i == (len(self.slices) - 2):
                        axis.set_title(self.title[-2])
                    elif i == 1:
                        axis.set_title(self.title[1])
                    elif i == 2:
                        axis.set_title(self.title[2])
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

    # code is borrowed from: https://github.com/DM-Berger/autocrop/blob/master/autocrop/visualize.py#L125
    def animate_masks(
        self,
        dpi: int = 100,
        n_frames: int = 128,
        fig_title: str = None,
        outfile: Path = None,
    ) -> None:
        def get_slice(img: ndarray, ratio: float) -> ndarray:
            """Returns eig_img, raw_img"""

            if ratio < 0 or ratio > 1:
                raise ValueError("Invalid slice position")
            if len(img.shape) == 3:
                x_max, y_max, z_max = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape) * ratio), dtype=int)
            elif len(img.shape) == 4:
                x_max, y_max, z_max, _ = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape[:-1]) * ratio), dtype=int)
            x = int(10 + ratio * (x_max - 20))  # make x go from 10:-10 of x_max
            y = int(10 + ratio * (y_max - 20))  # make x go from 10:-10 of x_max
            x = x - 1 if x == x_max else x
            y = y - 1 if y == y_max else y
            z = z - 1 if z == z_max else z
            img_np = np.concatenate([img[x, :, :], img[:, y, :], img[:, :, z]], axis=1)
            return img_np

        def init_frame(img: ndarray, ratio: float, fig: Figure, ax: Axes, title) -> Tuple[AxesImage, Colorbar, Text]:
            image_slice = get_slice(img, ratio=ratio)
            # the bigger alpha, the image would become more black
            true_args = dict(vmin=0, vmax=255, cmap="bone", alpha=0.8)

            im = ax.imshow(image_slice, animated=True, **true_args)
            # im = ax.imshow(image_slice, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            title = ax.set_title(title)
            cb = fig.colorbar(im, ax=ax)
            return im, cb, title

        def update_axis(img: ndarray, ratio: float, im: AxesImage) -> AxesImage:
            image_slice = get_slice(img, ratio=ratio)
            # mask_slice = get_slice(mask, ratio=ratio)

            # vn, vm = get_vranges()
            im.set_data(image_slice)
            # im.set_data(mask_slice)
            # im.set_clim(vn, vm)
            # we don't have to update cb, it is linked
            return im

        # owe a lot to below for animating the colorbars
        # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
        def init() -> Tuple[Figure, Axes, List[AxesImage], List[Colorbar]]:
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False)  # 3

            ims: List[AxesImage] = []
            cbs: List[Colorbar] = []

            for ax, img, mask, title in zip(axes.flat, self.scale_imgs, self.masks, self.mask_video_names):
                im, cb, title = init_frame(img=img, ratio=0.0, fig=fig, ax=ax, title=title)
                ims.append(im)
                cbs.append(cb)

            if fig_title is not None:
                fig.suptitle(fig_title)
            fig.tight_layout(h_pad=0)
            fig.set_size_inches(w=12, h=10)  # The width of the entire image displayed
            fig.subplots_adjust(hspace=0.2, wspace=0.0)
            return fig, axes, ims, cbs

        N_FRAMES = n_frames
        ratios = np.linspace(0, 1, num=N_FRAMES)

        fig, axes, ims, cbs = init()

        # awkward, but we need this defined after to close over the above variables
        def animate(f: int) -> Any:
            ratio = ratios[f]
            updated = []
            for im, img, mask in zip(ims, self.scale_imgs, self.masks):
                updated.append(update_axis(img=img, ratio=ratio, im=im))
            return updated

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=N_FRAMES,
            blit=False,
            interval=24000 / N_FRAMES,
            repeat_delay=100 if outfile is None else None,
        )

        if outfile is None:
            plt.show()
        else:
            pbar = tqdm(total=100, position=1, desc="mp4")

            def prog_logger(current_frame: int, total_frames: int = N_FRAMES) -> Any:
                if (current_frame % (total_frames // 10)) == 0 and (current_frame != 0):
                    pbar.update(10)
                # tqdm.write("Done task %i" % (100 * current_frame / total_frames))
                #     print("Saving... {:2.1f}%".format(100 * current_frame / total_frames))

            # writervideo = animation.FFMpegWriter(fps=60)
            ani.save(outfile, codec="h264", dpi=dpi, progress_callback=prog_logger)
            # ani.save(outfile, progress_callback=prog_logger, writer=writervideo)
            pbar.close()


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_all_info(
    module: LightningModule,
    img: Union[Tensor, ndarray],
    target: Union[Tensor, ndarray],
    preb: Union[Tensor, ndarray],
    loss: float,
    batch_idx: int,
    state: str,
    input_img_type: str,
    target_img_type: str,
) -> None:
    brainSlice = BrainSlices(module, img, target, preb, input_img_type=input_img_type, target_img_type=target_img_type)
    fig = brainSlice.plot()

    # fig.savefig("test.png")
    brainSlice.log(state, fig, loss, batch_idx)
