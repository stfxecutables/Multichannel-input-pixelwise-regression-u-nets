import random
from argparse import ArgumentParser
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import L1Loss, MSELoss, Sigmoid, SmoothL1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.unet.unet import UNet
from utils.visualize import log_all_info


def scale_img_to_0_255(img: np.ndarray, imin: Any = None, imax: Any = None) -> np.ndarray:
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) * (1 / (imax - imin))) * 255, dtype="uint8")
    scaled[scaled < 0] = 0
    scaled[scaled >= 255] = 255
    return scaled


class LitModelLongitudinal(pl.LightningModule):
    def __init__(self, hparams: AttributeDict):
        super(LitModelLongitudinal, self).__init__()
        self.hparams = hparams
        self.model = UNet(
            in_channels=hparams.in_channels,
            out_classes=1,
            dimensions=3,
            padding_mode="zeros",
            activation=hparams.activation,
            conv_num_in_layer=[1, 2, 3, 3, 3],
            residual=False,
            out_channels_first_layer=16,
            kernel_size=5,
            normalization=hparams.normalization,
            downsampling_type="max",
            use_sigmoid=False,
            use_bias=True,
        )
        self.sigmoid = Sigmoid()
        if self.hparams.loss == "l2":
            self.criterion = MSELoss()
        elif self.hparams.loss == "l1":
            self.criterion = L1Loss()
        elif self.hparams.loss == "smoothl1":
            self.criterion = SmoothL1Loss()
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)
        self.clip_min = self.hparams.clip_min
        self.clip_max = self.hparams.clip_max

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        ### before ###
        # loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        ### it should be ###
        loss = self.criterion(logits.view(-1), targets.view(-1))

        if self.current_epoch % 75 == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                preb=logits[0],
                loss=loss,
                batch_idx=batch_idx,
                state="train",
                input_img_type=self.hparams.X_image,
                target_img_type=self.hparams.y_image,
            )
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        if self.current_epoch % 75 == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                preb=logits[0],
                loss=loss,
                batch_idx=batch_idx,
                state="val",
                input_img_type=self.hparams.X_image,
                target_img_type=self.hparams.y_image,
            )
        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        mse = np.square(np.subtract(targets, predicts)).mean()
        ssim_ = ssim(targets, predicts, data_range=predicts.max() - predicts.min())
        psnr_ = psnr(targets, predicts, data_range=predicts.max() - predicts.min())

        if batch_idx <= 5:
            np.savez(f"{batch_idx}.npz", inputs=inputs, target=targets, predict=predicts)

        if self.hparams.in_channels >= 2:
            brain_mask = inputs[0] == inputs[0][0][0][0]
        elif self.hparams.in_channels == 1:
            brain_mask = inputs == inputs[0][0][0]

        pred_clip = np.clip(predicts, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(predicts)
        )
        targ_clip = np.clip(targets, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(targets)
        )
        pred_255 = np.floor(256 * (pred_clip / (self.clip_min + self.clip_max)))
        targ_255 = np.floor(256 * (targ_clip / (self.clip_min + self.clip_max)))
        pred_255[brain_mask] = 0
        targ_255[brain_mask] = 0

        diff_255 = np.absolute(pred_255.ravel() - targ_255.ravel())
        mae = np.mean(diff_255)

        # diff_255_mask = np.absolute(pred_255[~brain_mask].ravel() - targ_255[~brain_mask].ravel())
        # mae_mask = np.mean(diff_255_mask)

        return {"MAE": mae, "MSE": mse, "SSIM": ssim_, "PSNR": psnr_}

    def validation_epoch_end(self, validation_step_outputs):
        average = np.mean(validation_step_outputs[0]["MAE"])
        self.log("val_MAE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on whole image: {average}")

        average = np.mean(validation_step_outputs[0]["MSE"])
        self.log("val_MSE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average square error on whole image: {average}")

        average = np.mean(validation_step_outputs[0]["SSIM"])
        self.log("val_SSIM", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"SSIM on whole image: {average}")

        average = np.mean(validation_step_outputs[0]["PSNR"])
        self.log("val_PSNR", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"PSNR on whole image: {average}")

        # average = np.mean(validation_step_outputs[0]["MAE_mask"])
        # self.log("val_MAE_mask", average, sync_dist=True, on_step=False, on_epoch=True)
        # print(f"average absolute error on mask: {average}")

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))

        if batch_idx <= 15:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                preb=logits[0],
                loss=loss,
                batch_idx=batch_idx,
                state="test",
                input_img_type=self.hparams.X_image,
                target_img_type=self.hparams.y_image,
            )

        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        mse = np.square(np.subtract(targets, predicts)).mean()

        if batch_idx <= 5:
            np.savez(f"{batch_idx}.npz", inputs=inputs, target=targets, predict=predicts)

        if self.hparams.in_channels >= 2:
            brain_mask = inputs[0] == inputs[0][0][0][0]
        elif self.hparams.in_channels == 1:
            brain_mask = inputs == inputs[0][0][0]

        pred_clip = np.clip(predicts, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(predicts)
        )
        targ_clip = np.clip(targets, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(targets)
        )
        pred_255 = np.floor(256 * (pred_clip / (self.clip_min + self.clip_max)))
        targ_255 = np.floor(256 * (targ_clip / (self.clip_min + self.clip_max)))
        pred_255[brain_mask] = 0
        targ_255[brain_mask] = 0

        diff_255 = np.absolute(pred_255.ravel() - targ_255.ravel())
        mae = np.mean(diff_255)

        diff_255_mask = np.absolute(pred_255[~brain_mask].ravel() - targ_255[~brain_mask].ravel())
        mae_mask = np.mean(diff_255_mask)

        return {"MAE": mae, "MAE_mask": mae_mask, "MSE": mse}

    def test_epoch_end(self, test_step_outputs):
        average = np.mean(test_step_outputs[0]["MAE"])
        print(f"average absolute error on whole image: {average}")

        average = np.mean(test_step_outputs[0]["MAE_mask"])
        print(f"average absolute error on mask: {average}")

        average = np.mean(test_step_outputs[0]["MSE"])
        print(f"average square error on whole image: {average}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = ReduceLROnPlateau(optimizer, threshold=1e-10)
        lr_dict = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=300, eta_min=0.000001),
            "monitor": "val_checkpoint_on",  # Default: val_loss
            "reduce_on_plateau": True,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_dict]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-15)
        parser.add_argument("--loss", type=str, choices=["l1", "l2", "smoothl1"], default="l2")
        parser.add_argument(
            "--activation", type=str, choices=["ReLU", "LeakyReLU"], default="LeakyReLU"
        )
        parser.add_argument(
            "--normalization",
            type=str,
            choices=["Batch", "Group", "InstanceNorm3d"],
            default="InstanceNorm3d",
        )
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--clip_min", type=int, default=2)
        parser.add_argument("--clip_max", type=int, default=5)
        # parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        # parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        # parser.add_argument("--deepth", type=int, default=4, help="the deepth of the unet")
        # parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        # parser.add_argument("--model", type=str, default="ResUnet")
        # parser.add_argument("--downsampling_type", type=str, default="max")
        return parser
