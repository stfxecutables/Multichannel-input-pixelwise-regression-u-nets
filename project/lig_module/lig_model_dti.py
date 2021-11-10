from argparse import ArgumentParser
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import MSELoss, Sigmoid
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.unet.unet import UNet
from utils.visualize_dti import log_all_info


class LitModelDiffusion(pl.LightningModule):
    def __init__(self, hparams: AttributeDict):
        super(LitModelDiffusion, self).__init__()
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
        self.clip_min = self.hparams.clip_min
        self.clip_max = self.hparams.clip_max

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1 - x))

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch["image"], batch["label"]

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))

        if self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                target=targets[0],
                preb=logits[0],
                loss=loss,
                batch_idx=batch_idx,
                state="train",
            )
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch["image"], batch["label"]

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))

        if self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                target=targets[0],
                preb=logits[0],
                loss=loss,
                batch_idx=batch_idx,
                state="val",
            )
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        if batch_idx <= 5:
            np.savez(f"{batch_idx}.npz", target=targets, predict=predicts)

        mse = np.square(np.subtract(targets, predicts)).mean()
        ssim_ = ssim(targets, predicts, data_range=predicts.max() - predicts.min())
        psnr_ = psnr(targets, predicts, data_range=predicts.max() - predicts.min())

        brain_mask = targets == targets[0][0][0]

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
        np.mean(diff_255_mask)

        # return {"MAE": mae, "MAE_mask": mae_mask, "MSE": mse}
        return {"MAE": mae, "MSE": mse, "SSIM": ssim_, "PSNR": psnr_}

    def validation_epoch_end(self, validation_step_outputs):
        average = np.mean(
            [validation_step_output["MAE"] for validation_step_output in validation_step_outputs]
        )
        self.log("val_MAE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on whole image: {average}")

        average = np.mean(
            [validation_step_output["MSE"] for validation_step_output in validation_step_outputs]
        )
        self.log("val_MSE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average square error on whole image: {average}")

        average = np.mean(
            [validation_step_output["SSIM"] for validation_step_output in validation_step_outputs]
        )
        self.log("val_SSIM", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"SSIM on whole image: {average}")

        average = np.mean(
            [validation_step_output["PSNR"] for validation_step_output in validation_step_outputs]
        )
        self.log("val_PSNR", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"PSNR on whole image: {average}")

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch["image"], batch["label"]
        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))

        if batch_idx <= 15:
            log_all_info(
                module=self,
                target=targets,
                preb=logits,
                loss=loss,
                batch_idx=batch_idx,
                state="val",
            )

        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        mse = np.square(np.subtract(targets, predicts)).mean()

        if batch_idx <= 5:
            np.savez(f"{batch_idx}.npz", target=targets, predict=predicts)

        brain_mask = targets == targets[0][0][0]

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
        average = np.mean(
            [validation_step_output["MAE"] for validation_step_output in test_step_outputs]
        )
        self.log("test_MAE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on whole image: {average}")

        average = np.mean(
            [validation_step_output["MAE_mask"] for validation_step_output in test_step_outputs]
        )
        self.log("test_MAE_mask", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on mask: {average}")

        average = np.mean(
            [validation_step_output["MSE"] for validation_step_output in test_step_outputs]
        )
        self.log("test_MSE", average, sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on mask: {average}")

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
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        # parser.add_argument("--loss", type=str, default="BCEWL", help="Loss Function")
        # parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        # parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        # parser.add_argument("--deepth", type=int, default=4, help="the deepth of the unet")
        # parser.add_argument("--normalization", type=str, default="Batch")
        # parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        # parser.add_argument("--model", type=str, default="ResUnet")
        # parser.add_argument("--downsampling_type", type=str, default="max")
        return parser
