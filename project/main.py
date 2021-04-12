import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lig_module.data_model import DataModule
from lig_module.data_model_2d_dti import DataModule2DDiffusion
from lig_module.data_model_dti import DataModuleDiffusion
from lig_module.data_model_longitudinal import DataModuleLongitudinal
from lig_module.lig_model import LitModel
from lig_module.lig_model_2d_dti import LitModel2DDiffusion
from lig_module.lig_model_dti import LitModelDiffusion
from lig_module.lig_model_longitudinal import LitModelLongitudinal
from utils.const import COMPUTECANADA


def main(hparams: Namespace) -> None:
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy,
    # python.random and sets PYTHONHASHSEED environment variable.
    pl.seed_everything(42)

    if COMPUTECANADA:
        cur_path = Path(__file__).resolve().parent
        default_root_dir = cur_path
        checkpoint_file = (
            Path(__file__).resolve().parent
            / "checkpoint/{epoch}-{val_loss:0.5f}-{val_MAE:0.5f}-{val_MSE:0.5f}-{val_SSIM:0.5f}-{val_PSNR:0.5f}"  # noqa
        )
        if not os.path.exists(Path(__file__).resolve().parent / "checkpoint"):
            os.mkdir(Path(__file__).resolve().parent / "checkpoint")
    else:
        default_root_dir = Path("./log")
        if not os.path.exists(default_root_dir):
            os.mkdir(default_root_dir)
        checkpoint_file = Path("./log/checkpoint")
        if not os.path.exists(checkpoint_file):
            os.mkdir(checkpoint_file)
        checkpoint_file = checkpoint_file / "{epoch}-{val_loss:0.5f}-{val_MAE:0.5f}"

    ckpt_path = (
        str(Path(__file__).resolve().parent / "checkpoint" / hparams.checkpoint_file)
        if hparams.checkpoint_file
        else None
    )

    # After training finishes, use best_model_path to retrieve the path to the best
    # checkpoint file and best_model_score to retrieve its score.
    checkpoint_callback = ModelCheckpoint(  # type: ignore
        filepath=str(checkpoint_file),
        monitor="val_loss",
        save_top_k=1,
        verbose=True,
        mode="min",
        save_weights_only=False,
    )
    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    # training
    trainer = Trainer(
        gpus=hparams.gpus,
        distributed_backend="ddp",
        fast_dev_run=hparams.fast_dev_run,
        checkpoint_callback=checkpoint_callback,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            # EarlyStopping("val_loss", patience=20, mode="min"),
        ],
        resume_from_checkpoint=ckpt_path,
        default_root_dir=str(default_root_dir),
        logger=tb_logger,
        max_epochs=75,
        # auto_scale_batch_size="binsearch", # for auto scaling of batch size
    )

    if hparams.task == "t1t2":
        model = LitModel(hparams)
        data_module = DataModule(
            hparams.batch_size,
            X_image=hparams.X_image,
            y_image=hparams.y_image,
            using_flair=hparams.use_flair,
            fine_tune=hparams.fine_tune,
        )
    elif hparams.task == "diffusion_adc":
        model = LitModelDiffusion(hparams)
        data_module = DataModuleDiffusion(
            hparams.batch_size,
            type="ADC",
            use_data_augmentation=hparams.use_data_augmentation,
            times=hparams.times,
        )
    elif hparams.task == "diffusion_fa":
        model = LitModelDiffusion(hparams)
        data_module = DataModuleDiffusion(
            hparams.batch_size,
            type="FA",
            use_data_augmentation=hparams.use_data_augmentation,
            times=hparams.times,
        )
    elif hparams.task == "diffusion_2d_adc":
        model = LitModel2DDiffusion(hparams)
        data_module = DataModule2DDiffusion(type="ADC")
    elif hparams.task == "diffusion_2d_fa":
        model = LitModel2DDiffusion(hparams)
        data_module = DataModule2DDiffusion(type="FA")
    elif hparams.task == "longitudinal":
        model = LitModelLongitudinal(hparams)
        data_module = DataModuleLongitudinal(
            hparams.batch_size, num_scan_training=hparams.in_channels
        )

    trainer.fit(model, data_module)

    # trainer = Trainer(gpus=hparams.gpus, distributed_backend="ddp")
    # trainer.test(
    #     model=model,
    #     ckpt_path=ckpt_path,
    #     datamodule=data_module,
    # )

    # trainer.test()


if __name__ == "__main__":  # pragma: no cover
    parser = ArgumentParser(description="Trainer args", add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size", dest="batch_size")
    parser.add_argument(
        "--tensor_board_logger",
        dest="TensorBoardLogger",
        default="/home/jq/Desktop/log",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="whether to run 1 train, val, test batch and program ends",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="using this mode to fine tune, only use 300 images here",
    )
    parser.add_argument("--use_flair", action="store_true")
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--in_channels", type=int, default=288)
    parser.add_argument("--X_image", type=str, choices=["t1", "t2"], default="t1")
    parser.add_argument("--y_image", type=str, choices=["t1", "t2"], default="t2")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "t1t2",
            "diffusion_adc",
            "diffusion_fa",
            "diffusion_2d_adc",
            "diffusion_2d_fa",
            "longitudinal",
        ],
        default="diffusion_adc",
    )
    parser.add_argument("--checkpoint_file", type=str, help="resume from checkpoint file")
    parser = LitModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
