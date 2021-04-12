import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary

# from monai.networks.nets import UNet
from model.unet.unet import UNet, VNet
import torch


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 256, 256)

        self.model = UNet(
            in_channels=1,
            out_classes=1,
            dimensions=2,
            padding_mode="zeros",
            activation="LeakyReLU",
            conv_num_in_layer=[1, 2, 3, 3, 3],
            residual=False,
            out_channels_first_layer=16,
            kernel_size=5,
            normalization="Batch",
            downsampling_type="max",
            use_bias=False,
        )

    def forward(self, x):
        return self.model(x)


# class HighResNetModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

#         self.unet = HighResNet(in_channels=1, out_channels=139, dimensions=3)

#     def forward(self, x):
#         return self.unet(x)


if __name__ == "__main__":
    # HighResNet = HighResNetModel()
    # print("highResNet Model:")
    # print(ModelSummary(HighResNet, mode="full"))

    Net = Model()
    print(ModelSummary(Net, mode="full"))
