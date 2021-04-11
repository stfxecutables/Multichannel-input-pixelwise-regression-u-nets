# Some code is adapted from: https://github.com/fepegar/unet/tree/master/unet

from typing import Optional

import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernal_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        block = nn.ModuleList()
        class_name = "Conv{}d".format(dimensions)
        conv_class = getattr(nn, class_name)

        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernal_size,
            padding=(kernal_size + 1) // 2 - 1,
            padding_mode=padding_mode,
            bias=use_bias,
        )

        norm_layer = None
        if normalization is not None:
            if normalization == "Batch":
                class_name = "{}Norm{}d".format(normalization.capitalize(), dimensions)
                norm_class = getattr(nn, class_name)
                norm_layer = norm_class(out_channels)
            elif normalization == "Group":
                class_name = "{}Norm".format(normalization.capitalize())
                norm_class = getattr(nn, class_name)
                norm_layer = norm_class(num_groups=1, num_channels=out_channels)
            elif normalization == "InstanceNorm3d":
                class_name = normalization
                norm_class = getattr(nn, class_name)
                norm_layer = norm_class(num_features=out_channels, affine=True, track_running_stats=True)

        activation_layer = None
        if activation is not None:
            if activation == "ReLU":
                activation_layer = getattr(nn, activation)()
            elif activation == "LeakyReLU":
                activation_layer = getattr(nn, activation)(0.2)

        self.add_if_not_none(block, conv_layer)
        self.add_if_not_none(block, norm_layer)
        self.add_if_not_none(block, activation_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)
