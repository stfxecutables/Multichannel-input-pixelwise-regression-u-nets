# Code in this file adapted from: https://github.com/fepegar/unet/blob/master/unet/conv.py

from typing import List, Optional

import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        block = nn.ModuleList()
        conv = getattr(nn, f"Conv{dimensions}d")

        conv_layer = [
            conv(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size + 1) // 2 - 1,
                padding_mode=padding_mode,
                bias=use_bias,
            )
        ]

        norm_layer: List[Module] = []
        if normalization is not None:
            if normalization == "Batch":
                norm = getattr(nn, f"BatchNorm{dimensions}d")
                norm_layer.append(norm(out_channels))
            elif normalization == "Group":
                norm_layer.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
            elif normalization == "InstanceNorm3d":
                norm_layer.append(
                    nn.InstanceNorm3d(
                        num_features=out_channels, affine=True, track_running_stats=True
                    )
                )

        activation_layer: List[Module] = []
        if activation is not None:
            if activation == "ReLU":
                activation_layer.append(nn.ReLU())
            elif activation == "LeakyReLU":
                activation_layer.append(nn.LeakyReLU(0.2))

        block.extend(conv_layer)
        block.extend(norm_layer)
        block.extend(activation_layer)
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)  # type: ignore
