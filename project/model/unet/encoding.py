# Code in this file adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py

from typing import Dict, List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .conv import ConvolutionalBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        conv_num_in_layer: List[int],
        residual: bool,
        kernel_size: int,
        normalization: str,
        downsampling_type: str,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        num_encoding_blocks = len(conv_num_in_layer) - 1
        out_channels = out_channels_first
        for idx in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                conv_num=conv_num_in_layer[idx],
                residual=residual,
                normalization=normalization,
                kernel_size=kernel_size,
                downsampling_type=downsampling_type,
                padding_mode=padding_mode,
                activation=activation,
                num_block=idx,
                use_bias=use_bias,
            )
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels
                out_channels = in_channels * 2
            elif dimensions == 3:
                in_channels = out_channels
                out_channels = in_channels * 2

            self.out_channels = self.encoding_blocks[-1].out_channels

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        skips = []
        for encoding_block in self.encoding_blocks:  # nn.ModuleList need to iterate!!!!
            x, skip = encoding_block(x)
            skips.append(skip)
        return skips, x


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int,
        residual: bool,
        normalization: Optional[str],
        conv_num: int,
        num_block: int,
        kernel_size: int = 5,
        downsampling_type: Optional[str] = "conv",
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.num_block = num_block
        self.residual = residual
        opts: Dict = dict(
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )
        conv_blocks = [ConvolutionalBlock(dimensions, in_channels, out_channels, **opts)]

        for _ in range(conv_num - 1):
            conv_blocks.append(
                ConvolutionalBlock(
                    dimensions, in_channels=out_channels, out_channels=out_channels, **opts
                )
            )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            maxpool = getattr(nn, f"MaxPool{dimensions}d")
            self.downsample = maxpool(kernel_size=2)
        elif downsampling_type == "conv":
            self.downsample = nn.Conv3d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2
            )

        self.out_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.residual:
            residual_layer = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += residual_layer
        else:
            x = self.conv_blocks(x)

        if self.downsample is None:
            return x

        skip = x
        return self.downsample(x), skip
