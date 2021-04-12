# Code in this file adapted from https://github.com/fepegar/unet/blob/master/unet/decoding.py

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = ("nearest", "linear", "bilinear", "bicubic", "trilinear")


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        conv_num_in_layer: Sequence[int],
        upsampling_type: str,
        residual: bool,
        normalization: Optional[str],
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()
        self.decoding_blocks = nn.ModuleList()
        for idx, conv_num in enumerate(conv_num_in_layer):
            decoding_block = DecodingBlock(
                in_channels_skip_connection=in_channels_skip_connection,
                dimensions=dimensions,
                upsampling_type=upsampling_type,
                normalization=normalization,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                activation=activation,
                residual=residual,
                conv_num=conv_num,
                block_num=idx,
                use_bias=use_bias,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2

        self.out_channels = in_channels_skip_connection * 4

    def forward(self, skip_connections: List[Tensor], x: Tensor) -> Tensor:
        # print(f"x type: {type(x)}, length: {len(x)}")
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        residual: bool,
        conv_num: int,
        block_num: int,
        normalization: Optional[str] = "Group",
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            if block_num == 0:
                in_channels = in_channels_skip_connection * 2
                out_channels = in_channels_skip_connection
            else:
                in_channels = in_channels_skip_connection * 4
                out_channels = in_channels_skip_connection * 2
            conv = getattr(nn, f"ConvTranspose{dimensions}d")
            self.upsample = conv(
                in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
            )
        else:
            raise NotImplementedError()

        if block_num == 0:
            in_channels_first = in_channels_skip_connection * 2
            out_channels = in_channels_skip_connection * 2
        else:
            in_channels_first = in_channels_skip_connection * 3
            out_channels = in_channels_skip_connection * 2

        options: Dict[str, Any] = dict(
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        conv_blocks = [ConvolutionalBlock(dimensions, in_channels_first, out_channels, **options)]

        for _ in range(conv_num - 1):
            conv_blocks.append(
                ConvolutionalBlock(
                    dimensions, in_channels=out_channels, out_channels=out_channels, **options
                )
            )

        self.conv_blocks = nn.Sequential(*conv_blocks)

        if self.residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection: Tensor, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)

        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += connection
        else:
            x = self.conv_blocks(x)
        return x
