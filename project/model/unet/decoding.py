# Some code is adapted from: https://github.com/fepegar/unet/tree/master/unet

from typing import Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "trilinear",
)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        conv_num_in_layer: Sequence[int],
        upsampling_type: str,
        residual: bool,
        normalization: Optional[str],
        kernal_size: int = 5,
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
                kernal_size=kernal_size,
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

    def forward(self, skip_connections, x):
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
        kernal_size: int = 5,
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
            self.upsample = get_conv_transpose_layer(dimensions, in_channels, out_channels)
        # else:
        #     self.upsample = get_upsampling_layer(upsampling_type)

        if block_num == 0:
            in_channels_first = in_channels_skip_connection * 2
            out_channels = in_channels_skip_connection * 2
        else:
            in_channels_first = in_channels_skip_connection * 3
            out_channels = in_channels_skip_connection * 2
        conv_blocks = []

        conv_blocks.append(
            ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                normalization=normalization,
                kernal_size=kernal_size,
                padding_mode=padding_mode,
                activation=activation,
                use_bias=use_bias,
            )
        )

        for _ in range(conv_num - 1):
            conv_blocks.append(
                ConvolutionalBlock(
                    dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    normalization=normalization,
                    kernal_size=kernal_size,
                    padding_mode=padding_mode,
                    activation=activation,
                    use_bias=use_bias,
                )
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)

        if self.residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernal_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection: Tensor, x: Tensor):
        x = self.upsample(x)
        # print(f"x from the upsample shape: {x.shape}")
        # if self.all_size_input:
        #     x = self.crop(x, skip_connection)  # crop x according skip_connection
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)

        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += connection
        else:
            x = self.conv_blocks(x)
        return x

    def crop(self, x: Tensor, skip: Tensor) -> Tensor:
        # Code is from
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py#L57
        # but change 2D to 3D

        diffT = skip.size()[2] - x.size()[2]
        diffH = skip.size()[3] - x.size()[3]
        diffW = skip.size()[4] - x.size()[4]

        if self.num_block % 2 == 0:
            x = F.pad(
                x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffT // 2, diffT - diffT // 2]
            )
        else:
            x = F.pad(
                x, [diffW - diffW // 2, diffW // 2, diffH - diffH // 2, diffH // 2, diffT - diffT // 2, diffT // 2]
            )
        return x


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = 'Upsampling type is "{}"' " but should be one of the following: {}"
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    # Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
    return nn.Upsample(scale_factor=2, mode=upsampling_type)


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = "ConvTranspose{}d".format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == "linear":
        if dimensions == 2:
            upsampling_type = "bilinear"
        elif dimensions == 3:
            upsampling_type = "trilinear"
    return upsampling_type
