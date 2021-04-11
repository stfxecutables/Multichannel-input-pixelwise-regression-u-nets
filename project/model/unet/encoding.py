# Code is adapted from: https://github.com/fepegar/unet/tree/master/unet

from typing import Optional, List
import torch.nn as nn
from .conv import ConvolutionalBlock
import torch.nn.functional as F


class Encoder(nn.Module):
    # Code is adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py#L6
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        conv_num_in_layer: List[int],
        residual: bool,
        kernal_size: int,
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
                kernal_size=kernal_size,
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

    def forward(self, x):
        # Code is adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py#L55
        skip_connections = []
        # nn.ModuleList need to iterate!!!!
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x


class EncodingBlock(nn.Module):
    # Code is adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py#L67
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int,
        residual: bool,
        normalization: Optional[str],
        conv_num: int,
        num_block: int,
        kernal_size: int = 5,
        downsampling_type: Optional[str] = "conv",
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.num_block = num_block
        self.residual = residual
        conv_blocks = []

        conv_blocks.append(
            ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels,
                normalization=normalization,
                kernal_size=kernal_size,
                padding_mode=padding_mode,
                activation=activation,
                use_bias=use_bias,
            )
        )

        for idx in range(conv_num - 1):
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

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernal_size=1,
                normalization=None,
                activation=None,
            )

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            self.downsample = get_downsampling_maxpooling_layer(dimensions, downsampling_type)
        elif downsampling_type == "conv":
            self.downsample = get_downsampling_conv_layer(in_channels=out_channels)

        self.out_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        # Code is adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py#L141
        if self.residual:
            residual_layer = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += residual_layer
        else:
            x = self.conv_blocks(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
        return x, skip_connection


def get_downsampling_maxpooling_layer(
    dimensions: int,
    pooling_type: str,
    kernel_size: int = 2,
    stride: int = 2,
):
# Code is adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py#L162
    class_name = "{}Pool{}d".format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


def get_downsampling_conv_layer(
    in_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
):
    class_name = "Conv3d"
    class_ = getattr(nn, class_name)
    return class_(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride)
