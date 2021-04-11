# Some code is adapted from: https://github.com/fepegar/unet/tree/master/unet

"""Main module."""

from typing import Optional, List
import torch
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dimensions: int,
        out_channels_first_layer: int,
        conv_num_in_layer: List[int],
        kernal_size: int,
        normalization: str,
        downsampling_type: str,
        residual: bool,
        padding_mode: str,
        activation: Optional[str],
        upsampling_type: str = "conv",
        use_bias: bool = True,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            residual=residual,
            kernal_size=kernal_size,
            normalization=normalization,
            downsampling_type=downsampling_type,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        in_channels = self.encoder.out_channels
        in_channels_skip_connection = in_channels
        self.bottom_block = EncodingBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            dimensions=dimensions,
            conv_num=conv_num_in_layer[-1],
            residual=residual,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
            downsampling_type=None,
            num_block=len(conv_num_in_layer),
            use_bias=use_bias,
        )

        conv_num_in_layer.reverse()
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type=upsampling_type,
            conv_num_in_layer=conv_num_in_layer[1:],
            kernal_size=kernal_size,
            residual=residual,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        in_channels = self.decoder.out_channels
        self.classifier = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels=out_classes,
            kernal_size=1,
            activation=None,
            normalization=None,
            use_bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, encoding = self.encoder(x)
        x = self.bottom_block(encoding)
        x = self.decoder(skip_connections, x)
        x = self.classifier(x)
        if self.use_sigmoid:
            return self.sigmoid(x)
        else:
            return x


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 2
        kwargs["num_encoding_blocks"] = 5
        kwargs["out_channels_first_layer"] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 3
        kwargs["num_encoding_blocks"] = 3  # 4
        kwargs["out_channels_first_layer"] = 8
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class VNet(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 3
        kwargs["num_encoding_blocks"] = 4  # 4
        kwargs["out_channels_first_layer"] = 16
        kwargs["kernal_size"] = 5
        kwargs["residual"] = True
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
