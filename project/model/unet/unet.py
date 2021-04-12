"""
The basic organization of the U-Net code into convolutional (conv.py), decoding (decoding.py), and
encoding (encoding.py) components here is drawn heavily from Fernando Perez-Garcia's excellent U-Net
code.

Perez-Garcia, F. (2019). fepegar/unet: First published version of PyTorch U-Net (v0.6.4)
[Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.3522306. Repository at
https://github.com/fepegar/unet/tree/master/unet

The code specifically in this `unet.py` file is based on
https://github.com/fepegar/unet/blob/master/unet/unet.py
"""

from typing import Any, Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from .conv import ConvolutionalBlock
from .decoding import Decoder
from .encoding import Encoder, EncodingBlock


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dimensions: int,
        out_channels_first_layer: int,
        conv_num_in_layer: List[int],
        kernel_size: int,
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
        shared_options: Dict = dict(
            residual=residual,
            kernel_size=kernel_size,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            downsampling_type=downsampling_type,
            **shared_options,
        )

        in_channels = self.encoder.out_channels
        in_channels_skip_connection = in_channels
        self.joiner = EncodingBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            dimensions=dimensions,
            conv_num=conv_num_in_layer[-1],
            num_block=len(conv_num_in_layer),
            downsampling_type=None,
            **shared_options,
        )

        conv_num_in_layer.reverse()
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type=upsampling_type,
            conv_num_in_layer=conv_num_in_layer[1:],
            **shared_options,
        )

        in_channels = self.decoder.out_channels
        self.output = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels=out_classes,
            kernel_size=1,
            activation=None,
            normalization=None,
            use_bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        skips, encoding = self.encoder(x)
        x = self.joiner(encoding)
        x = self.decoder(skips, x)
        x = self.output(x)
        if self.use_sigmoid:
            return self.sigmoid(x)  # type: ignore
        else:
            return x


class UNet2D(UNet):
    def __init__(self, *args: Any, **user_kwargs: Any):
        kwargs: Dict = dict(dimensions=2, num_encoding_blocks=5, out_channels_first_layer=64)
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args: Any, **user_kwargs: Any):
        kwargs: Dict = dict(dimensions=3, num_encoding_blocks=3, out_channels_first_layer=8)
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class VNet(UNet):
    def __init__(self, *args: Any, **user_kwargs: Any):
        kwargs: Dict = dict(
            dimensions=3,
            num_encoding_blocks=4,
            out_channels_first_layer=16,
            kernel_size=5,
            residual=True,
        )
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
