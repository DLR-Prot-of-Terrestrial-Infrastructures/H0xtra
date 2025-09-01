# partially adapted from https://github.com/karpathy/nanoGPT, (copyright (c) 2022 Andrej Karpathy), and https://github.com/milesial/Pytorch-UNet

"""
This file contains the pytorch model for different hypernetwork architectures
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.

        Copyright (c) 2020 Peter Tatkowski, source: https://github.com/tatp22/multidim-positional-encoding
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        
        """if self.cached_penc is not None and self.cached_penc.shape[0] != tensor.shape[0] and self.cached_penc.shape[1:] == tensor.shape[1:]:
            return self.cached_penc[0]"""

        self.cached_penc = None
        batch_size, orig_ch, x, y = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1).permute(0,3,1,2).contiguous()
        return self.cached_penc
    
    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, d_embed, mid_channels=None, stride=2, padding=0):
        super().__init__()

        if not mid_channels:
            mid_channels = d_embed

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_embed),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, d_embed, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_embed),
            nn.ReLU(inplace=True)
        )

    def forward(self, matrix):
        return self.double_conv(matrix)
    

class Down(nn.Module):
    def __init__(self, in_channels, d_embed):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, d_embed))

    def forward(self, matrix):
        return self.conv_layer(matrix)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            #self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)
        x = x1+x2
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_layers, in_channels, d_embed, bilinear=False, coords='s'):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.d_embed = d_embed
        self.bilinear = bilinear
        self.n_layers = n_layers
        self.pos_enc = (coords == 's')

        if self.pos_enc:
            self.in_channels = 2 * self.in_channels
        elif coords == 'c':
            self.in_channels += 2

        if self.pos_enc:
            self.pe = PositionalEncoding2D(in_channels)

        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(self.in_channels, self.d_embed))
        self.decoder = nn.ModuleList()

        self.out_layer = OutConv(d_embed, d_embed)

    
        for i in range(n_layers-1):
            self.encoder.append(Down(d_embed, d_embed))
            self.decoder.append(Up(d_embed, d_embed, self.bilinear))

    def forward(self, matrix):
        buffer = []
        if self.pos_enc:
            for i, layer in enumerate(self.encoder):
                if i == 0:
                    matrix = layer(torch.cat((matrix, self.pe(matrix)), dim=1))
                else:
                    matrix = layer(matrix)
                buffer.append(matrix)
        else:
            for layer in self.encoder:
                matrix = layer(matrix)
                buffer.append(matrix)
        
        out = buffer[-1]

        for k in range(1, self.n_layers):
            out = self.decoder[-k](out, buffer[-k-1])
            
        out = self.out_layer(out)
        return out
    

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channels))
        
    def forward(self, matrix):
        return self.block(matrix)
    
class BasicCNN(nn.Module):
    def __init__(self, n_layers, in_channels, d_embed, kernel_size):
        super().__init__()

        self.n_layers = n_layers

        self.encoder = nn.Sequential()
        self.encoder.append(CNNBlock(in_channels, d_embed, kernel_size, padding='same'))

        for _ in range(n_layers-1):
            self.encoder.append(CNNBlock(d_embed, d_embed, kernel_size, padding='same'))


    def forward(self, matrix):

        out = self.encoder(matrix)

        return out
    
class HyperMLP(nn.Module):
    def __init__(self, n_layers, in_channels, d_embed):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(in_channels, d_embed), nn.ReLU(), nn.Linear(d_embed, d_embed))
        
        for _ in range(n_layers-1):
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Linear(d_embed, d_embed))

    def forward(self, x):
        return self.encoder(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
    
HyperDict = {}

@dataclass
class HyperNetworkConfig:
    n_layers: int = 4
    in_channels: int = 4
    d_embed: int = 32
    kernel_size: int = 4
    hypernetwork: str = 'unet'
    contr_loss: bool = True
    crop_by_day: bool = False
    coords: str = 's'


class HyperNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if self.config.hypernetwork == 'unet':
            self.block = UNet(n_layers=config.n_layers,
                              in_channels=config.in_channels,
                              d_embed=config.d_embed,
                              coords = config.coords)
        elif self.config.hypernetwork == 'cnn':
            self.block = BasicCNN(n_layers=config.n_layers,
                                  in_channels=config.in_channels,
                                  d_embed=config.d_embed,
                                  kernel_size=config.kernel_size)
        elif self.config.hypernetwork == 'mlp':
            self.block = HyperMLP(n_layers=config.n_layers,
                              in_channels=config.in_channels,
                              d_embed=config.d_embed,)
            
        self.LN = LayerNorm(config.d_embed, bias=False)


    def forward(self, matrix):

        output = self.block(matrix)
        output = self.LN(torch.flatten(output, start_dim=2).permute(0,2,1).contiguous())

        return output