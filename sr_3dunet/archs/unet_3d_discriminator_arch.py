import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
import functools
import math
from sr_3dunet.archs.arch_utils import DoubleConv, get_norm_layer

from basicsr.archs.arch_util import ResidualBlockNoBN, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY
     
@ARCH_REGISTRY.register()
class CubeDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], norm_type='batch', *, dim=3):
        super(CubeDiscriminator, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        if norm_type is not None:
            norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        kw = 4
        padw = 1
        sequence = [Conv(in_channels, features[0], kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        n_layers = len(features)
        for i in range(1, n_layers-1):
            sequence.append(Conv(features[i-1], features[i], kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            if norm_type is not None:
                sequence.append(norm_layer(features[i]))
            sequence.append(nn.LeakyReLU(0.2, True))
        
        sequence.append(Conv(features[-2], features[-1], kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        if norm_type is not None:
            sequence.append(norm_layer(features[-1]))
        sequence.append(nn.LeakyReLU(0.2, True))

        sequence += [Conv(features[-1], 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)