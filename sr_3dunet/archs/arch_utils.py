
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
import functools
import math

# ==========
# normaliz layer
# ==========
def get_norm_layer(norm_type='instance', dim=2):
    if dim == 2:
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    elif dim == 3:
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    else:
        raise Exception('Invalid dim.')

    if norm_type == 'batch':
        norm_layer = functools.partial(BatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm, affine=False, track_running_stats=False)
    elif norm_type == 'identity':
        def norm_layer(x):
            return lambda t:t
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# ==========
# UNet
# ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type=None, dim=3):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        if norm_type is not None:
            norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        conv_layers = []
        conv_layers.append(Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        if norm_type is not None:
            conv_layers.append(norm_layer(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        if norm_type is not None:
            conv_layers.append(norm_layer(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)
        return x