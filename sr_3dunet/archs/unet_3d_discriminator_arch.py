import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
import functools
import math

from basicsr.archs.arch_util import ResidualBlockNoBN, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY

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
# Discriminator
# ==========
@ARCH_REGISTRY.register()
class ProjectionDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], norm_type='batch', *, dim=2):
        super(ProjectionDiscriminator, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        kw = 4
        padw = 1
        sequence = [Conv(in_channels, features[0], kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        n_layers = len(features)
        for i in range(1, n_layers-1):
            sequence += [
                Conv(features[i-1], features[i], kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(features[i]),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [
            Conv(features[-2], features[-1], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(features[-1]),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [Conv(features[-1], 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
    
@ARCH_REGISTRY.register()
class ClassDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], norm_type='batch', *, dim=2):
        super(ClassDiscriminator, self).__init__()
        self.dim = dim

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        kw = 4
        padw = 1
        sequence = [Conv(in_channels, features[0], kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        n_layers = len(features)
        for i in range(1, n_layers-1):
            sequence += [
                Conv(features[i-1], features[i], kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(features[i]),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [
            Conv(features[-2], features[-1], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(features[-1]),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [Conv(features[-1], 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        # aux-classifier fc
        if self.dim == 2:
            self.fc_num = 128*1*14*14
        elif self.dim == 3:
            self.fc_num = 128*1*14*14*14
        self.fc_aux = nn.Linear(self.fc_num, 1)

    def forward(self, input):
        output = self.model(input)
        
        class_res = output.view(-1, self.fc_num)
        class_res = self.fc_aux(class_res)
        class_res = nn.Sigmoid()(class_res).view(-1, 1).squeeze(1)
        
        return output, class_res
    

@ARCH_REGISTRY.register()
class CubeDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], norm_type='batch', *, dim=3):
        super(CubeDiscriminator, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        kw = 4
        padw = 1
        sequence = [Conv(in_channels, features[0], kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        n_layers = len(features)
        for i in range(1, n_layers-1):
            sequence += [
                Conv(features[i-1], features[i], kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(features[i]),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [
            Conv(features[-2], features[-1], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(features[-1]),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [Conv(features[-1], 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)