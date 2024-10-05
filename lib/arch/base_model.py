import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
import functools
import math

# ==========
# norm
# ==========
def get_norm_layer(norm_type:str, dim=2):
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
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# ==========
# Conv
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

# ==========
# Discriminator
# ==========
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

# ==========
# Generator
# ==========
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256], *, norm_type='batch', dim=3):
        super(UNetGenerator, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            ConvTranspose = nn.ConvTranspose3d
            self.MaxPool = nn.MaxPool3d
        else:
            raise Exception('Invalid dim.')
            
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, norm_type=norm_type, dim=dim))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.Sequential(
                    upsample, 
                    Conv(feature*2, feature, kernel_size=1, stride=1)
                )
                # ConvTranspose(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = Conv(features[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        input = x
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = self.MaxPool(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)
        x = self.final_conv(x)
        return x
    
if __name__ == '__main__':
    model = UNetGenerator(norm_type=None)
    device = torch.device('cuda:0')
    data_in = torch.rand((1,1,64,64,64))
    out = model(data_in)
    print(out.shape)