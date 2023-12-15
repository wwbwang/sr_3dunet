import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

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
# ResNet
# ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, *, norm_type='batch', dim=2):
        super().__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
        )

        self.activate = nn.ReLU(inplace=True)
        if use_1x1conv:
            self.conv1x1 = Conv(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1x1 = None

    def forward(self, x):
        y = self.double_conv(x)
        if self.conv1x1:
            x = self.conv1x1(x)
        y += x
        y = self.activate(y)
        return y

class ResNet(nn.Module):
    def  __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], *, norm_type='batch', dim=2):
        super(ResNet, self).__init__()
        if dim == 2:
            Conv=nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')

        res_block = []
        n_layers = len(features)
        for i in range(n_layers):
            if i == 0:
                res_block.append(ResidualBlock(in_channels=in_channels, out_channels=features[i], use_1x1conv=True, norm_type=norm_type, dim=dim))
            else:
                use_1x1conv = features[i]!=features[i-1]
                res_block.append(ResidualBlock(in_channels=features[i-1], out_channels=features[i], use_1x1conv=use_1x1conv, norm_type=norm_type, dim=dim))
        res_block.append(ResidualBlock(in_channels=features[-1], out_channels=out_channels, use_1x1conv=True, norm_type=norm_type, dim=dim))
        self.res_block = nn.Sequential(*res_block)

        self.final_conv = nn.Sequential(
            Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.res_block(x)
        x = self.final_conv(x)
        return x


# ==========
# UNet
# ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type='batch', dim=3):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')

        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

@ARCH_REGISTRY.register()
class UNet_3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], *, norm_type='batch', dim=3):
        super(UNet_3d, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
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
                ConvTranspose(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = Conv(features[0], out_channels, kernel_size=1)

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
        return x + input

# ==========
# Discriminator
# ==========
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], norm_type='batch', *, dim=2):
        super(NLayerDiscriminator, self).__init__()

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

# if __name__ == '__main__':
#     model = UNet(dim=3)
#     # model = ResNet(dim=3)
#     # model = NLayerDiscriminator(features=[64,128,256], dim=3)
#     device = torch.device('cuda:0')
#     model.to(device)
#     from torchsummary import summary
#     summary(model, torch.rand((1,1,16,64,64)))