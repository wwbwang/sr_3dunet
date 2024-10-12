import torch
import torch.nn as nn
import random
import functools

# ==========
# base mdoel
# ==========

# === norm ====
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

# === residual conv ====
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type=None, dim=3):
        super(ResConv, self).__init__()

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
        conv_layers.append(norm_layer(out_channels)) if norm_type else None
        conv_layers.append(nn.ReLU())
        conv_layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        conv_layers.append(norm_layer(out_channels)) if norm_type else None
        self.conv = nn.Sequential(*conv_layers)

        if in_channels != out_channels:
            self.conv1x1 = Conv(in_channels, out_channels, kernel_size=1, bias=use_bias)
        else:
            self.conv1x1 = None
        self.final_act = nn.ReLU()

    def forward(self, x):
        input = x
        x = self.conv(x)
        if self.conv1x1:
            input = self.conv1x1(input)
        x = x + input
        x = self.final_act(x)
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
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.MaxPool = nn.MaxPool3d
        else:
            raise Exception('Invalid dim.')
            
        # Encoder
        for feature in features:
            self.downs.append(ResConv(in_channels, feature, norm_type=norm_type, dim=dim))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.Sequential(
                    upsample, 
                    Conv(feature*2, feature, kernel_size=1, stride=1),
                    nn.ReLU()
                )
            )
            self.ups.append(ResConv(feature*2, feature, norm_type=norm_type, dim=dim))

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
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)
        x = self.final_conv(x)
        x = x + input
        return x


def define_G(in_channels, out_channels, features, norm_type=None, *, dim=3):
    net_G = UNetGenerator(in_channels, out_channels, features, norm_type=norm_type, dim=dim)
    return net_G

def define_D(in_channels, features, norm_type=None, *, dim=3):
    net_D = CubeDiscriminator(in_channels, features, norm_type=norm_type, dim=dim)
    return net_D

class RESIN_base_subtract(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, 
                 features_G=[64,128,256], features_D=[64,128,256], norm_type=None,
                 aniso_dim=-2, iso_dim=-1) -> None:
        super().__init__()
        self.aniso_dim = aniso_dim
        self.iso_dim = iso_dim

        # Generator
        self.G_A = define_G(in_channels, out_channels, features_G, norm_type=norm_type, dim=3)
        self.G_B = define_G(in_channels, out_channels, features_G, norm_type=norm_type, dim=3)
        # MIP Discriminator
        self.D_AnisoMIP = define_D(in_channels, features_D, norm_type=norm_type, dim=2)
        self.D_IsoMIP_1 = define_D(in_channels, features_D, norm_type=norm_type, dim=2)
        self.D_IsoMIP_2 = define_D(in_channels, features_D, norm_type=norm_type, dim=2)
        # Cube Discriminator
        self.D_RecA_1 = define_D(in_channels, features_D, norm_type=norm_type, dim=3)
        self.D_RecA_2 = define_D(in_channels, features_D, norm_type=norm_type, dim=3)

    def forward(self, real_A):
        fake_B = self.G_A(real_A)
        rec_A1 = self.G_B(fake_B)

        dim0 = self.aniso_dim
        fake_B_T, dim0, dim1 = self.transpose(fake_B, dim0=dim0)
        rec_A2 = self.G_B(fake_B_T)

        with torch.no_grad():
            rec_A3 = self.G_B(self.transpose(self.G_A(fake_B_T), dim0, dim1)[0])

        return fake_B, rec_A1, fake_B_T, rec_A2, rec_A3
    
    def transpose(self, img:torch.Tensor, dim0=-2, dim1=None):
        if dim1 is None:
            dim_list = [-1, -2, -3]
            dim_list.remove(dim0)
            dim1 = random.choice(dim_list)
        img = img.transpose(dim0, dim1)
        return img, dim1, dim0

def get_model(args):
    model = RESIN_base_subtract(in_channels=1, out_channels=1, 
                  features_G=args.features_G, features_D=args.features_D, norm_type=args.norm_type,
                  aniso_dim=args.aniso_dim, iso_dim=args.iso_dim)
    return model

if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cuda:0')
    size = 64
    real_A = torch.rand(1,1,size,size,size).to(device)

    model = RESIN_base_subtract(1, 1, [64,128,256], [64,128,256], norm_type=None)
    model.to(device)
    summary(model, (1,1,size,size,size))
    fake_B, rec_A1, fake_B_T, rec_A2 = model(real_A)

    # model = UNetGenerator(norm_type=None)
    # model.to(device)
    # summary(model, (1,1,size,size,size))