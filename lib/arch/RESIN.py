import torch
import torch.nn as nn
import random

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())
from lib.arch.base_model import UNetGenerator, CubeDiscriminator

def define_G(in_channels, out_channels, features, norm_type=None, *, dim=3):
    net_G = UNetGenerator(in_channels, out_channels, features, norm_type=norm_type, dim=dim)
    return net_G

def define_D(in_channels, features, norm_type=None, *, dim=3):
    net_D = CubeDiscriminator(in_channels, features, norm_type=norm_type, dim=dim)
    return net_D

class RESIN(nn.Module):
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
        fake_B_T = self.transpose(fake_B, dim0=dim0)
        rec_A2 = self.G_B(fake_B_T)

        return fake_B, rec_A1, fake_B_T, rec_A2
    
    def transpose(self, img:torch.Tensor, dim0=-2, dim1=None):
        if dim1 is None:
            dim_list = [-1, -2, -3]
            dim_list.remove(dim0)
            dim1 = random.choice(dim_list)
        img = img.transpose(dim0, dim1)
        return img

def get_model(args):
    model = RESIN(in_channels=1, out_channels=1, 
                  features_G=args.features_G, features_D=args.features_D, norm_type=args.norm_type,
                  aniso_dim=args.aniso_dim, iso_dim=args.iso_dim)
    return model

if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cuda:0')
    model = RESIN(1, 1, [64,128,256], [64,128,256], norm_type=None)
    model.to(device)
    size = 64
    real_A = torch.rand(1,1,size,size,size).to(device)
    summary(model, (1,1,size,size,size))
    fake_B, rec_A1, fake_B_T, rec_A2 = model(real_A)