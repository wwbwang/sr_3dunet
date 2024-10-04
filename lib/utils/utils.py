import torch
import math
import random

def affine_sample(img:torch.Tensor, angel=-45):
    b,c,d,h,w = img.shape
    angel = math.radians(angel)
    theta = torch.tensor(
                [[math.cos(angel), 0, math.sin(angel), 0],
                [0, 1, 0, 0],
                [-math.sin(angel), 0, math.cos(angel), 0]], 
                dtype=torch.float, device=img.device)
    theta = theta.repeat(b,*[1,]*len(theta.shape))
    grid = torch.nn.functional.affine_grid(theta, img.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(img, grid=grid, align_corners=False)
    return out

def get_slant_mip(img:torch.Tensor, angel=-45, iso_dim=-1, augment=True):
    img_affine = affine_sample(img, angel=angel)
    mip = torch.max(img_affine, dim=iso_dim).values
    if augment and random.random() < 0.5:
        mip = torch.flip(mip, dims=[-1])
    if augment and random.random() < 0.5:
        mip = torch.flip(mip, dims=[-2])
    if augment and random.random() < 0.5:
        mip = mip.transpose(-1,-2)
    return mip