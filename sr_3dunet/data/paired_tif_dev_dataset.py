import numpy as np
import os
import random
import torch
from os import path as osp
from torch.utils import data as data
import tifffile
import math

from basicsr.utils import FileClient, get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Paired_Tif_dev_Dataset(data.Dataset):

    def __init__(self, opt):
        super(Paired_Tif_dev_Dataset, self).__init__()
        self.opt = opt
        self.cube_keys = []
        
        self.datasets_cube = opt['datasets_cube']
        self.aniso_dimension = opt['aniso_dimension']
        self.iso_dimension = opt['iso_dimension']
        self.gt_size = opt['gt_size']
        self.gt_probs = opt['gt_probs']
        self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
        self.mean = opt['mean']
        self.min_value = opt['min_value']
        self.max_value = opt['max_value']
        self.percentiles = opt['percentiles']
        self.logger = get_root_logger()

        img_names = os.listdir(self.datasets_cube)
        for img_name in img_names:
            self.cube_keys.append(osp.join(self.datasets_cube, img_name))

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        self.logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        
        self.device = torch.device(torch.cuda.current_device())

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        img_cube_name = self.cube_keys[index]
        img_cube = tifffile.imread(img_cube_name).astype(np.float32)
        # d,h,w
        img_cube = torch.from_numpy(img_cube).to(self.device)
        if self.gt_size < img_cube.shape[-1]:
            img_cube = random_crop_3d(img_cube, self.gt_size)
        img_cube, min_value, max_value = preprocess(img_cube, self.percentiles, self.mean)
        
        img_MIP = get_45d_projection(img_cube, self.iso_dimension)
        img_cube = augment_3d(img_cube, self.aniso_dimension, 
                              self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        img_MIP = augment_2d(img_MIP, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
        return {'img_cube': img_cube[None, ].cpu(),
                'img_MIP': img_MIP[None, ].cpu(),
                'img_name': ''}
    
    def __len__(self):
        return len(self.cube_keys)

def random_crop_3d(img:torch.Tensor, patch_size, start_d=None, start_h=None, start_w=None):
    d, h, w= img.size()[-3:]
    # randomly choose top and left coordinates
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)
    if start_d is None:
        start_d = random.randint(0, d - patch_size)
    img = img[..., start_d:start_d + patch_size, start_h:start_h + patch_size, start_w:start_w + patch_size]
    return img

def preprocess(img:torch.Tensor, percentiles, dataset_mean):
    flattened_arr = torch.sort(img.flatten())[0]
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = torch.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

    min_value = torch.min(clipped_arr)
    max_value = torch.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    img = img - dataset_mean
    return img, min_value, max_value

def get_45d_projection(img:torch.Tensor, mip_dim) -> torch.Tensor:
    shape = img.shape
    assert len(shape) == 3 or len(shape) == 5, 'invalid img tensor shape'
    if len(img.shape) != 5:
        img = img[None,None]
    angel = -45
    angel = math.radians(angel)
    theta = torch.tensor(
                [[math.cos(angel), 0, math.sin(angel), 0],
                [0, 1, 0, 0],
                [-math.sin(angel), 0, math.cos(angel), 0]], 
                dtype=torch.float, device=img.device)
    size = img.shape[-1]
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), (1,1,size,size,size), align_corners=False).to(img.device)
    img_affine = torch.nn.functional.grid_sample(img, grid=grid, align_corners=False)

    mip = torch.max(img_affine, dim=mip_dim)[0]
    return mip[0,0]

def augment_2d(img:torch.Tensor, hflip=True, vflip=True, rotation=True) -> torch.Tensor:
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    if hflip:  # horizontal
        img = torch.flip(img, dims=[-1])
    if vflip:  # vertical
        img = torch.flip(img, dims=[-2])
    if rot90:
        img = img.transpose(1, 0)
    return img

def augment_3d(img:torch.Tensor, halfiso_dim, flip_aniso=True, flip_halfiso=True, transpose_halfiso=True) -> torch.Tensor:
    if flip_aniso and random.random() < 0.5: 
        img = torch.flip(img, dims=[-3])
        img = torch.flip(img, dims=[-1])
    if flip_halfiso and random.random() < 0.5:
        img = torch.flip(img, dims=[-2])
    if transpose_halfiso and halfiso_dim==-2 and random.random() < 0.5:
        img = img.transpose(2, 0)
    return img

if __name__ == '__main__':
    opt = {
        'datasets_cube': '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/fg_blocks',
        'aniso_dimension': -2,
        'iso_dimension': -1,
        'gt_size': [64,32],
        'gt_probs': [1,0],
        'mean': 0,
        'max_value': 65535,
        'min_value': 0,
        'percentiles': [0,1],
        'use_flip': True,
        'use_rot': True,
        'io_backend': {'type': 'disk'}
    }
    from ryu_pytools import arr_info
    ds = Paired_Tif_dev_Dataset(opt)
    print(len(ds))
    out = ds.__getitem__(np.random.randint(len(ds)))
    cube = out['img_cube']
    arr_info(cube)
    cube = cube[0].cpu().numpy()
    mip = out['img_MIP']
    arr_info(mip)
    mip = mip[0].cpu().numpy()
    import napari
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(cube)
    viewer.add_image(mip)
    napari.run()