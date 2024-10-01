import cv2
import ffmpeg
import glob
import numpy as np
import os
import random
import torch
from os import path as osp
from torch.utils import data as data
import tifffile

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from ..utils.data_utils import random_crop_3d, random_crop_2d, augment_3d_rotated, augment_2d, preprocess, get_projection, augment_3d, get_45d_projection

@DATASET_REGISTRY.register()
class Paired_Tif_45d_Dataset(data.Dataset):

    def __init__(self, opt):
        super(Paired_Tif_45d_Dataset, self).__init__()
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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        img_cube_name = self.cube_keys[index]
        
        img_cube = tifffile.imread(img_cube_name).astype(np.float32)
        img_cube = np.clip(img_cube, self.min_value, self.max_value)
        if self.gt_size < img_cube.shape[-1]:
            img_cube = random_crop_3d(img_cube, self.gt_size)
        img_cube, min_value, max_value = preprocess(img_cube, self.percentiles, self.mean)
        
        img_MIP, _, _ = get_45d_projection(img_cube, self.iso_dimension)
        
        img_cube = augment_3d(img_cube, self.iso_dimension, 
                                    self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        img_MIP = augment_2d(img_MIP, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
        return {'img_cube': img_cube[None, ].astype(np.float32),
                'img_MIP': img_MIP[None, ].astype(np.float32),
                'img_name': ''}
    
    def __len__(self):
        return len(self.cube_keys)

# if __name__ == '__main__':
#     opt = {
#         'datasets_cube': '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/Train_datasets',
#         'aniso_dimension': -2,
#         'iso_dimension': -1,
#         'gt_size': [64,32],
#         'gt_probs': [1,0],
#         'mean': 0,
#         'max_value': 65535,
#         'min_value': 0,
#         'percentiles': [0,1],
#         'use_flip': True,
#         'use_rot': True,
#         'io_backend': {'type': 'disk'}
#     }
#     ds = Paired_Tif_45d_Dataset_30(opt)
#     print(len(ds))
#     out = ds.__getitem__(np.random.randint(len(ds)))
#     cube = out['img_cube']
#     mip = out['img_MIP']
#     from ryu_pytools import arr_info
#     arr_info(cube)
#     arr_info(mip)
#     import napari
#     viewer = napari.Viewer(ndisplay=3)
#     viewer.add_image(cube)
#     viewer.add_image(mip)
#     napari.run()