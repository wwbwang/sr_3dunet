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
from ..utils.data_utils import random_crop_3d, random_crop_2d, augment_3d_aux, augment_2d, preprocess, get_projection, augment_3d


@DATASET_REGISTRY.register()
class Paired_Tif_Aux_Dataset(data.Dataset):

    def __init__(self, opt):
        super(Paired_Tif_Aux_Dataset, self).__init__()
        self.opt = opt
        self.cube_keys = []
        self.aux_cube_keys = []
        
        self.datasets_cube = opt['datasets_cube']
        self.datasets_aux_cube = opt['datasets_aux_cube']
        self.aniso_dimension = opt['aniso_dimension']
        self.iso_dimension = opt['iso_dimension']
        self.gt_size = opt['gt_size']
        self.gt_probs = opt['gt_probs']
        self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
        self.mean = opt['mean']
        self.min_value = opt['min_value']
        self.max_value = opt['max_value']
        self.percentiles = opt['percentiles']
        self.aug3dflag = opt['aug3dflag']
        self.logger = get_root_logger()

        img_names = os.listdir(self.datasets_aux_cube)
        for img_name in img_names:
            self.cube_keys.append(osp.join(self.datasets_cube, img_name))
            self.aux_cube_keys.append(osp.join(self.datasets_aux_cube, img_name))

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
        img_aux_cube_name = self.aux_cube_keys[index]
        
        img_cube = tifffile.imread(img_cube_name)
        img_cube = np.clip(img_cube, self.min_value, self.max_value)
        img_cube = random_crop_3d(img_cube, self.gt_size)
        img_cube, _, _ = preprocess(img_cube, self.percentiles, self.mean)
        
        img_aux_cube = tifffile.imread(img_aux_cube_name)
        img_aux_cube = np.clip(img_aux_cube, self.min_value, self.max_value)
        img_aux_cube = random_crop_3d(img_aux_cube, self.gt_size)
        img_aux_cube, _, _ = preprocess(img_aux_cube, self.percentiles, self.mean)
        img_MIP, _, _ = get_projection(img_aux_cube, self.iso_dimension)
        # start_index = random.randint(0, img_MIP_cube.shape[self.iso_dimension]-self.gt_size-1)        
        # img_MIP, _, _ = get_projection(img_MIP_cube[:,:,start_index:start_index+self.gt_size], self.iso_dimension)
        # img_MIP = random_crop_2d(img_MIP, self.gt_size)
        
        if self.aug3dflag:
            img_cube = augment_3d_aux(img_cube, self.aniso_dimension, 
                                        self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        img_MIP = augment_2d(img_MIP, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
        # img_MIP = (img_MIP-min_value)/(max_value-min_value)
        # img_MIP = img_MIP - self.mean
        
        return {'img_cube': img_cube[None, ].astype(np.float32),
                'img_MIP': img_MIP[None, ].astype(np.float32),
                'img_name': ''}
    
    def __len__(self):
        return len(self.aux_cube_keys)