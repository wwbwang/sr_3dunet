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
from ..utils.data_utils import random_crop_3d, random_crop_2d, augment_3d, preprocess


@DATASET_REGISTRY.register()
class CycleGAN_2D_Dataset(data.Dataset):

    def __init__(self, opt):
        super(CycleGAN_2D_Dataset, self).__init__()
        self.opt = opt
        self.keys = []
        self.gt_root = opt['dataroot_gt']
        self.dataroot_rotate = opt['dataroot_rotate']
        self.gt_size = opt['gt_size']
        self.gt_probs = opt['gt_probs']
        self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
        self.mean = opt['mean']
        self.percentiles = opt['percentiles']
        logger = get_root_logger()

        img_names = os.listdir(self.gt_root)
        for img_name in img_names:
            self.keys.append(img_name)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        img_name_1 = os.path.join(self.gt_root, self.keys[index])
        img_name_2 = os.path.join(self.dataroot_rotate, self.keys[index])
        
        img_1 = tifffile.imread(img_name_1)
        img_2 = tifffile.imread(img_name_2)
        
        # random crop
        img_1 = random_crop_2d(img_1, self.gt_size)
        img_2 = random_crop_2d(img_2, self.gt_size)
        
        # augmentation
        # img_1 = augment_3d(img_1, self.iso_dimension, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        # img_2 = augment_3d(img_2, self.iso_dimension, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
        # preprocess # by liuy
        img_1, _, _ = preprocess(img_1, self.percentiles, self.mean)
        img_2, _, _ = preprocess(img_2, self.percentiles, self.mean)
        
        return img_1[None, ].astype(np.float32), img_2[None, ].astype(np.float32)

    def __len__(self):
        return len(self.keys)
