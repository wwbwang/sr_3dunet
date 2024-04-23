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
from ..utils.data_utils import random_crop_3d, random_crop_2d, augment_3d_rotated, augment_2d, preprocess, get_projection, merge_img


@DATASET_REGISTRY.register()
class Paired_tif_Dataset(data.Dataset):

    def __init__(self, opt):
        super(Paired_tif_Dataset, self).__init__()
        self.opt = opt
        self.cube_keys = []
        self.rotated_cube_keys = []
        
        self.datasets_cube = opt['datasets_cube']
        self.datasets_rotated_cube = opt['datasets_rotated_cube']
        self.iso_dimension = opt['iso_dimension']
        self.aniso_dimension = opt['aniso_dimension']
        self.gt_size = opt['gt_size']
        self.gt_probs = opt['gt_probs']
        self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
        self.mean = opt['mean']
        self.min_value = opt['min_value']
        self.max_value = opt['max_value']
        self.percentiles = opt['percentiles']
        self.threshold_percentiles_cube = opt['threshold_percentiles_cube']
        self.threshold_percentiles_MIP = opt['threshold_percentiles_MIP']
        self.threshold = opt['threshold']
        self.add_syn_times = opt['add_syn_times']
        self.screen_flag = opt['screen_flag']
        self.logger = get_root_logger()

        img_names = os.listdir(self.datasets_rotated_cube)
        for img_name in img_names:
            self.cube_keys.append(osp.join(self.datasets_cube, img_name))
            self.rotated_cube_keys.append(osp.join(self.datasets_rotated_cube, img_name))

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
        
        index_list = random.sample(range(len(self.rotated_cube_keys)), self.add_syn_times)

        while(True):
            img_cube_name_list = [self.cube_keys[idx] for idx in index_list]
            img_rotated_cube_name_list = [self.rotated_cube_keys[idx] for idx in index_list]
            
            img_cube_list = [tifffile.imread(img_cube_name) for img_cube_name in img_cube_name_list]
            img_cube = merge_img(img_cube_list)
            img_cube = np.clip(img_cube, self.min_value, self.max_value)
            img_cube = random_crop_3d(img_cube, self.gt_size)
            if self.screen_flag:
                if  preprocess(img_cube, self.threshold_percentiles_cube, self.mean)[2] < self.threshold:
                    index_list = random.sample(range(len(self.rotated_cube_keys)), self.add_syn_times)
                    continue
            img_cube, min_value, max_value = preprocess(img_cube, self.percentiles, self.mean) # FIXME 还可能是norm的问题
           
            img_rotated_cube_list = [tifffile.imread(img_rotated_cube_name) for img_rotated_cube_name in img_rotated_cube_name_list]
            img_rotated_cube = merge_img(img_rotated_cube_list)
            img_rotated_cube = random_crop_3d(img_rotated_cube, self.gt_size)
            img_rotated_cube = np.clip(img_rotated_cube, self.min_value, self.max_value)
            img_MIP, _, _ = get_projection(img_rotated_cube, self.iso_dimension)
            if self.screen_flag:
                if preprocess(img_MIP, self.threshold_percentiles_MIP, self.mean)[2] < self.threshold:
                    index_list = random.sample(range(len(self.rotated_cube_keys)), self.add_syn_times)
                    continue
            img_MIP = (img_MIP-min_value)/(max_value-min_value)
            img_MIP = img_MIP - self.mean     
            
            # img_cube = augment_3d_rotated(img_cube, self.aniso_dimension, 
            #                             self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
            img_MIP = augment_2d(img_MIP,  self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
            

            break
        
        return {'img_cube': img_cube[None, ].astype(np.float32),
                'img_MIP': img_MIP[None, ].astype(np.float32),
                # 'img_rotated_cube': img_rotated_cube[None, ].astype(np.float32),
                'img_name': img_cube_name_list[0]}
    

    def __len__(self):
        return len(self.rotated_cube_keys)

