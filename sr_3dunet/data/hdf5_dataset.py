# import cv2
# import ffmpeg
# import glob
# import numpy as np
# import os
# import random
# import torch
# from os import path as osp
# from torch.utils import data as data
# import tifffile

# from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
# from basicsr.utils.registry import DATASET_REGISTRY
# from ..utils.data_utils import random_crop_3d, augment_3d, preprocess, get_projection


# @DATASET_REGISTRY.register()
# class H5_Dataset(data.Dataset):

#     def __init__(self, opt):
#         super(H5_Dataset, self).__init__()
#         self.opt = opt
#         self.keys = []
#         self.gt_root = opt['dataroot_gt']
#         self.iso_dimension = opt['iso_dimension']
#         self.gt_size = opt['gt_size']
#         self.gt_probs = opt['gt_probs']
#         self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
#         self.mean = opt['mean']
#         self.min_value = opt['min_value']
#         self.max_value = opt['max_value']
#         self.percentiles = opt['percentiles']
#         logger = get_root_logger()

#         img_names = os.listdir(self.gt_root)
#         for img_name in img_names:
#             self.keys.append(osp.join(self.gt_root, img_name))

#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.is_lmdb = False

#         # temporal augmentation configs
#         self.interval_list = opt.get('interval_list', [1])
#         self.random_reverse = opt.get('random_reverse', False)
#         interval_str = ','.join(str(x) for x in self.interval_list)
#         logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
#                     f'random reverse is {self.random_reverse}.')

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
#         img_name = self.keys[index]
        
#         img = tifffile.imread(img_name)
        
#         # random crop
#         img = random_crop_3d(img, self.gt_size)
#         # augmentation
#         img = augment_3d(img, self.iso_dimension, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
#         # preprocess
#         img = np.clip(img, self.min_value, self.max_value)
#         img, _, _ = preprocess(img, self.percentiles, self.mean)
        
#         return img[None, ].astype(np.float32)

#     def __len__(self):
#         return len(self.keys)

# @DATASET_REGISTRY.register()
# class MIP2d_Dataset(data.Dataset):

#     def __init__(self, opt):
#         super(MIP2d_Dataset, self).__init__()
#         self.opt = opt
#         self.keys = []
#         self.gt_root = opt['dataroot_gt']
#         self.iso_dimension = opt['iso_dimension']
#         self.gt_size = opt['gt_size'] 
#         self.gt_probs = opt['gt_probs']
#         self.gt_size = random.choices(self.gt_size, self.gt_probs)[0]
#         self.mean = opt['mean']
#         self.min_value = opt['min_value']
#         self.max_value = opt['max_value']
#         self.percentiles = opt['percentiles']
#         logger = get_root_logger()

#         img_names = os.listdir(self.gt_root)
#         for img_name in img_names:
#             self.keys.append(osp.join(self.gt_root, img_name))

#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.is_lmdb = False

#         # temporal augmentation configs
#         self.interval_list = opt.get('interval_list', [1])
#         self.random_reverse = opt.get('random_reverse', False)
#         interval_str = ','.join(str(x) for x in self.interval_list)
#         logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
#                     f'random reverse is {self.random_reverse}.')


#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
#         img_name = self.keys[index]
        
#         img = tifffile.imread(img_name)
        
#         # random crop
#         img = random_crop_3d(img, self.gt_size)
#         # augmentation
#         img = augment_3d(img, self.iso_dimension, self.opt['use_flip'], self.opt['use_flip'], self.opt['use_flip'], self.opt['use_rot'])
        
#         # preprocess
#         img = np.clip(img, self.min_value, self.max_value)
#         img, _, _ = preprocess(img, self.percentiles, self.mean)
        
#         img_iso, img_aniso0, img_aniso1 = get_projection(img, self.iso_dimension)
#         img_aniso = random.choice([img_aniso0, img_aniso1])
        
#         return  {'img_iso': img_iso[None, ].astype(np.float32), 'img_aniso': img_aniso[None, ].astype(np.float32)}

#     def __len__(self):
#         return len(self.keys)