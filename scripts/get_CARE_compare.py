import tifffile
import numpy as np
import torch
import h5py
import os
import random
from tqdm import tqdm
import cv2
import math

from sr_3dunet.utils.data_utils import get_projection, get_rotated_img, crop_block

filepath = '/share/home/wangwb/workspace/RESIN_datasets/neuron/val_datasets/LGN-V1-ROI1-1um_128_2560_3840.tif'
# filepath = '/share/home/wangwb/workspace/RESIN_datasets/neuron/RESIN_results/outputLGN-V1-ROI1-1um_128_2560_3840.tif'
now_img = tifffile.imread(filepath)
now_img = now_img.transpose(2,1,0)[2:-2, 2:-2, 2:-2]

# 256 256 256 --> 358 256 358
rotated_cube_img = get_rotated_img(now_img, None)
# 358 256 358 --> 174 256 174
rotated_cube_img = rotated_cube_img[rotated_cube_img.shape[0]//4*1+5:rotated_cube_img.shape[0]//4*3-5,
                                    :,
                                    rotated_cube_img.shape[2]//4*1+5:rotated_cube_img.shape[2]//4*3-5]
# save 174 256 174 cube
tifffile.imwrite('datasets/neuron_LGN-V1-ROI/val_withCare_datasets/rotated_img.tif', rotated_cube_img[0:64, 0:64, 0:64])
    


