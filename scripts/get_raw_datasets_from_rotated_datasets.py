import tifffile
import numpy as np
import torch
import h5py
import os
import random
from tqdm import tqdm
import cv2
import math

from sr_3dunet.utils.data_utils import get_projection, get_anti_rotated_img, crop_block

def img_loader(img, start_x, start_y, start_z, size):
    return img[start_x:start_x+size, start_y:start_y+size, start_z:start_z+size]

def rotated_img2raw_img(img):
    # transpose TODO
    img = img
    # affine
    img = get_anti_rotated_img(img, device=None, aniso_dimension=-2)
    
    img = img[img.shape[0]//2-new_size//2:img.shape[0]//2+new_size//2, :, img.shape[2]//2-new_size//2:img.shape[2]//2+new_size//2]
    img = np.flip(img, axis=-1)
    return img

size = 128
new_size = 64

input_path = '/share/home/wangwb/workspace/sr_3dunet/datasets/Monkey_Brain/rotated_datasets'
output_path = '/share/home/wangwb/workspace/sr_3dunet/datasets/Monkey_Brain/Raw_datasets'

img_name_list = os.listdir(input_path)

for index in tqdm(range(0, len(img_name_list)), desc="1st loop"):
    img_name = img_name_list[index]
    input_img_path = os.path.join(input_path, img_name)
    output_img_path = os.path.join(output_path, img_name)
    
    img = tifffile.imread(input_img_path)
    out_img = rotated_img2raw_img(img)
    
    tifffile.imwrite(output_img_path, out_img)