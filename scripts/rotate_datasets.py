import tifffile
import numpy as np
import torch
import os
import random

from sr_3dunet.utils.data_utils import random_crop_3d, random_crop_2d, augment_3d, augment_2d, preprocess, get_projection, get_rotated_img, crop_block


input_folder = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_ori" # /home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_ori"
output_folder = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_extend_rotated"
croped_output_folder = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_croped_rotated"
aniso_dimension = -2

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(croped_output_folder, exist_ok=True)
    
def crop_from_rotated(img):
    img = img[:, img.shape[1]//4-1 : img.shape[1]//4*3+1, img.shape[2]//4-1 : img.shape[2]//4*3+1]
    img = crop_block(img, step_size=16, dim=3)
    return img
    
img_name_list = os.listdir(input_folder)
for img_name in img_name_list:
    input_path = os.path.join(input_folder, img_name)
    input_img = tifffile.imread(input_path)
    
    output_img = get_rotated_img(input_img, aniso_dimension)
    tifffile.imsave(os.path.join(output_folder, "rotated_"+img_name), output_img)
    
    croped_output_img = crop_from_rotated(output_img)
    tifffile.imsave(os.path.join(croped_output_folder, "croped_rotated_"+img_name), croped_output_img)