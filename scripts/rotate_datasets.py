import tifffile
import numpy as np
import torch
import os
import random

from sr_3dunet.utils.data_utils import random_crop_3d, random_crop_2d, augment_3d, augment_2d, preprocess, get_projection, get_rotated_img


input_folder = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_test/mask" # /home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_ori"
output_folder = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_test/mask_rotated"

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder+"_iso", exist_ok=True)
    os.makedirs(output_folder+"_aniso0", exist_ok=True)
    os.makedirs(output_folder+"_aniso1", exist_ok=True)
    
img_name_list = os.listdir(input_folder)
for img_name in img_name_list:
    input_path = os.path.join(input_folder, img_name)
    input_img = tifffile.imread(input_path)
    
    output_img = get_rotated_img(input_img, -2)
    tifffile.imsave(os.path.join(output_folder, "rotated_"+img_name), output_img)
    
    # output_img, _, _ = preprocess(output_img, percentiles=[0.01,0.9995], dataset_mean=0.153)
    img_iso, img_aniso0, img_aniso1 = get_projection(output_img, -1)
    tifffile.imsave(os.path.join(output_folder+"_iso", "rotated_"+img_name), img_iso)
    tifffile.imsave(os.path.join(output_folder+"_aniso0", "rotated_"+img_name), img_aniso0)
    tifffile.imsave(os.path.join(output_folder+"_aniso1", "rotated_"+img_name), img_aniso1)