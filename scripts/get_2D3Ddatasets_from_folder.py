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

def img_loader(img, start_x, start_y, start_z, size):
    return img[start_x:start_x+size, start_y:start_y+size, start_z:start_z+size]

def save_jpg_from_np(img, img_folder, img_name):
    img = np.sqrt(img)
    cv2.imwrite(os.path.join(img_folder, img_name), img)
    
def judge_img(img, percentiles, minmax):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

    max_value = np.max(clipped_arr) 
    if max_value > minmax:
        return True
    else:
        return False

size = 128
stride = size//2 # 128

# threshold = 650
percentiles0 = [0, 0.9999]
percentiles1 = [0, 0.99999]
minmax = 400

input_folder = '/share/home/wangwb/workspace/sr_3dunet/datasets/neuron_LGN-V1-ROI/rotated_datasets'
output_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/neuron_LGN-V1-ROI/front_"+str(size)+"_datasets"
output_back_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/neuron_LGN-V1-ROI/back_"+str(size)+"_datasets"
output_rotated_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/neuron_LGN-V1-ROI/rotated_front_"+str(size)+"_datasets"

aniso_dimension = -2    # do not change
iso_dimension = -1      # do not change
if aniso_dimension!=-2 or iso_dimension!=-1:
    raise RuntimeError('iso_dimension or aniso_dimension is ERROR')

if not os.path.exists(output_front_folder):
    os.makedirs(output_front_folder, exist_ok=True)
if not os.path.exists(output_rotated_front_folder):
    os.makedirs(output_rotated_front_folder, exist_ok=True)
if not os.path.exists(output_back_folder):
    os.makedirs(output_back_folder, exist_ok=True)

filename_list = os.listdir(input_folder)
pbar1 = tqdm(total=len(filename_list), unit='img', desc='create dataset')

index_partial = 0
index_total = 0

for filename in filename_list:
    filepath = os.path.join(input_folder, filename)
    index_total += 1
    now_img = tifffile.imread(filepath)
    now_img = now_img.transpose(2,1,0)

    if judge_img(now_img, percentiles0, minmax):
        index_partial += 1
        tifffile.imwrite(os.path.join(output_front_folder, filename), now_img)
        
        # 256 256 256 --> 358 256 358
        rotated_cube_img = get_rotated_img(now_img, aniso_dimension)
        # 358 256 358 --> 174 256 174
        rotated_cube_img = rotated_cube_img[rotated_cube_img.shape[0]//4*1+5:rotated_cube_img.shape[0]//4*3-5,
                                            :,
                                            rotated_cube_img.shape[2]//4*1+5:rotated_cube_img.shape[2]//4*3-5]
        if judge_img(rotated_cube_img, percentiles0, minmax):
            # save 174 256 174 cube
            tifffile.imwrite(os.path.join(output_rotated_front_folder, filename), rotated_cube_img)
    else:
            tifffile.imwrite(os.path.join(output_back_folder, filename), now_img)
    
    pbar1.update(1)

print('front_ground/back_ground: '+str(index_partial)+'/'+str(index_total))


