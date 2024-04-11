import tifffile
import numpy as np
import torch
import h5py
import os
import random
from tqdm import tqdm
import cv2

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

    # img_mean = img.mean()
    # img_max = img.max()
    # if img_mean> threshold and img_max > minmax:
    if max_value > minmax:
        return True
    else:
        return False
    
# def get_MIPlist_from_cube(cube_img, MIP_depth, time):
#     MIP_list = []
    
    
#     for index in range(cube_img.shape[1]//MIP_depth-1):
#         MIP_img, _, _ = get_projection(cube_img[:,:,index*MIP_depth:(index+1)*MIP_depth], iso_dimension)
#         MIP_list.append(MIP_img)
        
#     return MIP_list

# MIP也要随机取样，存厚度64和128的分别一次。看128会不会结果会变好？
MIP_depth = 64 # 128 # 64

size = 256
stride = size//2 # 128

# 1024 for test, 3072 for use 4096 for stable
length = 4096 # 3072 # 384*8 

x_center = 5216
y_center = 4192
z_center = 8848

x_floor = x_center-length//2
x_upper = x_center+length//2

y_floor = y_center-length//2
y_upper = y_center+length//2

z_floor = z_center-length//2
z_upper = z_center+length//2

# threshold = 650
percentiles0 = [0, 0.9999]
percentiles1 = [0, 0.99999]
minmax = 500

input_ims_name = 'z00002_c1'
input_ims_path = "/share/data/VISoR_Reconstruction/SIAT_SIAT/WeiPengfei/Mouse_Brain/20211125_WFY_ZJJ_AI14_149_1/ROIReconstruction/ROIImage/1.0/" + input_ims_name + ".ims"
output_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/front_"+str(size)+"_datasets"
output_rotated_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/rotated_front_"+str(size)+"_datasets"
output_back_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/back_"+str(size)+"_datasets"
# output_MIP_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/MIP"+str(MIP_depth)+"_datasets"
# output_MIPScreen_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/MIP"+str(MIP_depth)+"Screen_datasets"
# output_MIP_jpg_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/MIP"+str(MIP_depth)+"_jpg_datasets"

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
# if not os.path.exists(output_MIP_folder):
#     os.makedirs(output_MIP_folder, exist_ok=True)
# if not os.path.exists(output_MIPScreen_folder):
#     os.makedirs(output_MIPScreen_folder, exist_ok=True)
# if not os.path.exists(output_MIP_jpg_folder):
#     os.makedirs(output_MIP_jpg_folder, exist_ok=True)

h5 = h5py.File(input_ims_path, 'r')
img_total = h5['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data']

index_total = 0
index_partial = 0

for start_x in tqdm(range(x_floor, x_upper-size+1, stride), desc="1st loop"):
    for start_y in tqdm(range(y_floor, y_upper-size+1, stride), desc="2st loop"):
        for start_z in tqdm(range(z_floor, z_upper-size+1, stride), desc="3st loop", leave=False):
            index_total += 1
            now_img = img_loader(img_total, start_z, start_y, start_x, size)
            now_img = now_img.transpose(2,1,0)

            if judge_img(now_img, percentiles0, minmax):
                index_partial += 1
                tifffile.imwrite(os.path.join(output_front_folder, input_ims_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), now_img)
                
                # 256 256 256 --> 358 256 358
                rotated_cube_img = get_rotated_img(now_img, aniso_dimension)
                # 358 256 358 --> 174 256 174
                rotated_cube_img = rotated_cube_img[rotated_cube_img.shape[0]//4*1+5:rotated_cube_img.shape[0]//4*3-5,
                                                    :,
                                                    rotated_cube_img.shape[2]//4*1+5:rotated_cube_img.shape[2]//4*3-5]
                if judge_img(rotated_cube_img, percentiles0, minmax):
                    # save 174 256 174 cube
                    tifffile.imwrite(os.path.join(output_rotated_front_folder, input_ims_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), rotated_cube_img)
                # # 358 256 358 --> MIP lists(depth is given)
                # MIP_list = get_MIPlist_from_cube(rotated_cube_img, MIP_depth)
                
                # for index, MIP_img in enumerate(MIP_list):
                #     MIP_img_name = input_ims_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'_index'+str(index).zfill(2)
                #     tifffile.imwrite(os.path.join(output_MIP_folder, MIP_img_name+'.tif'), MIP_img)
                #     if judge_img(MIP_img, percentiles1, minmax):
                #         tifffile.imwrite(os.path.join(output_MIPScreen_folder, MIP_img_name+'.tif'), MIP_img)
        
                #     save_jpg_from_np(MIP_img, output_MIP_jpg_folder, MIP_img_name+'.jpg')
            else:
                 tifffile.imwrite(os.path.join(output_back_folder, input_ims_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), now_img)
            # print(index_total)
            
print('front_ground/back_ground: '+str(index_partial)+'/'+str(index_total))


