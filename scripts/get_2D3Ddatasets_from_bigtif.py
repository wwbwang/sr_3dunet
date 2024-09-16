import tifffile
import numpy as np
import os
from tqdm import tqdm
import cv2
import math

from sr_3dunet.utils.data_utils import get_projection, get_rotated_img, crop_block

def img_loader(img, start_x, start_y, start_z, size):
    return img[start_x:start_x+size, start_y:start_y+size, start_z:start_z+size]

def save_jpg_from_np(img, img_folder, img_name):
    img = np.sqrt(img)
    cv2.imwrite(os.path.join(img_folder, img_name), img)
    
def judge_img(img, percentiles, minmax, maxmin):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

    max_value = np.max(clipped_arr) 
    min_value = np.min(clipped_arr) 
    if max_value > minmax and min_value < maxmin:
        return True
    else:
        return False

size = 128
stride = size//2 # 128


# threshold = 650
percentiles0 = [0.75, 0.99]
# percentiles1 = [0, 0.99999]
minmax = 450
maxmin = 60000

# 40x20x030p26sp1 40x20x03px0p26ax 
input_tif_name = '40x20x030p26sp1'
input_tif_path = "/share/home/wangwb/workspace/sr_3dunet/datasets/40X/" + input_tif_name + ".tif"
output_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/40X/"+ input_tif_name + '_' + str(size)+"_newdatasets"
output_rotated_front_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/40X/"+ input_tif_name + '_rotated' +str(size)+"_newdatasets"
output_back_folder = "/share/home/wangwb/workspace/sr_3dunet/datasets/40X/"+ input_tif_name + '_back' +str(size)+"_newdatasets"

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

img_total = tifffile.imread(input_tif_path).transpose(0,2,1)
index_total = 0
index_partial = 0

# (1521, 576, 1413)

clip_size = 50
x_floor = clip_size
x_upper = img_total.shape[0]-clip_size

y_floor = img_total.shape[1] //3
y_upper = img_total.shape[1] //3 * 2 # img_total.shape[1]-clip_size

z_floor = clip_size
z_upper = img_total.shape[2]-clip_size

len1 = math.ceil((x_upper-size+1-x_floor)/stride)
len2 = math.ceil((y_upper-size+1-y_floor)/stride)
len3 = math.ceil((z_upper-size+1-z_floor)/stride)
pbar1 = tqdm(total=len1*len2*len3, unit='img', desc='create dataset')


for start_x in range(x_floor, x_upper-size+1, stride):
    for start_y in range(y_floor, y_upper-size+1, stride):
        for start_z in range(z_floor, z_upper-size+1, stride):
            index_total += 1
            now_img = img_loader(img_total, start_x, start_y, start_z, size)
            # now_img = now_img.transpose(0,2,1)

            if judge_img(now_img, percentiles0, minmax, maxmin):
                index_partial += 1
                tifffile.imwrite(os.path.join(output_front_folder, input_tif_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), now_img)
                
                # 256 256 256 --> 358 256 358
                rotated_cube_img = get_rotated_img(now_img, aniso_dimension)
                # 358 256 358 --> 174 256 174
                rotated_cube_img = rotated_cube_img[rotated_cube_img.shape[0]//4*1+5:rotated_cube_img.shape[0]//4*3-5,
                                                    :,
                                                    rotated_cube_img.shape[2]//4*1+5:rotated_cube_img.shape[2]//4*3-5]
                if judge_img(rotated_cube_img, percentiles0, minmax, maxmin):
                    # save 174 256 174 cube
                    tifffile.imwrite(os.path.join(output_rotated_front_folder, input_tif_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), rotated_cube_img)
            else:
                 tifffile.imwrite(os.path.join(output_back_folder, input_tif_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'), now_img)
            
            pbar1.update(1)

print('front_ground/back_ground: '+str(index_partial)+'/'+str(index_total))



