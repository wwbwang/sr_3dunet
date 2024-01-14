import tifffile
import os
import numpy as np
import torch

def preprocess(img, percentiles=[0.01,0.997]):  # 再加一个数量级就获得和原图差不多的视觉效果
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    img = np.sqrt(img)
    return img.mean()


folder_path = "/home/wangwb/workspace/sr_3dunet/datasets/monkey_skel/img_ori" # "/home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/train"
img_list = os.listdir(folder_path)

mean_list = []
for img_path in img_list:
    img = tifffile.imread(os.path.join(folder_path, img_path))
    mean_list.append(preprocess(img))

print(sum(mean_list)/len(mean_list))

# 0.15352097743749618 monkey
# 0.2538556633301327 rotated_blocks