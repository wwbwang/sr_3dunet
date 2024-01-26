import tifffile
import numpy as np
import torch
import os
import random
from tqdm import tqdm

def main():
    input_dir = '/home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/sub_train'
    output_dir = '/home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/sub_train_2drotate'
    
    os.makedirs(output_dir, exist_ok=True)
    
    imgs = os.listdir(input_dir)
    pbar = tqdm(total=len(imgs), unit='tif_img', desc='Cut')
    for img_name in imgs:
        input_img_name = os.path.join(input_dir, img_name)
        output_img_name = os.path.join(output_dir, img_name)
        img = tifffile.imread(input_img_name)
        rotate_img = img.transpose(1,0)
        tifffile.imsave(output_img_name, rotate_img)
        pbar.update(1)

if __name__ == '__main__':
    main()