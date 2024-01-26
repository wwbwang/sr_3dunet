import tifffile
import numpy as np
import os
from tqdm import tqdm

def get_slice(img, crop_dimension):
    slices = np.split(img, img.shape[crop_dimension], axis=crop_dimension)
    return slices

def main():
    input_dir = '/home/wangwb/workspace/sr_3dunet/datasets/val'
    out_dir = '/home/wangwb/workspace/sr_3dunet/datasets/val_sub'
    os.makedirs(out_dir, exist_ok=True)
    
    dimension = -3
    
    imgs = os.listdir(input_dir)
    pbar = tqdm(total=len(imgs), unit='tif_img', desc='Cut')
    for img_name in imgs:
        input_img_name = os.path.join(input_dir, img_name)
        img = tifffile.imread(input_img_name)
        slices = np.split(img, img.shape[dimension], axis=dimension)
        slices = [np.squeeze(slice, axis=dimension) for slice in slices]
        for index, slice in enumerate(slices):
            output_img_name = os.path.join(out_dir, str(index)+img_name)        # '_'+
            tifffile.imsave(output_img_name, slice)
            # print(slice.shape)
        pbar.update(1)

if __name__ == '__main__':
    main()