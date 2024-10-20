import os
import tifffile as tiff
import numpy as np
from ryu_pytools import check_dir
from tqdm import tqdm
def center_crop(img, size=64):
    center = [i//2 for i in img.shape]
    d,h,w = center
    ds,de = d-size//2, d+size//2
    hs,he = h-size//2, h+size//2
    ws,we = w-size//2, w+size//2
    img_crop = img[ds:de, hs:he, ws:we]
    return img_crop

root_path = '/home/wangwb/data/RESIN_datasets/NISSL/Train_datasets'
save_path = '/share/home/liuy/workspace/data/RESIN/nissl/128/'
check_dir(save_path)

img_name_list = os.listdir(root_path)

# percentage = 0.2
# num = int(len(img_name_list)*percentage)
num = 8000

img_name_sampled_list = np.random.choice(img_name_list, size=num, replace=False)
print(len(img_name_sampled_list))

for i,name in tqdm(enumerate(img_name_sampled_list)):
    img = tiff.imread(os.path.join(root_path,name)).astype(np.uint16)
    img_crop = center_crop(img, size=128)
    tiff.imwrite(os.path.join(save_path, f'{str(i+1).zfill(4)}.tiff'), img_crop)
