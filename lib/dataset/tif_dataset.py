import numpy as np
import os
import random
import torch
from torch.utils import data
import tifffile

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

class tif_dataset(data.Dataset):
    def __init__(self, data_path, data_norm_type='min_max', augment=True, aniso_dim=-2):
        super(tif_dataset, self).__init__()
        img_name_list = os.listdir(data_path)
        self.img_path_list = []
        for img_name in img_name_list:
            self.img_path_list.append(os.path.join(data_path, img_name))
        
        self.augment = augment
        self.aniso_dim = aniso_dim
        self.data_norm_type = data_norm_type

    def getitem_np(self, index):
        img_path = self.img_path_list[index]
        img = tifffile.imread(img_path).astype(np.float32)
        img = normalize(img, type=self.data_norm_type)
        return img
        
    def __getitem__(self, index):
        img_np = self.getitem_np(index)
        img_tensor = torch.from_numpy(img_np)
        if self.augment:
            img_tensor = augment_3d(img_tensor, self.aniso_dim)
        img_tensor = img_tensor[None]
        return img_tensor
    
    def __len__(self):
        return len(self.img_path_list)

def norm_min_max(img:np.ndarray, percentiles=[0,1]):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    return img

def norm_abs(img:np.ndarray, abs_value=65535):
    assert abs_value >= 0, 'invalid abs_value'
    img = img/abs_value
    return img

def normalize(img:np.ndarray, type):
    if type=='min_max':
        return norm_min_max(img)
    elif type=='abs':
        return norm_abs(img)

def augment_3d(img:torch.Tensor, dim_keep=-2, flip=True, transpose=True) -> torch.Tensor:
    dim_alter = [-3,-2,-1]
    dim_alter.remove(dim_keep)
    if flip and random.random() < 0.5:
        img = torch.flip(img, dims=[dim_alter[0]])
        img = torch.flip(img, dims=[dim_alter[1]])
    if flip and random.random() < 0.5:
        img = torch.flip(img, dims=[dim_keep])
    if transpose and random.random() < 0.5:
        img = img.transpose(dim_alter[0], dim_alter[1])
    return img

def get_dataset(args):
    train_dataset = tif_dataset(data_path=args.data,
                                data_norm_type=args.data_norm_type,
                                augment=args.augment,
                                aniso_dim=args.aniso_dim)
    return train_dataset

if __name__ == '__main__':
    from ryu_pytools import arr_info, plot_mip, plot_some
    import napari
    from lib.utils.utils import get_slant_mip

    data_path = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/fg_blocks'
    ds = tif_dataset(data_path)
    print(len(ds))
    cube = ds.__getitem__(np.random.randint(len(ds)))
    mip = get_slant_mip(cube[None], iso_dim=-1)

    cube_np = cube[0].cpu().numpy()
    mip_np = mip[0,0].cpu().numpy()
    arr_info(cube_np)
    arr_info(mip_np)

    plot_mip(cube_np, figsize=(15,5))
    plot_some([mip_np])
    pass
    # viewer = napari.Viewer(ndisplay=3)
    # viewer.add_image(cube_np)
    # viewer.add_image(mip_np)
    # napari.run()