import random
import torch
import cv2
import numpy as np


def random_crop_3d(imgs, patch_size, start_h=None, start_w=None, start_d=None):

    if not isinstance(imgs, list):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w, d = imgs[0].size()[-3:]
    else:
        h, w, d = imgs[0].shape[0:3]

    # randomly choose top and left coordinates
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)
    if start_d is None:
        start_d = random.randint(0, d - patch_size)

    if input_type == 'Tensor':
        imgs = [v[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size, start_d:start_d + patch_size] for v in imgs]
    else:
        imgs = [v[start_h:start_h + patch_size, start_w:start_w + patch_size, start_d:start_d + patch_size, ...] for v in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs

def augment_3d(imgs, aniso_dimension, hflip=True, vflip=True, dflip=True, rotation=True, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    dflip = dflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=-1)
        if vflip:  # vertical
            img = np.flip(img, axis=-2)
        if dflip:  # 
            img = np.flip(img, axis=-3)
        if rot90:
            img = img.transpose(1, 0, 2) if aniso_dimension==-1 else img
            img = img.transpose(2, 1, 0) if aniso_dimension==-2 else img
            img = img.transpose(0, 2, 1) if aniso_dimension==-3 else img
            
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
        
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs
    
def preprocess(img, percentiles=[0.01,0.9999]):
    # input img [0,65535]
    # output img [0,1]
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    # img = np.sqrt(img)
    return img, min_value, max_value

def postprocess(img, min_value, max_value):
    img = img * (max_value-min_value) + min_value
    return img

