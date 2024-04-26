import random
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import argparse
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def random_crop_2d(imgs, patch_size, start_h=None, start_w=None):

    if not isinstance(imgs, list):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    # randomly choose top and left coordinates
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)

    if input_type == 'Tensor':
        imgs = [v[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size] for v in imgs]
    else:
        imgs = [v[start_h:start_h + patch_size, start_w:start_w + patch_size, ...] for v in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs

def augment_3d(imgs, iso_dimension, hflip=True, vflip=True, dflip=True, rotation=True, return_status=False):
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
            img = img.transpose(1, 0, 2) if iso_dimension==-1 else img
            img = img.transpose(2, 1, 0) if iso_dimension==-2 else img
            img = img.transpose(0, 2, 1) if iso_dimension==-3 else img
            
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
    
def augment_3d_rotated(imgs, aniso_dimension, hflip=True, vflip=True, dflip=True, rotation=True, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    dflip = dflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=-1)
            img = np.flip(img, axis=-3)
        if vflip:  # vertical
            img = np.flip(img, axis=-2)
        # if dflip:  # 
        #     img = np.flip(img, axis=-3)
        #     img = img.transpose(2, 1, 0) if aniso_dimension==-2 else img
        if rot90:
            # img = img.transpose(1, 0, 2) if aniso_dimension==-1 else img
            img = img.transpose(2, 1, 0) if aniso_dimension==-2 else img
            # img = img.transpose(0, 2, 1) if aniso_dimension==-3 else img
            
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
    
def augment_2d(imgs, hflip=True, vflip=True, rotation=True, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=-1)
        if vflip:  # vertical
            img = np.flip(img, axis=-2)
        if rot90:
            img = img.transpose(1, 0)
            
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
    
def preprocess(img, percentiles, dataset_mean):  # 再加一个数量级就获得和原图差不多的视觉效果
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    # img = np.sqrt(img)
    img = img - dataset_mean
    return img, min_value, max_value

def postprocess(img, min_value, max_value, dataset_mean=0.153):
    img = img + dataset_mean
    # img = np.square(img)
    img = img * (max_value - min_value) + min_value
    return img

def get_projection(img, iso_dimension):
    list_dimensions = [-1, -2, -3]
    list_dimensions.remove(iso_dimension)
    if isinstance(img, np.ndarray):
        img_iso = np.max(img, axis=iso_dimension)
        img_aniso0 = np.max(img, axis=list_dimensions[0])
        img_aniso1 = np.max(img, axis=list_dimensions[1])
    elif isinstance(img, torch.Tensor):
        img_iso = torch.max(img, dim=iso_dimension).values
        img_aniso0 = torch.max(img, dim=list_dimensions[0]).values
        img_aniso1 = torch.max(img, dim=list_dimensions[1]).values
    return img_iso, img_aniso0, img_aniso1

def affine_img(img, iso_dimension=-1, aniso_dimension=None):
    if aniso_dimension is None:
        list_dimensions = [-1, -2, -3]
        list_dimensions.remove(iso_dimension)
        aniso_dimension = random.choice(list_dimensions)

    img = img.transpose(iso_dimension, aniso_dimension)
    return img, aniso_dimension

def affine_img_VISoR(img, aniso_dimension=-2, half_iso_dimension=None):

    if half_iso_dimension is None:
        list_dimensions = [-1, -2, -3]
        list_dimensions.remove(aniso_dimension)
        half_iso_dimension = random.choice(list_dimensions)
    
    img = img.transpose(half_iso_dimension, aniso_dimension)
    half_iso_dimension, aniso_dimension = aniso_dimension, half_iso_dimension
    return img, half_iso_dimension, aniso_dimension

def extend_block_utils(img, step_size=16, dim=3):
    def extend_block_(img):
        if dim==3:
            h, w, d = img.shape
            pad_height = (step_size - h % step_size) if h % step_size != 0 else 0
            pad_width = (step_size - w % step_size) if w % step_size != 0 else 0
            pad_depth = (step_size - d % step_size) if d % step_size != 0 else 0
            
            padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, pad_depth)), mode='constant') if isinstance(img, np.ndarray)\
                else F.pad(img, (0, pad_depth, 0, pad_width, 0, pad_height), mode='constant') 
        elif dim==2:
            h, w = img.shape
            pad_height = (step_size - h % step_size) if h % step_size != 0 else 0
            pad_width = (step_size - w % step_size) if w % step_size != 0 else 0
            
            padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant') if isinstance(img, np.ndarray)\
                else F.pad(img, (0, pad_width, 0, pad_height), mode='constant')
        return padded_img
                
    if img.ndim > 3:
        return extend_block_(img[0,0])[None, None]
    else:
        return extend_block_(img)

def get_rotated_img(raw_img, device, aniso_dimension=-2):
    raw_img = raw_img.astype(np.float32)
    list_img = []
    for i in range(raw_img.shape[aniso_dimension]):
        img = raw_img[..., i, :]
        height, width = img.shape
        max_size = max(height, width)

        desired_height = int(max_size * 1.414213) // 2 * 2
        desired_width = int(max_size * 1.414213) // 2 * 2

        border_height = (desired_height - height) // 2
        border_width = (desired_width - width) // 2

        center_x = desired_height // 2
        center_y = desired_height // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 45, 1)

        extend_img = np.pad(img, ((border_height, border_height), (border_width, border_width)), mode='constant')
        rotated_img = cv2.warpAffine(extend_img, rotation_matrix, (desired_width, desired_height))
        list_img.append(rotated_img)
        
def get_anti_rotated_img(raw_img, device, aniso_dimension=-2):
    raw_img = raw_img.astype(np.float32)
    list_img = []
    for i in range(raw_img.shape[aniso_dimension]):
        img = raw_img[..., i, :]
        height, width = img.shape
        max_size = max(height, width)

        desired_height = int(max_size * 1.414213) // 2 * 2
        desired_width = int(max_size * 1.414213) // 2 * 2

        border_height = (desired_height - height) // 2
        border_width = (desired_width - width) // 2

        center_x = desired_height // 2
        center_y = desired_height // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 45, 1)

        extend_img = np.pad(img, ((border_height, border_height), (border_width, border_width)), mode='constant')
        rotated_img = cv2.warpAffine(extend_img, rotation_matrix, (desired_width, desired_height))
        list_img.append(rotated_img)
        
    return extend_block_utils(np.stack(list_img, axis=1))

# def get_anti_rotated_img(raw_img, origin_shape=(128, 128, 128), aniso_dimension=-2):
#     raw_img = raw_img.astype(np.float32)
#     list_img = []
#     origin_height, _, origin_width = origin_shape
#     max_size = max(origin_height, origin_width)
#     desired_height = int(max_size * 1.414213) // 2 * 2
#     desired_width = int(max_size * 1.414213) // 2 * 2
    
#     border_height = (desired_height - origin_height) // 2
#     border_width = (desired_width - origin_width) // 2

#     center_x = desired_height // 2
#     center_y = desired_height // 2
    
#     for i in range(raw_img.shape[aniso_dimension]):
#         img = raw_img[..., i, :]
#         img = img[:desired_height, :desired_width]
#         rotation_matrix_inv = cv2.getRotationMatrix2D((center_x, center_y), 45, 1)

#         rotated_img = cv2.warpAffine(img, rotation_matrix_inv, (desired_width, desired_height))
#         list_img.append(rotated_img[border_height:-border_height, border_width:-border_width])
        
#     out_img = np.stack(list_img, axis=1)
#     return out_img

def get_rotated_projection(img, aniso_dimension=-2):   # 默认左上到右下倾斜
    if aniso_dimension == -2:
        list_iso = []
        list_aniso = []
        for i in range(img.shape[aniso_dimension]):
            rotated_raw_img = get_rotated_img(img[..., i, :])
            iso_array, aniso_array = np.max(rotated_raw_img, axis=1), np.max(rotated_raw_img, axis=0)
            list_iso.append(iso_array)
            list_aniso.append(aniso_array)
        iso_proj = np.stack(list_iso)
        aniso_proj = np.stack(list_aniso)
        return iso_proj, aniso_proj
    else:
        return "error"

def crop_block(img, step_size=16, dim=3):
    if dim==3:
        h, w, d = img.shape
        min_shape = min(h, w, d)
        crop_shape = min_shape // step_size * step_size
        
        img = img[0:crop_shape, 0:crop_shape, 0:crop_shape]
        
    elif dim==2:
        h, w = img.shape
        min_shape = min(h, w)
        crop_shape = min_shape // step_size * step_size
        
        img = img[0:crop_shape, 0:crop_shape]
    
    return img

# !!! Just used in standard VISoR data
# Aniso_dimension is -2 in source VISoR data
# After rotation the isotropic dimension is -1, and the blur direction is horizontal in -2 and -3 (viewed in imagej)
# def rotate_block(raw_img, aniso_dimension=-2):
#     iso_dimension = -1
#     height, width, depth = raw_img.shape
#     max_size = max(height, depth)
    
#     desired_height = int(max_size * 1.414213) // 2 * 2
#     desired_depth = int(max_size * 1.414213) // 2 * 2
    
#     border_height = (desired_height - height) // 2
#     border_depth = (desired_depth - depth) // 2
    
#     raw_img = raw_img.transpose(1, 0, 2)
    
#     rotation_matrix = cv2.getRotationMatrix2D((desired_height // 2, desired_depth // 2), 45, 1)
#     rotation_matrix_inv = cv2.getRotationMatrix2D((desired_height // 2, desired_depth // 2), -45, 1)
    
#     extend_raw_img = np.pad(raw_img, ((0, 0), (border_height, border_height), (border_depth, border_depth)), mode='constant')
#     rotated_raw_img = np.zeros_like(extend_raw_img)
#     for i in range(width):
#         rotated_raw_img[i] = cv2.warpAffine(extend_raw_img[i], rotation_matrix, (desired_depth, desired_height))
    
#     # out_img = np.zeros_like(rotated_raw_img)
#     # for i in range(width):
#     #     out_img[i] = cv2.warpAffine(xz_img[i], rotation_matrix_inv, (desired_depth, desired_height))
        
#     # out_img = out_img[:, border_height:-border_height, border_depth:-border_depth]
#     # out_img = out_img.transpose(1, 0, 2)
    
#     return rotated_raw_img # , iso_dimension

def merge_img(img_list):
    matrix_array = np.array(img_list)
    return np.max(matrix_array, axis=0)