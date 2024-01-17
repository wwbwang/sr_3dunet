import argparse
import cv2
import glob
import numpy as np
import os
import psutil
import queue
import threading
import time
import torch
import sys
import tifffile
from os import path as osp
from tqdm import tqdm

from sr_3dunet.utils.inference_base import get_base_argument_parser
from sr_3dunet.utils.video_util import frames2video
from sr_3dunet.utils.data_utils import preprocess, postprocess, rotate_block, extend_block, get_rotated_img
from scripts.get_MIP import get_and_save_MIP
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def remove_outer_layer(matrix):
    return matrix
    step = 2
    height, width, depth = matrix.shape
    removed_matrix = matrix[step:height-step, step:width-step, step:depth-step]
    return removed_matrix

def get_inference_model(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256, 512], dim=3)
    model_back = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256, 512], dim=3)

    model_path = args.model_path
    model_back_path = args.model_back_path
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet['params'], strict=True)
    model.eval()
    model = model.to(device)
    
    loadnet = torch.load(model_back_path)
    model_back.load_state_dict(loadnet['params'], strict=True)
    model_back.eval()
    model_back = model_back.to(device)

    return model.half() if args.half else model, model_back.half() if args.half else model_back

@torch.no_grad()
def main():
    parser = get_base_argument_parser()
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--model_back_path', type=str, help='model_back_path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_back = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))

    # prepare output dir
    os.makedirs(os.path.join(args.output, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "output"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "back"), exist_ok=True)

    img_path_list = os.listdir(args.input)

    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    num_imgs = len(img_path_list)       # 17
    percentiles=[0.01,0.9985] 
    dataset_mean=0
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path))
        img, min_value, max_value = preprocess(img, percentiles, dataset_mean)
        tifffile.imwrite(os.path.join(args.output, "input", "input" + img_path), postprocess(remove_outer_layer(img), min_value, max_value, dataset_mean=dataset_mean))   
        img = img.astype(np.float32)[None, None, ]
        img = torch.from_numpy(img).to(device)     # to float32
        out = model(img)[0][0]
        tifffile.imwrite(os.path.join(args.output, "output", "output" + img_path), postprocess(remove_outer_layer(out).cpu().numpy(), min_value, max_value, dataset_mean=dataset_mean))
        
        back = model_back(out[None, None, ])[0][0]
        tifffile.imwrite(os.path.join(args.output, "back", "back" + img_path), postprocess(remove_outer_layer(back).cpu().numpy(), min_value, max_value, dataset_mean=dataset_mean))
        
        get_and_save_MIP(postprocess(remove_outer_layer(out).cpu().numpy(), min_value, max_value, dataset_mean=dataset_mean), os.path.join(args.output, "output"), "output"+img_path)
        
        pbar1.update(1)

if __name__ == '__main__':
    main()
