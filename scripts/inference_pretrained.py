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
from functools import partial

from sr_3dunet.utils.inference_base import get_base_argument_parser
from sr_3dunet.utils.data_utils import preprocess, postprocess, extend_block, get_rotated_img, get_anti_rotated_img, str2bool
from scripts.get_MIP import get_and_save_MIP
from scripts.inference_big_tif import handle_bigtif
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def remove_outer_layer(matrix, overlap):
    return matrix
    height, width, depth = matrix.shape
    removed_matrix = matrix[overlap:height-overlap, overlap:width-overlap, overlap:depth-overlap]
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
    parser.add_argument('--piece_flag', type=str2bool, default=False, help='piece_flag')
    parser.add_argument('--piece_size', type=int, default=128, help='piece_size')
    parser.add_argument('--overlap', type=int, default=16, help='overlap')
    parser.add_argument('--step_size', type=int, default=16, help='step_size')
    parser.add_argument('--rotated_flag', type=str2bool, default=False, help='rotated_flag')
    args = parser.parse_args()
    final_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_back = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    if args.piece_flag:
        model = partial(handle_bigtif, model, args.piece_size, args.overlap, args.step_size)
        model_back = partial(handle_bigtif, model_back, args.piece_size, args.overlap, args.step_size)

    # prepare output dir
    os.makedirs(os.path.join(args.output, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "output"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec1"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "out_affine"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "C"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "C_affine"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec_out"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec2"), exist_ok=True)
    percentiles=[0.01, 0.9985] # [0.01,0.9999] 
    dataset_mean=0

    img_path_list = os.listdir(args.input)
    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    num_imgs = len(img_path_list)       # 17
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path)) # [:200,:200,:200]
        img = np.clip(img, 0, 65535)
        origin_shape = img.shape
        img, min_value, max_value = preprocess(img, percentiles, dataset_mean)
        tifffile.imwrite(os.path.join(args.output, "input", "input" + img_path),
                         remove_outer_layer(postprocess(img[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size], min_value, max_value, dataset_mean=dataset_mean), args.overlap))
                
        if args.rotated_flag:
            img = get_rotated_img(img, device)
            
        img = img.astype(np.float32)[None, None,]
        img = torch.from_numpy(img).to(device)     # to float32

        out_img = model(img)
        rec1_img = model_back(out_img)
        
        out_affine_img = out_img.transpose(-1, -2)
        C_img = model_back(out_affine_img)
        C_affine_img = C_img.transpose(-1, -2)
        rec_outimg = model(C_img).transpose(-1, -2)
        rec2_img = model_back(rec_outimg)
        
        
        if args.rotated_flag:
            out_img = get_anti_rotated_img(out_img[0,0].cpu().numpy(), origin_shape)[None, None]
            rec1_img = get_anti_rotated_img(rec1_img[0,0].cpu().numpy(), origin_shape)[None, None]
            out_affine_img = get_anti_rotated_img(out_affine_img[0,0].cpu().numpy(), origin_shape)[None, None]
            C_img = get_anti_rotated_img(C_img[0,0].cpu().numpy(), origin_shape)[None, None]
            C_affine_img = get_anti_rotated_img(C_affine_img[0,0].cpu().numpy(), origin_shape)[None, None]
            rec_outimg = get_anti_rotated_img(rec_outimg[0,0].cpu().numpy(), origin_shape)[None, None]
            rec2_img = get_anti_rotated_img(rec2_img[0,0].cpu().numpy(), origin_shape)[None, None]

        out_img = out_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        rec1_img = rec1_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        out_affine_img = out_affine_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        C_img = C_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        C_affine_img = C_affine_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        rec_outimg = rec_outimg[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        rec2_img = rec2_img[0,0].cpu().numpy()[8:-8, 8:-8, 8:-8][0:final_size, 0:final_size, 0:final_size]
        
        tifffile.imwrite(os.path.join(args.output, "output", "output" + img_path),
                         remove_outer_layer(postprocess(out_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "rec1", "rec1" + img_path),
                         remove_outer_layer(postprocess(rec1_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "out_affine", "out_affine" + img_path),
                         remove_outer_layer(postprocess(out_affine_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "C", "C" + img_path),
                         remove_outer_layer(postprocess(C_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "C_affine", "C_affine" + img_path),
                         remove_outer_layer(postprocess(C_affine_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "rec_out", "rec_out" + img_path),
                         remove_outer_layer(postprocess(rec_outimg, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "rec1", "rec1" + img_path),
                         remove_outer_layer(postprocess(rec1_img, min_value, max_value, dataset_mean), args.overlap))
        tifffile.imwrite(os.path.join(args.output, "rec2", "rec2" + img_path),
                         remove_outer_layer(postprocess(rec2_img, min_value, max_value, dataset_mean), args.overlap))
        
        # get_and_save_MIP(postprocess(out_img.cpu().numpy(), min_value, max_value, dataset_mean),
        #     os.path.join(args.output, "output_proj"), "output"+img_path)
        
        pbar1.update(1)

if __name__ == '__main__':
    main()
