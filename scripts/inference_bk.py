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

from sr_3dunet.utils.data_utils import preprocess, postprocess, get_rotated_img, get_anti_rotated_img, str2bool
from sr_3dunet.utils.inference_big_tif import handle_bigtif_bk as handle_bigtif
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def remove_outer_layer(matrix, remove_size):
    # return matrix
    height, width, depth = matrix.shape
    removed_matrix = matrix[remove_size:height-remove_size, remove_size:width-remove_size, remove_size:depth-remove_size]
    return removed_matrix

def get_inference_model(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)
    model_back = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)

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

    return model, model_back
    # return model.half() if args.half else model, model_back.half() if args.half else model_back

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '--expname', type=str, default='MPCN', help='A unique name to identify your current inference')
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--model_back_path', type=str, help='model_back_path')
    parser.add_argument('--piece_flag', type=str2bool, default=False, help='piece_flag')
    parser.add_argument('--piece_size', type=int, default=128, help='piece_size')
    parser.add_argument('--piece_overlap', type=int, default=16, help='piece_overlap')
    parser.add_argument('--piece_mod_size', type=int, default=16, help='piece_mod_size')
    parser.add_argument('--remove_size', type=int, default=16, help='remove_size')
    parser.add_argument('--rotated_flag', type=str2bool, default=False, help='rotated_flag')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_back = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    print("Model_back size: {:.5f}M".format(sum(p.numel() for p in model_back.parameters())*4/1048576))
    if args.piece_flag:
        model = partial(handle_bigtif, model, args.piece_size, args.piece_overlap)
        model_back = partial(handle_bigtif, model_back, args.piece_size, args.piece_overlap)

    # prepare output dir
    os.makedirs(os.path.join(args.output, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "output"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec1"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "out_affine"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "C"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "C_affine"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec_out"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec2"), exist_ok=True)
    percentiles=[0, 1] # [0, 0.9999] # [0.01,0.999999] # [0.01, 0.9985]
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
                         remove_outer_layer(postprocess(img, min_value, max_value, dataset_mean=dataset_mean), args.remove_size).astype(np.uint16))
                
        if args.rotated_flag:
            img = get_rotated_img(img, device)
            
        img = img.astype(np.float32)[None, None,]
        img = torch.from_numpy(img).to(device)     # to float32

        start_time = time.time()
        torch.cuda.synchronize()
        print('input', img.max())
        print('input', img.shape)
        out_img = model(img)
        print('output', out_img.max())
        torch.cuda.synchronize()
        end_time = time.time()
        # print("avg-time_model:", (end_time-start_time)*1000, "N, C, H, W, D:", origin_shape)
        
        start_time = time.time()
        torch.cuda.synchronize()
        rec1_img = model_back(out_img)
        print('rec', rec1_img.max())
        torch.cuda.synchronize()
        end_time = time.time()
        # print("avg-time_model_back:", (end_time-start_time)*1000, "N, C, H, W, D:", origin_shape)
        
        out_affine_img = out_img.transpose(-1, -3)
        C_img = model_back(out_affine_img)
        C_affine_img = C_img.transpose(-1, -3)
        rec_outimg = model(C_img).transpose(-1, -3)
        rec2_img = model_back(rec_outimg)
        
        
        if args.rotated_flag:
            out_img = get_anti_rotated_img(out_img[0,0].cpu().numpy(), origin_shape) # [None, None]
            rec1_img = get_anti_rotated_img(rec1_img[0,0].cpu().numpy(), origin_shape) # [None, None]
            out_affine_img = get_anti_rotated_img(out_affine_img[0,0].cpu().numpy(), origin_shape) # [None, None]
            C_img = get_anti_rotated_img(C_img[0,0].cpu().numpy(), origin_shape) # [None, None]
            C_affine_img = get_anti_rotated_img(C_affine_img[0,0].cpu().numpy(), origin_shape) # [None, None]
            rec_outimg = get_anti_rotated_img(rec_outimg[0,0].cpu().numpy(), origin_shape) # [None, None]
            rec2_img = get_anti_rotated_img(rec2_img[0,0].cpu().numpy(), origin_shape) # [None, None]

        else:
            out_img = out_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            rec1_img = rec1_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            out_affine_img = out_affine_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            C_img = C_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            C_affine_img = C_affine_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            rec_outimg = rec_outimg[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
            rec2_img = rec2_img[0,0].cpu().numpy() # [0:final_size, 0:final_size, 0:final_size]
        
        tifffile.imwrite(os.path.join(args.output, "output", "output" + img_path),
                         remove_outer_layer(postprocess(out_img, min_value, max_value, dataset_mean), args.remove_size).astype(np.uint16))
        tifffile.imwrite(os.path.join(args.output, "rec1", "rec1" + img_path),
                         remove_outer_layer(postprocess(rec1_img, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "out_affine", "out_affine" + img_path),
                         remove_outer_layer(postprocess(out_affine_img, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "C", "C" + img_path),
                         remove_outer_layer(postprocess(C_img, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "C_affine", "C_affine" + img_path),
                         remove_outer_layer(postprocess(C_affine_img, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "rec_out", "rec_out" + img_path),
                         remove_outer_layer(postprocess(rec_outimg, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "rec1", "rec1" + img_path),
                         remove_outer_layer(postprocess(rec1_img, min_value, max_value, dataset_mean), args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "rec2", "rec2" + img_path),
                         remove_outer_layer(postprocess(rec2_img, min_value, max_value, dataset_mean), args.remove_size))
        
        # get_and_save_MIP(postprocess(out_img.cpu().numpy(), min_value, max_value, dataset_mean),
        #     os.path.join(args.output, "output_proj"), "output"+img_path)
        
        pbar1.update(1)

if __name__ == '__main__':
    main()
