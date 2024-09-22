import argparse
import numpy as np
import os
import time
import torch
import tifffile
import math
from os import path as osp
from tqdm import tqdm
from functools import partial

from sr_3dunet.utils.data_utils import preprocess, postprocess
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def get_inference_model(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)

    model_path = args.model_path
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet['params'], strict=True)
    model.eval()
    model = model.to(device)

    return model.half() if args.half else model

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '--expname', type=str, default='MPCN', help='A unique name to identify your current inference')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--piece_size', type=int, default=64, help='piece_size')
    parser.add_argument('--piece_overlap', type=int, default=16, help='piece_overlap')
    parser.add_argument('--h5_dir', default='DataSet/ResolutionLevel 0/TimePoint 0/Channel 3/Data', help='Directory of the h5 file, separated by : ')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    percentiles=[0, 1]
    dataset_mean=0
    
    os.makedirs(args.output, exist_ok=True)

    img_path_list = os.listdir(args.input)
    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path)) # [:200,:200,:200]
        img = np.clip(img, 0, 65535)
        
        img_total = img
        h, w, d = img_total.shape
        img_out = np.zeros(img_total.shape)
        
        len1 = math.ceil(h/(args.piece_size-args.piece_overlap))
        len2 = math.ceil(w/(args.piece_size-args.piece_overlap))
        len3 = math.ceil(d/(args.piece_size-args.piece_overlap))
        pbar1 = tqdm(total=len1*len2*len3, unit='h5_img', desc='inference')
        
        piece_size = args.piece_size
        overlap = args.piece_overlap
    
        for start_h in range(0, h, piece_size-overlap):
            end_h = start_h + piece_size
            
            for start_w in range(0, w, piece_size-overlap):
                end_w = start_w + piece_size
                
                for start_d in range(0, d, piece_size-overlap):
                    end_d = start_d + piece_size

                    img = img_total[start_h:end_h, start_w:end_w, start_d:end_d]
                    img = np.clip(img, 0, 65535)
                    img, min_value, max_value = preprocess(img, percentiles, dataset_mean)
                    
                    end_h = h if end_h>h else end_h
                    end_w = w if end_w>w else end_w
                    end_d = d if end_d>d else end_d
                    origin_shape = img.shape
                    
                    if end_h == h or end_w==w or end_d==d:
                        img =  np.pad(img, ((0, piece_size-end_h+start_h), 
                            (0, piece_size-end_w+start_w), 
                            (0, piece_size-end_d+start_d)), mode='constant')
                        
                    h_cutleft = 0 if start_h==0 else overlap//2
                    w_cutleft = 0 if start_w==0 else overlap//2
                    d_cutleft = 0 if start_d==0 else overlap//2
                    
                    h_cutright = 0 if end_h==h else overlap//2
                    w_cutright = 0 if end_w==w else overlap//2
                    d_cutright = 0 if end_d==d else overlap//2
                    
                    img = img.astype(np.float32)[None, None,]
                    img = torch.from_numpy(img).to(device)     # to float32

                    out_img = model(img).cpu().numpy()
                    out_img = postprocess(out_img, min_value, max_value, dataset_mean)
                    out_img = out_img[:,:,0+h_cutleft:end_h-start_h-h_cutright, 0+w_cutleft:end_w-start_w-w_cutright, 0+d_cutleft:end_d-start_d-d_cutright]

                    img_out[start_h+h_cutleft:end_h-h_cutright, start_w+w_cutleft:end_w-w_cutright, start_d+d_cutleft:end_d-d_cutright] = out_img[0,0]
                    
                    pbar1.update(1)
                    
        tifffile.imwrite(os.path.join(args.output, "output" + img_path), img_out.astype(np.uint16))
if __name__ == '__main__':
    main()
