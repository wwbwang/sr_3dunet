import numpy as np
import os
import sys
import tifffile as tiff
import torch
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    sys.path.append(os.getcwd())
from lib.dataset.tif_dataset import norm_min_max
from lib.utils.utils import check_dir

import ruamel.yaml as yaml

def read_cfg(args):
    with open(args.cfg, 'r') as f:
        yml = yaml.safe_load(f)
    cmd = [c[1:] for c in sys.argv if c[0]=='-']
    for k,v in yml.items():
        if k not in cmd:
            args.__dict__[k] = v
    return args
    
def handle_bigTif(model, img, args, device):
    overlap = args.overlap
    piece_size = args.piece_size
    D, H, W = img.shape
    img_out = np.zeros_like(img)
    
    def cal_index(idx, BORDER):
        if idx + piece_size >= BORDER:
            start = BORDER - piece_size
            end = BORDER
            cut_left = idx + overlap//2 - start
            cut_right = 0
        else:
            start = idx
            end = start + piece_size
            cut_left = 0 if idx==0 else overlap//2
            cut_right = overlap//2
        return start, end, cut_left, cut_right

    for d_idx in range(0, D, piece_size-overlap):
        d_s, d_e, d_cL, d_cR = cal_index(d_idx, D)

        for h_idx in range(0, H, piece_size-overlap):
            h_s, h_e, h_cL, h_cR = cal_index(h_idx, H)
            
            for w_idx in range(0, W, piece_size-overlap):
                w_s, w_e, w_cL, w_cR = cal_index(w_idx, W)        

                piece_in = img[d_s:d_e, h_s:h_e, w_s:w_e]
                # preprocess
                piece_in, min_value, max_value = norm_min_max(piece_in, return_info=True)
                piece_in = torch.from_numpy(piece_in)[None,None].to(device)

                piece_out = model(piece_in)
                piece_out = torch.clip(piece_out, 0, 1)
                piece_out = piece_out[0,0].cpu().numpy()
                # postprocess
                piece_out = piece_out*(max_value-min_value) + min_value

                piece_out = piece_out[d_cL:piece_size-d_cR, h_cL:piece_size-h_cR, w_cL:piece_size-w_cR]                
                img_out[d_s+d_cL:d_e-d_cR, h_s+h_cL:h_e-h_cR, w_s+w_cL:w_e-w_cR] = piece_out
                
                if w_e==W: break
            if h_e==H: break
        if d_e==D: break
    
    return img_out

def main():
    parser = argparse.ArgumentParser(description='eval args')
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-ckpt_path', type=str)
    parser.add_argument('-img_path', type=str)
    parser.add_argument('-save_dir', type=str, default='inference/result')
    parser.add_argument('-save_name', type=str, default='RESIN_output.tif')
    parser.add_argument('-piece_size', type=int, default=64)
    parser.add_argument('-overlap', type=int, default=16)

    args = parser.parse_args()
    args = read_cfg(args)

    ckpt_path = args.ckpt_path
    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device)

    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    model = get_model(args).to(device)
    if 'model' in ckpt.keys():
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
    else:
        model = model.G_A
        model.load_state_dict(ckpt)
    model.eval()

    img_path = args.img_path
    save_dir = args.save_dir
    check_dir(save_dir)
    save_name = args.save_name
    save_path = os.path.join(save_dir, save_name)

    with torch.no_grad():
        real_A = tiff.imread(os.path.join(img_path)).astype(np.float32)
        if real_A.shape[-1]>64:
            fake_B = handle_bigTif(model, real_A, args, device)
            fake_B = fake_B.astype(np.uint16)
            tiff.imwrite(save_path, fake_B)
        else:
            real_A, min_value, max_value = norm_min_max(real_A, return_info=True)
            real_A = torch.from_numpy(real_A)[None,None].to(device)
            fake_B = model(real_A) 
            fake_B = torch.clip(fake_B, 0, 1)
            fake_B = fake_B[0,0].cpu().numpy()
            fake_B = fake_B*(max_value-min_value) + min_value
            fake_B = fake_B.astype(np.uint16)
            tiff.imwrite(save_path, fake_B)

if __name__ == '__main__':
    main()