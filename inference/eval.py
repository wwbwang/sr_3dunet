import numpy as np
import os
import sys
import tifffile as tiff
from empatches import EMPatches
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from ryu_pytools import tensor_to_ndarr, check_dir

if __name__ == '__main__':
    sys.path.append(os.getcwd())
from lib.dataset.tif_dataset import norm_min_max, norm_fn

import ruamel.yaml as yaml

def read_cfg(args):
    with open(args.cfg, 'r') as f:
        yml = yaml.safe_load(f)
    cmd = [c[1:] for c in sys.argv if c[0]=='-']
    for k,v in yml.items():
        if k not in cmd:
            args.__dict__[k] = v
    return args

# def handle_bigtif(model, img, args, device):
#     crop_size = args.crop_size
#     overlap = args.overlap / crop_size
#     emp = EMPatches()
#     patches, indices = emp.extract_patches(img, patchsize=crop_size, overlap=overlap, stride=None, vox=True)
#     patches_res = []
#     for real_A in patches:
#         real_A, min_value, max_value = norm_min_max(real_A, return_info=True)
#         real_A = torch.from_numpy(real_A)[None,None].to(device)
#         fake_B = model.G_A(real_A)
#         fake_B = torch.clip(fake_B, 0, 1)
#         fake_B = fake_B*(max_value-min_value) + min_value
#         patches_res.append(tensor_to_ndarr(fake_B[0,0]))
#     res = emp.merge_patches(patches_res, indices, mode='max')
#     return res.astype(np.uint16)
    
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
                piece_in, min_value, max_value = norm_min_max(piece_in, return_info=True)
                piece_in = torch.from_numpy(piece_in)[None,None].to(device)

                piece_out = model.G_A(piece_in)
                piece_out = piece_out[0,0].cpu().numpy()
                piece_out = piece_out*(max_value-min_value) + min_value
                
                img_out[d_s+d_cL:d_e-d_cR, h_s+h_cL:h_e-h_cR, w_s+w_cL:w_e-w_cR] = piece_out[d_cL:piece_size-d_cR, h_cL:piece_size-h_cR, w_cL:piece_size-w_cR]
                
                if w_e==W: break
            if h_e==H: break
        if d_e==D: break
    
    return img_out

def main():
    parser = argparse.ArgumentParser(description='eval args')
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-weight', type=str)
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-img_path', type=str)
    parser.add_argument('-save_base_path', type=str, default='inference/result')
    parser.add_argument('-big_tif', action='store_true')
    parser.add_argument('-piece_size', type=int, default=64)
    parser.add_argument('-overlap', type=int, default=16)

    args = parser.parse_args()
    args = read_cfg(args)

    weight = args.weight
    epoch = args.epoch
    ckpt_path = f'out/weights/{weight}/Epoch_{str(epoch).zfill(4)}.pth'
    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device)
    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    model = get_model(args).to(device)
    model.eval()
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

    img_path = args.img_path
    save_base_path = args.save_base_path
    save_path = os.path.join(save_base_path, f'{weight}_{str(epoch).zfill(4)}')
    check_dir(save_path)

    with torch.no_grad():
        img_name_list = os.listdir(img_path)
        for name in tqdm(img_name_list):
            real_A = tiff.imread(os.path.join(img_path, name)).astype(np.float32)
            if args.big_tif and real_A.shape[-1]>64:
                fake_B = handle_bigTif(model, real_A, args, device)
                tiff.imwrite(os.path.join(save_path, f'fake_{name}'), fake_B)
            else:
                real_A = norm_fn(args.data_norm_type)(real_A)
                real_A = torch.from_numpy(real_A)[None,None].to(device)
                fake_B = model.G_A(real_A) 
                tiff.imwrite(os.path.join(save_path, f'fake_{name}'), tensor_to_ndarr(fake_B[0,0]).astype(np.float16))

if __name__ == '__main__':
    main()