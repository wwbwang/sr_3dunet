import numpy as np
import os
import sys
import tifffile as tiff
from empatches import EMPatches
import torch
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

def handle_big_tif(model, img, args, device):
    crop_size = args.crop_size
    overlap = args.overlap
    emp = EMPatches()
    patches, indices = emp.extract_patches(img, patchsize=crop_size, overlap=overlap, stride=None, vox=True)
    patches_res = []
    for real_A in patches:
        real_A, min_value, max_value = norm_min_max(real_A, return_info=True)
        real_A = torch.from_numpy(real_A)[None,None].to(device)
        fake_B = model.G_A(real_A)
        fake_B = torch.clip(fake_B, 0, 1)
        fake_B = fake_B*(max_value-min_value) + min_value
        patches_res.append(tensor_to_ndarr(fake_B[0,0]))
    res = emp.merge_patches(patches_res, indices, mode='avg')
    return res.astype(np.uint16)

def main():
    parser = argparse.ArgumentParser(description='eval args')
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-weight', type=str)
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-img_path', type=str)
    parser.add_argument('-save_base_path', type=str, default='inference/result')
    parser.add_argument('-big_tif', action='store_true')
    parser.add_argument('-crop_size', type=int, default=64)
    parser.add_argument('-overlap', type=float, default=0.25)

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
            if args.big_tif:
                fake_B = handle_big_tif(model, real_A, args, device)
                tiff.imwrite(os.path.join(save_path, f'fake_{name}'), fake_B)
            else:
                real_A = norm_fn(args.data_norm_type)(real_A)
                real_A = torch.from_numpy(real_A)[None,None].to(device)
                fake_B = model.G_A(real_A) 
                tiff.imwrite(os.path.join(save_path, f'fake_{name}'), tensor_to_ndarr(fake_B[0,0]).astype(np.float16))

if __name__ == '__main__':
    main()