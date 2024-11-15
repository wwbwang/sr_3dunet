import torch
import numpy as np
import tifffile as tiff
import h5py
from tqdm import tqdm
import argparse
from ruamel.yaml import YAML
from pprint import pprint

import os
import sys
import time

if __name__ == '__main__':
    sys.path.append(os.getcwd())
from lib.utils.utils import check_dir
from lib.dataset.tif_dataset import norm_min_max

def read_cfg(args):
    with open(args.cfg, 'r') as f:
        yaml = YAML(typ='safe', pure=True)
        yml = yaml.load(f)
    cmd = [c[1:] for c in sys.argv if c[0]=='-']
    for k,v in yml.items():
        if k not in cmd:
            args.__dict__[k] = v
    return args
    
def list_tif_file(dir):
    names = []
    for file_name in os.listdir(dir):
        if file_name.lower().endswith(('.tif', '.tiff')):
            names.append(file_name)
    return names

def tensor2ndarr(x:torch.Tensor):
    return x.detach().cpu().numpy()

def get_ROI(shape, piece_size, overlap):
    D, H, W = shape
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

    ROI = []
    for d_idx in range(0, D, piece_size-overlap):
        d_s, d_e, d_cL, d_cR = cal_index(d_idx, D)
        for h_idx in range(0, H, piece_size-overlap):
            h_s, h_e, h_cL, h_cR = cal_index(h_idx, H)
            for w_idx in range(0, W, piece_size-overlap):
                w_s, w_e, w_cL, w_cR = cal_index(w_idx, W)        
                ROI.append([[d_s,d_e,d_cL,d_cR], [h_s,h_e,h_cL,h_cR], [w_s,w_e,w_cL,w_cR]])
                if w_e==W: break
            if h_e==H: break
        if d_e==D: break

    return ROI

def process_oneImage(model:torch.nn.Module, directory:str, file_name:str, save_path:str, device:torch.device, args:argparse.Namespace, *, n_idx:int=0):
    overlap = args.overlap
    piece_size = args.piece_size

    # load data
    if file_name.endswith(('tif', '.tiff')):
        real_A = tiff.imread(os.path.join(directory, file_name)).astype(np.float32)
        fake_B = np.zeros_like(real_A)
    elif file_name.endswith('ims'):
        dataset_key = 'DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'
        SRC_IMS = h5py.File(os.path.join(directory, file_name), 'r')
        real_A = SRC_IMS[dataset_key]
        DST_IMS = h5py.File(os.path.join(save_path, f'SR_{file_name}'), 'w')
        fake_B = DST_IMS.create_dataset(dataset_key, shape=real_A.shape, chunks=real_A.chunks, dtype=real_A.dtype)

    # process data
    ROI = get_ROI(shape=real_A.shape, piece_size=overlap, overlap=piece_size)
    for roi in tqdm(ROI, desc=f'[{n_idx:>2}] {file_name:<10}'):
        [d_s,d_e,d_cL,d_cR], [h_s,h_e,h_cL,h_cR], [w_s,w_e,w_cL,w_cR] = roi
        piece_in = real_A[d_s:d_e, h_s:h_e, w_s:w_e]
        # preprocess
        piece_in, min_value, max_value = norm_min_max(piece_in, return_info=True)
        piece_in = torch.from_numpy(piece_in)[None,None].to(device)

        piece_out = model.G_A(piece_in)
        piece_out = torch.clip(piece_out, 0, 1)
        piece_out = tensor2ndarr(piece_out[0,0])
        # postprocess
        piece_out = piece_out*(max_value-min_value) + min_value

        piece_out = piece_out[d_cL:piece_size-d_cR, h_cL:piece_size-h_cR, w_cL:piece_size-w_cR]                
        fake_B[d_s+d_cL:d_e-d_cR, h_s+h_cL:h_e-h_cR, w_s+w_cL:w_e-w_cR] = piece_out
    
    if file_name.endswith(('tif', '.tiff')):
        tiff.imwrite(os.path.join(save_path, f'SR_{file_name}'), fake_B.astype(np.uint16),
                     compression='zlib', compressionargs={'level': 8})
        if args.debug:
            residual = real_A - fake_B
            tiff.imwrite(os.path.join(save_path, f'Residual_{file_name}'), residual.astype(np.float16),
                         compression='zlib', compressionargs={'level': 8})

def main(simulated_args=None):
    start_time = time.time()
    # === args ===
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-ckpt_path', type=str)
    parser.add_argument('-img_path', type=str)
    parser.add_argument('-save_path', type=str, default='inference/result')
    parser.add_argument('-piece_size', type=int, default=64)
    parser.add_argument('-overlap', type=int, default=16)
    parser.add_argument('-debug', action='store_true')

    if simulated_args:
        args = parser.parse_args(simulated_args)
    else:
        args = parser.parse_args()
    args = read_cfg(args)

    # for inference
    # data
    img_path = args.img_path
    save_path = args.save_path
    ckpt_path = args.ckpt_path
    # === === ===

    # load ckpt
    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # load model
    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    model = get_model(args).to(device)
    if 'model' in ckpt.keys():
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
    else:
        model.G_A.load_state_dict(ckpt)
    model.eval()

    check_dir(save_path)
    pprint(vars(args))
    with torch.no_grad():
        if os.path.isfile(img_path):
            file_name = os.path.basename(img_path)
            directory = os.path.dirname(img_path)
            process_oneImage(model, directory, file_name, save_path, device, args, n_idx=0)
        else:
            directory = img_path
            img_name_list = list_tif_file(img_path)
            for n_idx, file_name in tqdm(enumerate(img_name_list)):
                process_oneImage(model, directory, file_name, save_path, device, args, n_idx=n_idx)
    
    end_time = time.time()
    print(f'total time: {time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))}')

if __name__ == '__main__':
    # for test
    # simulated_args = ['-cfg', 'config/VideoSD.yaml', '-ckpt_path', 'out/weights/VideoSD/Epoch_0200.pth', 
    #                   '-video_dir', '/home/ryuuyou/E5/project/data/VideoSD/snr10', '-save_dir', 'eval/result/snr10']
    simulated_args = None
    main(simulated_args)