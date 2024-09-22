import argparse
import numpy as np
import os
import torch
import h5py
import math
from tqdm import tqdm

from sr_3dunet.utils.data_utils import preprocess, postprocess
from sr_3dunet.utils.inference_utils import get_inference_model

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Your test h5/ims file\'s path.')
    parser.add_argument('-o', '--output', type=str, default='results', help='Your model\'s output h5/ims file\'s path.')
    parser.add_argument('--model_path', type=str, help='The path of your Restoration model file (.pth).')
    parser.add_argument('--piece_size', type=int, default=128, help='Defines the dimensions of the smaller image segments.')
    parser.add_argument('--piece_overlap', type=int, default=16, help='Indicates the overlap area between adjacent smaller image segments.')
    args = parser.parse_args()
    
    percentiles=[0, 1]
    dataset_mean=0
    min_clip = 0
    max_clip = 65535

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    h5 = h5py.File(args.input, 'r')
    img_path = args.h5_dir.split('/')
    img_total = h5
    for key in img_path:
        img_total = img_total[key]
    h, w, d = img_total.shape
    
    # prepare output dir
    if not os.path.exists(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        f.create_dataset(args.h5_dir, shape=img_total.shape, chunks=(1, 256, 256), dtype=img_total.dtype)
    
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
                img = np.clip(img, min_clip, max_clip)
                img, min_value, max_value = preprocess(img, percentiles, dataset_mean)
                
                end_h = h if end_h>h else end_h
                end_w = w if end_w>w else end_w
                end_d = d if end_d>d else end_d
                
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

                out_img = model(img)
                out_img = out_img[:,:,0+h_cutleft:end_h-start_h-h_cutright, 0+w_cutleft:end_w-start_w-w_cutright, 0+d_cutleft:end_d-start_d-d_cutright]

                out_img = out_img[0,0].cpu().numpy()
                out_img = postprocess(out_img, min_value, max_value, dataset_mean)
                
                with h5py.File(args.output, 'r+') as f:
                    f[args.h5_dir][start_h+h_cutleft:end_h-h_cutright, start_w+w_cutleft:end_w-w_cutright, start_d+d_cutleft:end_d-d_cutright] = out_img

                pbar1.update(1)

if __name__ == '__main__':
    main()
