import argparse
import numpy as np
import os
import time
import torch
import tifffile
from os import path as osp
from tqdm import tqdm
from functools import partial

from sr_3dunet.utils.data_utils import str2bool
from sr_3dunet.utils.inference_utils import get_inference_model_debug, handle_bigtif, handle_smalltif, remove_outer_layer

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='The folder path containing your test TIFF files.')
    parser.add_argument('-o', '--output', type=str, default='results', help='The folder path for the model\'s output TIFF files; this folder may not already exist.')
    parser.add_argument('--model_path', type=str, help='The path of your Restoration model file (.pth).')
    parser.add_argument('--model_back_path', type=str, help='The path of the simulated degradation model file obtained through cycle-consistency training, which aids in visualizing more training results.')
    parser.add_argument('--piece_flag', type=str2bool, default=False, help='Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.')
    parser.add_argument('--piece_size', type=int, default=128, help='Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.')
    parser.add_argument('--piece_overlap', type=int, default=16, help='Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.')
    parser.add_argument('--remove_size', type=int, default=16, help='The number of pixels to be trimmed from the outermost edge.')
    parser.add_argument('--rotated_flag', type=str2bool, default=False, help='Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).')
    args = parser.parse_args()
    
    percentiles=[0, 1]
    dataset_mean=0
    min_clip = 0
    max_clip = 65535

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_back = get_inference_model_debug(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    if args.piece_flag:
        model = partial(handle_bigtif, model, args.piece_size, args.piece_overlap, args.rotated_flag, percentiles, dataset_mean, device)
        model_back = partial(handle_bigtif, model_back, args.piece_size, args.piece_overlap, args.rotated_flag, percentiles, dataset_mean, device)
    else:
        model = partial(handle_smalltif, model, args.piece_size, args.piece_overlap, args.rotated_flag, percentiles, dataset_mean, device)
        model_back = partial(handle_smalltif, model_back, args.piece_size, args.piece_overlap, args.rotated_flag, percentiles, dataset_mean, device)

    # prepare output dir
    os.makedirs(os.path.join(args.output, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "output"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rec"), exist_ok=True)        # FIXME, 可能和真实处理过程有出入

    img_path_list = os.listdir(args.input)
    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path))
        img = np.clip(img, min_clip, max_clip)
        # img = img.transpose(1,0,2)
        # img = np.flip(img, axis=-1)
        origin_shape = img.shape
        tifffile.imwrite(os.path.join(args.output, "input", "input" + img_path),
                         remove_outer_layer(img, args.remove_size).astype(np.uint16))

        start_time = time.time()
        torch.cuda.synchronize()
        out_img = model(img)
        # out_img = out_img.transpose(1,0,2)
        torch.cuda.synchronize()
        end_time = time.time()
        print("avg-time_model:", (end_time-start_time)*1000, "ms,", "N, C, H, W, D:", origin_shape)

        rec_img = model_back(out_img)
        # rec_img = rec_img.transpose(1,0,2)
        
        tifffile.imwrite(os.path.join(args.output, "output", "output" + img_path),
                         remove_outer_layer(out_img, args.remove_size))
        tifffile.imwrite(os.path.join(args.output, "rec", "rec" + img_path),
                         remove_outer_layer(rec_img, args.remove_size))
        pbar1.update(1)

if __name__ == '__main__':
    main()
