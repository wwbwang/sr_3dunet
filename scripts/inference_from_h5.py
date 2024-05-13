import argparse
import numpy as np
import os
import time
import torch
import tifffile
import h5py
from os import path as osp
from tqdm import tqdm
from functools import partial

from sr_3dunet.utils.data_utils import preprocess, postprocess
from sr_3dunet.utils.inference_big_tif import handle_bigtif
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
    parser.add_argument('--piece_size', type=int, default=128, help='piece_size')
    parser.add_argument('--piece_overlap', type=int, default=16, help='piece_overlap')
    parser.add_argument('--h5_dir', default='DataSet/ResolutionLevel 0/TimePoint 0/Channel 3/Data', help='Directory of the h5 file, separated by : ')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    model = partial(handle_bigtif, model, args.piece_size, args.piece_overlap)

    percentiles=[0, 0.9999] # [0.01,0.999999] # [0.01, 0.9985]
    dataset_mean=0
    
    h5 = h5py.File(args.input, 'r')
    # img = h5['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 3']['Data']
    img_path = args.h5_dir.split('/')
    img = h5
    for key in img_path:
        img = img[key]
    
    img = np.clip(img, 0, 65535)
    origin_shape = img.shape
    img, min_value, max_value = preprocess(img, percentiles, dataset_mean)

    img = img.astype(np.float32)[None, None,]
    img = torch.from_numpy(img).to(device)     # to float32

    start_time = time.time()
    torch.cuda.synchronize()
    out_img = model(img)
    torch.cuda.synchronize()
    end_time = time.time()
    print("avg-time_model:", (end_time-start_time)*1000, "ms,", "N, C, H, W, D:", origin_shape)

    out_img = out_img[0,0].cpu().numpy()
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset(args.h5_dir, data=postprocess(out_img, min_value, max_value, dataset_mean))
    # tifffile.imwrite(args.output, postprocess(out_img, min_value, max_value, dataset_mean))
        
if __name__ == '__main__':
    main()
