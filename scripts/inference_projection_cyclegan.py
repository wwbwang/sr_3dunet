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
from sr_3dunet.archs.projection_cyclegan_arch import Real2Fake_Generator

from sr_3dunet.utils.inference_base import get_base_argument_parser
from sr_3dunet.utils.video_util import frames2video
from sr_3dunet.utils.data_utils import preprocess, postprocess, get_projection
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img


def get_inference_model(args, device) -> Real2Fake_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = Real2Fake_Generator(input_nc=1, output_nc=1, ngf=64)

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
    parser = get_base_argument_parser()
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--aniso_dimension', type=int, help='aniso_dimension')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))

    # prepare output dir
    os.makedirs(os.path.join(args.output, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "output"), exist_ok=True)

    img_path_list = os.listdir(args.input)

    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    num_imgs = len(img_path_list)       # 17
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path))        
        img, min_value, max_value = preprocess(img)
        img = img.astype(np.float32)[None, None, ]
        _, _, img = get_projection(img, args.aniso_dimension)
        tifffile.imwrite(os.path.join(args.output, "input", img_path), img)
        img = torch.from_numpy(img).to(device)
        out = model(img)[0][0]
        out = postprocess(out.cpu().numpy(), min_value, max_value)
        tifffile.imwrite(os.path.join(args.output, "output", img_path), out)
        pbar1.update(1)

if __name__ == '__main__':
    main()