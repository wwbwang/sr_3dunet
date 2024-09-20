import argparse
import numpy as np
import os
import time
import torch
import tifffile
from os import path as osp
from tqdm import tqdm
from functools import partial

from sr_3dunet.utils.data_utils import preprocess, postprocess, get_rotated_img, get_anti_rotated_img, str2bool
from sr_3dunet.utils.bigimg_utils import handle_bigtif
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def remove_outer_layer(img, remove_size):
    height, width, depth = img.shape
    removed_matrix = img[remove_size:height-remove_size, remove_size:width-remove_size, remove_size:depth-remove_size]
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

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='The folder path containing your test TIFF files.')
    parser.add_argument('-o', '--output', type=str, default='results', help='The folder path for the model\'s output TIFF files; this folder may not already exist.')
    parser.add_argument('--model_path', type=str, help='The path to your Restoration model file (.pth).')
    parser.add_argument('--model_back_path', type=str, help='The path to the simulated degradation model file obtained through cycle-consistency training, which aids in visualizing more training results.')
    parser.add_argument('--piece_flag', type=str2bool, default=False, help='Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.')
    parser.add_argument('--piece_size', type=int, default=128, help='Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.')
    parser.add_argument('--piece_overlap', type=int, default=16, help='Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.')
    parser.add_argument('--remove_size', type=int, default=16, help='The number of pixels to be trimmed from the outermost edge.')
    parser.add_argument('--rotated_flag', type=str2bool, default=False, help='Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).')
    args = parser.parse_args()
    
    percentiles=[0, 1]
    dataset_mean=0
    min_value = 0
    max_value = 65535

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_back = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

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

    img_path_list = os.listdir(args.input)
    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')
    
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path))
        img = np.clip(img, min_value, max_value)
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
        print("avg-time_model:", (end_time-start_time)*1000, "N, C, H, W, D:", origin_shape)
        
        rec1_img = model_back(out_img)
        out_affine_img = out_img.transpose(-1, -3)
        C_img = model_back(out_affine_img)
        C_affine_img = C_img.transpose(-1, -3)
        rec_outimg = model(C_img).transpose(-1, -3)
        rec2_img = model_back(rec_outimg)
        
        if args.rotated_flag:
            out_img = get_anti_rotated_img(out_img[0,0].cpu().numpy(), origin_shape)
            rec1_img = get_anti_rotated_img(rec1_img[0,0].cpu().numpy(), origin_shape)
            out_affine_img = get_anti_rotated_img(out_affine_img[0,0].cpu().numpy(), origin_shape)
            C_img = get_anti_rotated_img(C_img[0,0].cpu().numpy(), origin_shape)
            C_affine_img = get_anti_rotated_img(C_affine_img[0,0].cpu().numpy(), origin_shape)
            rec_outimg = get_anti_rotated_img(rec_outimg[0,0].cpu().numpy(), origin_shape)
            rec2_img = get_anti_rotated_img(rec2_img[0,0].cpu().numpy(), origin_shape)
        else:
            out_img = out_img[0,0].cpu().numpy()
            rec1_img = rec1_img[0,0].cpu().numpy()
            out_affine_img = out_affine_img[0,0].cpu().numpy()
            C_img = C_img[0,0].cpu().numpy()
            C_affine_img = C_affine_img[0,0].cpu().numpy()
            rec_outimg = rec_outimg[0,0].cpu().numpy()
            rec2_img = rec2_img[0,0].cpu().numpy()
        
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
        
        pbar1.update(1)

if __name__ == '__main__':
    main()
