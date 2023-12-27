import argparse
import os.path
import torch

from sr_3dunet.archs.unet_3d_arch import UNet_3d_Generator


def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '--expname', type=str, default='animesr', help='A unique name to identify your current inference')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')

    return parser


def get_inference_model(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256, 512], dim=3)

    model_path = args.model_path
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet['params'], strict=True)
    model.eval()
    model = model.to(device)

    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print(num_parameters)
    # exit(0)

    return model.half() if args.half else model
