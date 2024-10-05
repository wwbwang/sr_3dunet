import numpy as np
import os
import tifffile as tiff
from empatches import EMPatches
import torch
from tqdm import tqdm

from ryu_pytools import tensor_to_ndarr, check_dir

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())
from lib.datasets.tif_dataset import normalize
from lib.arch.RESIN import RESIN

def main():
    img_path = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron_45d/test'
    model_name = 'neuron_dev'
    epoch = 225
    ckpt_path = f'out/weights/{model_name}/Epoch_{str(epoch).zfill(4)}.pth'

    save_base_path = 'inference/result'
    save_path = os.path.join(save_base_path, f'{model_name}_{str(epoch).zfill(4)}')
    check_dir(save_path)

    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RESIN(features_G=[32,64,128]).to(device)
    model.eval()
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

    img_name_list = os.listdir(img_path)
    for name in tqdm(img_name_list):
        img = tiff.imread(os.path.join(img_path, name)).astype(np.float32)
        img = normalize(img, 'abs')
        img_tensor = torch.from_numpy(img)[None,None].to(device)
        res_tensor = model.G_A(img_tensor)
        res = tensor_to_ndarr(res_tensor[0,0])
        tiff.imwrite(os.path.join(save_path, f'res_{name}'), res)

if __name__ == '__main__':
    main()