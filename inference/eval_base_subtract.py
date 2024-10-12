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
from lib.dataset.tif_dataset import normalize
from lib.arch.RESIN_base_subtract import RESIN_base_subtract

def main():
    # img_path = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/val_64'
    img_path = 'data/RESIN/neuron/val_64'
    img_path = 'data/RESIN/neuron/val_128'
    model_name = 'neuron_base_subtract'
    epoch = 2000
    ckpt_path = f'out/weights/{model_name}/Epoch_{str(epoch).zfill(4)}.pth'

    save_base_path = 'inference/result'
    save_path = os.path.join(save_base_path, f'{model_name}_{str(epoch).zfill(4)}')
    check_dir(save_path)

    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RESIN_base_subtract().to(device)
    model.eval()
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

    with torch.no_grad():
        img_name_list = os.listdir(img_path)
        for name in tqdm(img_name_list):
            real_A = tiff.imread(os.path.join(img_path, name)).astype(np.float32)
            real_A = normalize(real_A, 'min_max')
            real_A = torch.from_numpy(real_A)[None,None].to(device)
            fake_B = model.G_A(real_A)
            tiff.imwrite(os.path.join(save_path, f'fake_{name}'), tensor_to_ndarr(fake_B[0,0]).astype(np.float16))
            # rec_A = model.G_B(fake_B)
            # tiff.imwrite(os.path.join(save_path, f'rec_{name}'), tensor_to_ndarr(rec_A[0,0]).astype(np.float16))
if __name__ == '__main__':
    main()