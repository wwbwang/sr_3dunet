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
# from lib.arch.RESIN_prelu import RESIN

def main():
    img_path = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron_45d/test'
    model_name = 'neuron_dev'
    epoch = 595
    ckpt_path = f'out/weights/{model_name}/Epoch_{str(epoch).zfill(4)}.pth'

    save_base_path = 'inference/result'
    save_path = os.path.join(save_base_path, f'{model_name}_{str(epoch).zfill(4)}')
    check_dir(save_path)

    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RESIN().to(device)
    model.eval()
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

    with torch.no_grad():
        img_name_list = os.listdir(img_path)
        for name in tqdm(img_name_list):
            real_A = tiff.imread(os.path.join(img_path, name)).astype(np.float32)
            real_A = normalize(real_A, 'min_max')
            real_A = torch.from_numpy(real_A)[None,None].to(device)
            fake_B = model.G_A(real_A)
            rec_A = model.G_B(fake_B)
            fake_B = tensor_to_ndarr(fake_B[0,0])
            tiff.imwrite(os.path.join(save_path, f'fake_{name}'), fake_B)
            rec_A = tensor_to_ndarr(rec_A[0,0])
            tiff.imwrite(os.path.join(save_path, f'rec_{name}'), rec_A)
if __name__ == '__main__':
    main()