# flake8: noqa
import os.path as osp

import sr_3dunet.archs
import sr_3dunet.data
import sr_3dunet.models
from basicsr.train import train_pipeline
import torch
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
