#!/bin/bash
python inference/inference_single.py -cfg config/RESIN_nissl.yaml \
                                     -ckpt_path out/weights_GA/nissl_fullDS/Epoch_0080.pth \
                                     -img_path data/RESIN/nissl/val/standard_128.tif \
                                     -save_dir inference/result/nissl \
                                     -save_name RESIN_output.tif \
