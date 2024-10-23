#!/bin/bash
python inference/inference_batch.py -cfg config/RESIN_nissl.yaml \
                                    -ckpt_path out/weights_GA/nissl_fullDS/Epoch_0080.pth \
                                    -img_dir data/RESIN/nissl/val \
                                    -save_dir inference/result/nissl \
