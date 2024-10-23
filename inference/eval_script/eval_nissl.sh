#!/bin/bash
python inference/eval.py -cfg config/RESIN_nissl.yaml \
                         -weight nissl_fullDS \
                         -epoch 80 \
                         -img_path data/RESIN/nissl/val \
                         -big_tif
