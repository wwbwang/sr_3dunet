#!/bin/bash
python inference/eval.py -cfg config/RESIN_nissl.yaml \
                        -weight nissl \
                        -epoch 480 \
                        -img_path data/RESIN/nissl/val \
                        -big_tif
