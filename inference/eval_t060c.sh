#!/bin/bash
python inference/eval.py -cfg config/RESIN_t060c.yaml \
                        -weight t060c \
                        -epoch 4000 \
                        -img_path data/RESIN/t060c/val \
                        -big_tif
