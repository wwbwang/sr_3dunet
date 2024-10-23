#!/bin/bash
python inference/eval.py -cfg config/RESIN_small.yaml \
                         -weight neuron_small \
                         -epoch 500 \
                         -img_path data/RESIN/neuron/val
                         # -big_tif
