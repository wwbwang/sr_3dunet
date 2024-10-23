#!/bin/bash
python inference/eval.py -cfg config/RESIN_small_8kDS.yaml \
                         -weight neuron_small_8kDS \
                         -epoch 1115 \
                         -img_path data/RESIN/neuron/val
                         # -big_tif
