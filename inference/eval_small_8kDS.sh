#!/bin/bash
python inference/eval.py -cfg config/RESIN_small_8kDS.yaml \
                        -weight neuron_small_8kDS \
                        -epoch 510 \
                        -img_path data/RESIN/neuron/val_128 \
                        -big_tif
                        # -img_path data/RESIN/neuron/val_64
