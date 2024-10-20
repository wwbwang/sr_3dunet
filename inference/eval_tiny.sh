#!/bin/bash
python inference/eval.py -cfg config/RESIN_tiny.yaml \
                        -weight neuron_tiny \
                        -epoch 330 \
                        -img_path data/RESIN/neuron/val \
                        # -big_tif
