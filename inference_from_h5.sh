#!/bin/bash

# -i: Your test h5/ims file's path.
# -o: Your model\'s output h5/ims file's path.
# --model_path: The path of your Restoration model file (.pth).
# --piece_size: Defines the dimensions of the smaller image segments.
# --piece_overlap: Indicates the overlap area between adjacent smaller image segments.

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_h5.py \
    -i /share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1/CJ4ROI2.ims \
    -o datasets/NISSL/CJ4ROI2_out_h5_piece64/output_res0.h5 \
    --model_path weights/MPCN_VISoR_NISSL_net_g_A_110000.pth \
    --piece_size 64 --piece_overlap 16
    