#!/bin/bash

# TODO num_io_consumer half 

# Input: Path to directory containing TIFF files
# Output: Path to directory containing output TIFF files, may not exist
# model_path:
# piece_flag: Used in large TIFF files, separate large TIFF files into smaller TIFFs before inference
# piece_size: Effective when piece_Flag is true, determining the size of smaller TIFFs
# piece_overlap: Effective when piece_Flag is true, overlap between neighboring small TIFFs

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_folder_bigtifs.py \
    -i /share/home/wangwb/workspace/sr_3dunet/datasets/axons \
    -o /share/home/wangwb/workspace/sr_3dunet/datasets/axons_sr_uint \
    --model_path /share/home/wangwb/workspace/sr_3dunet/weights/MPCN_VISoR_oldbaseline_256_net_g_A_140000.pth \
    --piece_size 64 --piece_overlap 16