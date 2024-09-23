#!/bin/bash

# -i, --input: The folder path containing your test TIFF files.
# -o, --output: The folder path for the model's output TIFF files; this folder may not already exist.
# --model_path: The path of your Restoration model file (.pth).
# --piece_flag: Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.
# --piece_size: Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.
# --piece_overlap: Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.
# --rotated_flag: Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_folder.py \
    -i /share/home/wangwb/workspace/sr_3dunet/datasets/rm009_labeled/bg \
    -o /share/home/wangwb/workspace/sr_3dunet/datasets/rm009_labeled/bg_sr \
    --model_path /share/home/wangwb/workspace/sr_3dunet/weights/MPCN_VISoR_oldbaseline_256_net_g_A_140000.pth \
    --piece_flag True --piece_size 128 --piece_overlap 16 --rotated_flag False