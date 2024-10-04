#!/bin/bash

# -i, --input: The folder path containing your test TIFF files.
# -o, --output: The folder path for the model's output TIFF files; this folder may not already exist.
# --model_path: The path of your Restoration model file (.pth).
# --piece_flag: Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.
# --piece_size: Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.
# --piece_overlap: Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.
# --rotated_flag: Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).

branch_name="nissl_base"
iter="24000"

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_folderTIFFs.py \
    -i /home/ryuuyou/E5/project/data/RESIN_datasets/NISSL/val_datastes \
    -o results/${branch_name}_net_g_${iter} \
    --model_path ./experiments/${branch_name}/models/net_g_A_${iter}.pth \
    --piece_flag True --piece_size 64 --piece_overlap 16 --rotated_flag False