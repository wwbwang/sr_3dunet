#!/bin/bash

# -i, --input: The folder path containing your test TIFF files.
# -o, --output: The folder path for the model's output TIFF files; this folder may not already exist.
# --model_path: The path of your Restoration model file (.pth).
# --model_back_path: The path of the simulated degradation model file obtained through cycle-consistency training, which aids in visualizing more training results.
# --piece_flag: Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.
# --piece_size: Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.
# --piece_overlap: Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.
# --remove_size: The number of pixels to be trimmed from the outermost edge.
# --rotated_flag: Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).

branch_names=("RESIN_concat_40X")
iters=("110000")

for branch_name in "${branch_names[@]}"
do
    for iter in "${iters[@]}"
    do 
        CUDA_VISIBLE_DEVICES=0 python scripts/inference_debug.py \
            -i /share/home/wangwb/workspace/sr_3dunet/datasets/40X/val_datasets\
            -o /share/home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
            --model_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
            --model_back_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
            --piece_flag True --piece_size 64 --piece_overlap 16 --remove_size 4 --rotated_flag True
    done
done



