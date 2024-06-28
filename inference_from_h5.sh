#!/bin/bash

# TODO num_io_consumer half

# i: Path to a single H5 file
# o: Path to the output single H5 file
# h5_dir: Dictionary of images in the specified H5 file
# model_path:
# piece_size: Determining the size of smaller images
# piece_overlap: Overlap between neighboring small images

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_h5.py \
    -i /share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1/CJ4ROI2.ims \
    -o datasets/NISSL/CJ4ROI2_out_h5_piece64/output_res0.h5 \
    --h5_dir "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data" \
    --model_path weights/MPCN_VISoR_NISSL_net_g_A_110000.pth \
    --piece_size 64 --piece_overlap 16

# CUDA_VISIBLE_DEVICES=1 python scripts/inference_from_h5.py \
#     -i /share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1/CJ4ROI2.ims \
#     -o datasets/cellbody/CJ4ROI2_out_h5/output_res0.h5 \
#     --h5_dir "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data" \
#     --model_path weights/MPCN_VISoR_oldbaseline_cellbody_3projD_256_net_g_A_80000.pth \
#     --piece_size 128 --piece_overlap 16
    