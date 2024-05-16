#!/bin/bash

#SBATCH --job-name=inference_cellbody
#SBATCH --nodelist=c001
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16

source activate MPCN

# TODO num_io_consumer half

# i: Path to a single H5 file
# o: Path to the output single H5 file
# h5_dir: Dictionary of images in the specified H5 file
# model_path:
# piece_size: Determining the size of smaller images
# piece_overlap: Overlap between neighboring small images

CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_h5.py \
    -i /share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1/CJ4ROI1.ims \
    -o datasets/cellbody/out_h5/output_res0.h5 \
    --h5_dir "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data" \
    --model_path weights/MPCN_VISoR_oldbaseline_cellbody_correctproj_net_g_80000.pth \
    --piece_size 128 --piece_overlap 16
    