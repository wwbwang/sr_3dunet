#!/bin/bash

#SBATCH --job-name=MPCN_inference
#SBATCH --nodelist=c001
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

branch_name=MPCN_VISoR_nonorm_noclip_disableAug_frozeDimension_archived_20240416_140306
iter='10000000000'  # nan Appoint in terminal

source activate MPCN

# Parsing arguments from the command line
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --iter)
            iter="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Running the Python script with CUDA on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/inference_pretrained.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
    -i /share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/val_datasets\
    -o /share/home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
    --model_back_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
    --piece_flag True --piece_size 128 --piece_overlap 16 --piece_mod_size 16 --remove_size 8 --rotated_flag False
