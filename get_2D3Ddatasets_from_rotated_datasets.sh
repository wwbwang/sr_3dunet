#!/bin/bash

#SBATCH --job-name=get_2D3Ddatasets_from_rotated_datasets
#SBATCH --nodelist=c001
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16

source activate MPCN

# CUDA_VISIBLE_DEVICES=0,1 srun\
#  --ntasks-per-node=2 --mpi=pmi2 python sr_3dunet/train.py \
#  -opt options/MPCN_VISoR.yml --launcher="slurm"

python /share/home/wangwb/workspace/sr_3dunet/scripts/get_2D3Ddatasets_from_rotated_datasets.py
# CUDA_VISIBLE_DEVICES=0,1 srun --ntasks-per-node=2 --mpi=pmi2 python sr_3dunet/train.py -opt options/MPCN.yml --launcher="slurm"