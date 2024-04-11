#!/bin/bash

#SBATCH --job-name=MPCN_VISoR_cellbody
#SBATCH --nodelist=c001
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source activate MPCN

# CUDA_VISIBLE_DEVICES=0,1 srun\
#  --ntasks-per-node=2 --mpi=pmi2 python sr_3dunet/train.py \
#  -opt options/MPCN_VISoR.yml --launcher="slurm"

CUDA_VISIBLE_DEVICES=0 srun --ntasks-per-node=1\
 --mpi=pmi2 python sr_3dunet/train.py -opt options/MPCN_VISoR_cellbody.yml --launcher="slurm"

# CUDA_VISIBLE_DEVICES=0,1 srun --ntasks-per-node=2\
#  --mpi=pmi2 python sr_3dunet/train.py -opt options/MPCN_VISoR_cellbody.yml --launcher="slurm"