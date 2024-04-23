#!/bin/bash

#SBATCH --job-name=MPCN_VISoR_cellbody
#SBATCH --nodelist=c002
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source activate MPCN

CUDA_VISIBLE_DEVICES=0,1 srun\
 --ntasks-per-node=2 --mpi=pmi2 python sr_3dunet/train.py \
 -opt options/MPCN_VISoR_noA2C.yml --launcher="slurm" # --auto_resume

# CUDA_VISIBLE_DEVICES=0,1 srun\
#  --ntasks-per-node=2 --mpi=pmi2 python sr_3dunet/train.py \
#  -opt options/MPCN_VISoR_noA2C.yml --launcher="slurm" # --auto_resume

# CUDA_VISIBLE_DEVICES=0,1 srun --ntasks-per-node=2\
#  --mpi=pmi2 python sr_3dunet/train.py -opt options/MPCN_VISoR_cellbody.yml --launcher="slurm"