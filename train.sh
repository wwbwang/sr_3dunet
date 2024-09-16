#!/bin/bash

#SBATCH --job-name=40X
#SBATCH --nodelist=c003
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8

source activate RESIN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun\
 --ntasks-per-node=8 --mpi=pmi2 python sr_3dunet/train.py \
 -opt options/MPCN_simulation.yml --launcher="slurm" # --auto_resume

 # MPCN_VISoR_40X