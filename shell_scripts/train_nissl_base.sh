#!/bin/bash

#SBATCH --job-name=nissl_base
#SBATCH --nodelist=mn00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8

source activate torch

CUDA_VISIBLE_DEVICES=0,1,2,3 srun\
 --mpi=pmi2 python sr_3dunet/train.py \
 -opt options/nissl_base_45d.yml --launcher="slurm" # --auto_resume
