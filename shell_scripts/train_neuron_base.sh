#!/bin/bash

#SBATCH --job-name=neuron_base
#SBATCH --nodelist=c001
#SBATCH --gres=gpu:3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8

source activate torch

CUDA_VISIBLE_DEVICES=0,1,2 srun\
 --mpi=pmi2 python sr_3dunet/train.py \
 -opt options/neuron_base_45d.yml --launcher="slurm" # --auto_resume
