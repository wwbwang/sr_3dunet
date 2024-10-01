#!/bin/bash

#SBATCH --job-name=neuron_tiny
#SBATCH --nodelist=c002
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

source activate torch

CUDA_VISIBLE_DEVICES=0,1 srun\
 --mpi=pmi2 python sr_3dunet/train.py \
 -opt options/neuron_45d.yml --launcher="slurm" # --auto_resume
