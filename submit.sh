#!/bin/bash
#
#SBATCH --job-name=testgpu
#SBATCH --output=current_%A_%a.out
#SBATCH --error=current_%A_%a.err
#SBATCH --time=820:00
##SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=iric
#SBATCH --array=1 ##these are the SLURM task ids

ml python/3.6.1
ml py-numpy/1.14.3_py36
ml py-tensorflow/1.12.0_py36
ml imkl/2019
ml viz
ml py-matplotlib/2.1.2_py36

python3 train.py 64
