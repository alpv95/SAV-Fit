#!/bin/bash
#
#SBATCH --job-name=NEST
####SBATCH --output=current_%A_%a.out
#SBATCH --time=1250:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=owners

source setup.sh

SAV1=0.25
SAV2=0.15
SAV4=1
SAV5=1
TYPE=1
LENGTH=short
SMOOTH=0.05

mpiexec -n $SLURM_NTASKS python3 lens_fit.py All-$SAV1-$SAV2-$SAV4-$SAV5-$TYPE-$SMOOTH-$LENGTH-dt-loweff $SAV1 $SAV2 $SAV4 $SAV5 --efficiency 0.12 --live 2000 --const --smooth $SMOOTH --type $TYPE --length $LENGTH --detrend

