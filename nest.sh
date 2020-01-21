#!/bin/bash
#
#SBATCH --job-name=NEST
#SBATCH --output=current_%A_%a.out
#SBATCH --time=350:00
##SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=owners

ml py-numpy/1.17.2_py36
#mpiexec -n 20 --oversubscribe python3 lens_fit2.py chains22 44400 45800 48300 49900 --freqs 4.8 --freqs 8.0 --freqs 14.5 --freqs 22.0 --freqs 37.0
mpiexec -n 20 --oversubscribe python3 lens_fit2.py chains+ 54860 55500 56550 57200 --freqs 15.0

