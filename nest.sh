#!/bin/bash
#
#SBATCH --job-name=NEST
#SBATCH --output=current_%A_%a.out
#SBATCH --time=550:00
##SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=owners

ml py-numpy/1.17.2_py36
#mpiexec -n 20 --oversubscribe python3 lens_fit2.py chains22 44400 45800 48300 49900 --freqs 4.8 --freqs 8.0 --freqs 14.5 --freqs 22.0 --freqs 37.0
#mpiexec -n 20 --oversubscribe python3 lens_fit2.py chains+ 54860 55500 56550 57200 --freqs 15.0
#mpiexec -n 22 --oversubscribe python3 lens_fit2.py chains3_456 54880 55400 --day2 56650 57240 --day3 57970 58330 --freqs 15.0 --freqs 37.0
#mpiexec -n 24 --oversubscribe python3 lens_fit2.py chains3_45_joint 54870 55330 --day2 56680 57190 --freqs 15.0
#mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains45_joint_detrend_open_negK 54870 57190 --day2 54870 57190 --freqs 15.0 --freqs 37.0 --detrend --fix toff
mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains2_wide_theta 48835 49850 --freqs 14.5 --freqs 22.0 #--day2 50800 52070 --freqs 14.5 --freqs 22.0 --detrend --fix toff
#mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains1_wide_detrend 44800 45750 --freqs 8.0 --freqs 14.5 --detrend
#mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains23_jointfull_wide_detrend 48835 52070 --day2 48835 52070 --freqs 14.5 --freqs 22.0 --detrend --fix toff
#mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains45_joint_detrend_open_theta3 54870 57190 --day2 54870 57190 --freqs 15.0 --freqs 37.0 --detrend --fix toff --dynesty

#python3 direct.py D_test 54870 57190 --day2 54870 57190 --freqs 15.0 --freqs 37.0 --detrend --method ip --fix K

#mpiexec -n 44 --oversubscribe python3 lens_fit2.py chains45_full_detrend_open 54870 57190 --freqs 15.0 --freqs 37.0 --detrend
#mpiexec -n 24 --oversubscribe python3 lens_fit2.py chains3_45_sum 54870 57190 --day2 54870 57190 --freqs 15.0 --freqs 37.0 --detrend --resume
#mpiexec -n 24 --oversubscribe python3 rot_fit.py LS_cyl_PA 2 1 1 0.0 0.0 --ratio_range 0.01
# MIN_DAY1 = 44790
# MAX_DAY1 = 45530
# MIN_DAY2 = 48860
# MAX_DAY2 = 49860
# MIN_DAY3 = 51100
# MAX_DAY3 = 52150

# MIN_DAY1 = 54880
# MAX_DAY1 = 55400
# MIN_DAY2 = 56650
# MAX_DAY2 = 57240
# MIN_DAY3 = 57970
# MAX_DAY3 = 58330
# MIN_DAY3 = 58520
# MAX_DAY3 = 59000
