#!/bin/bash
#SBATCH --job-name=script
#SBATCH --partition=broadwl-lc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=4G
#SBATCH --time=36:00:00

srun --nodes=1 python3 run.py -f 0 -t 37
