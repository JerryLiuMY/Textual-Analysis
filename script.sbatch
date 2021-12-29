#!/bin/bash
#SBATCH --job-name=script
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=4000
#SBATCH --time=36:00:00

srun --nodes=1 python3 run.py -f 0 -t 12
