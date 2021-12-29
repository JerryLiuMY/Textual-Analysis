#!/bin/bash
#SBATCH --job-name=script
#SBATCH --partition=broadwl
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=4G
#SBATCH --time=36:00:00

srun --nodes=1 python3 run.py -f 0 -t 8
srun --nodes=1 python3 run.py -f 8 -t 16
srun --nodes=1 python3 run.py -f 16 -t 24
srun --nodes=1 python3 run.py -f 24 -t 32
srun --nodes=1 python3 run.py -f 32 -t 37
