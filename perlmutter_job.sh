#!/bin/bash
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun /global/u1/b/bthorne/miniforge3/envs/fairuniverse/bin/python run_experiment.py --DEBUG \
    --data-dir /global/cfs/cdirs/m4287/cosmology/dessn/stamps \
    --labels-path /global/cfs/cdirs/m4287/cosmology/dessn/relabeled_autoscan_features.csv \
    --mode training \
    --num-workers 32 \
    --train-length 430000 \
    --test-length 40000 \
    --epochs 30 \
    --weight-decay