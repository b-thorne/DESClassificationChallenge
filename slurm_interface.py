import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description="Generate a Slurm batch script.")
parser.add_argument("features_path", type=Path, help="Path to the input features")
parser.add_argument("labels_path", type=Path, help="Path to the input labels")

ARGS = parser.parse_args()

def generate_slurm_script(current_dir, features_path, labels_path):
    slurm_template = f"""#!/bin/bash
#SBATCH -A m4287
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

podman-hpc run --rm --gpu  \\ 
    --env WANDB_API_KEY=58827ea170a691e73a8be61b84116e864ecbe774 \\
    --volume {current_dir}:/app \\
    docker.io/bthorne93/des_snae:latest python run_experiment.py --DEBUG \\
    --data-dir {features_path} \\
    --labels-path {labels_path} \\
    --mode training \\
    --num-workers 128 \\
    --train-length 430000 \\
    --test-length 40000 \\
    --epochs 30 \\
    --batch-size 16 \\
    --learning-rate 0.005
"""
    return slurm_template

def main():
    current_dir = os.path.abspath(os.getcwd())
    slurm_script = generate_slurm_script(current_dir, ARGS.features_path, ARGS.labels_path)
    with open("generated_slurm_script.sh", "w") as f:
        f.write(slurm_script)

    print(f"Slurm script generated and saved as generated_slurm_script.sh")

if __name__ == "__main__":
    main()