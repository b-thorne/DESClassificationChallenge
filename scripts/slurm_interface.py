import argparse
from pathlib import Path
import subprocess
import os
import sys

parser = argparse.ArgumentParser(description="Generate a Slurm batch script.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default="/global/cfs/cdirs/m4287/cosmology/dessn",
    help="Path to the input labels",
)
parser.add_argument(
    "--account", type=str, default="m4287", help="Path to the input features"
)
parser.add_argument("--constraint", type=str, default="gpu")
parser.add_argument("--qos", type=str, default="debug")
parser.add_argument("--time", type=str, default="00:30:00")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--train-length", type=int, default=40000)
parser.add_argument("--test-length", type=int, default=1000)
parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--script-name", type=str, default="batch_job.sh")
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Track experiment with Weights and Biases",
)
parser.add_argument(
    "--wandb-api-key", default="58827ea170a691e73a8be61b84116e864ecbe774"
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=128,
    help="Number of workers used for async data loading",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="info",
    choices=("critical", "error", "warning", "info", "debug"),
    help="logging level",
)

ARGS = parser.parse_args()


def slurm_script():
    return f"""#!/bin/bash
#SBATCH -A {ARGS.account}
#SBATCH -C {ARGS.constraint}
#SBATCH -q {ARGS.qos}
#SBATCH -t {ARGS.time}
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

podman-hpc run --rm --gpu                                                   \\
    --env WANDB_API_KEY={ARGS.wandb_api_key}                                \\
    --volume {os.path.abspath(os.getcwd())}:/app                            \\
    --volume {ARGS.data_dir}:/data                                          \\
    docker.io/bthorne93/des_snae:latest python scripts/run_experiment.py    \\
    --data-dir /data/stamps                                                 \\
    --labels-path /data/relabeled_autoscan_features.csv                     \\
    --mode training                                                         \\
    --num-workers {ARGS.num_workers}                                        \\
    --train-length {ARGS.train_length}                                      \\
    --test-length {ARGS.test_length}                                        \\
    --epochs {ARGS.epochs}                                                  \\
    --batch-size {ARGS.batch_size}                                          \\
    --learning-rate {ARGS.learning_rate}                                    \\
    {"--wandb" if ARGS.wandb else ""}                                       \\
    --verbosity {ARGS.verbosity}
"""


def main():
    filepath = Path("/tmp") / ARGS.script_name
    with open(filepath, "w") as f:
        f.write(slurm_script())
    sys.exit(subprocess.Popen(["sbatch", filepath]).wait())


if __name__ == "__main__":
    main()
