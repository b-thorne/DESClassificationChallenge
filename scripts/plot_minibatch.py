import argparse

from pathlib import Path

from src.plotting import plot_batch
from src.data import load_and_split_dataset

parser = argparse.ArgumentParser(description="Plotting of thumbnails")

parser.add_argument(
    "--mode", choices=["raw", "inference"], help="Set mode to training or inferece"
)
parser.add_argument(
    "--data-dir", type=Path, required=True, help="Path to DES data directory."
)
parser.add_argument("--labels-path", type=Path, required=True, help="Path to labels.")

ARGS = parser.parse_args()


def main():
    trn_length = 1000
    tst_length = 100
    val_length = 100
    batch_size = 5
    trn, _, _ = load_and_split_dataset(
        ARGS.data_dir, ARGS.labels_path, trn_length, tst_length, val_length, batch_size
    )

    fig = plot_batch(next(enumerate(trn))[1])
    fig.savefig(f"plots/thumbnails_nbatch{batch_size}.png", bbox_inches="tight")
    return


if __name__ == "__main__":
    main()
