import argparse

from src.model import BinaryClassifierCNN
from src.data import DESFitsDataset

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.otpim as optim

from pathlib import Path

# Create the argparse object
parser = argparse.ArgumentParser(description='DES transient classification')

# Add the DEBUG flag
parser.add_argument('--debug', action='store_true', help='Set DEBUG flag')

# Add the mode flag
parser.add_argument('--mode', choices=['training', 'evaluation'], help='Set mode to training or evaluation')

parser.add_argument('--data-dir', type=Path, required=True, help='Path to DES data directory.')

# Parse the command-line arguments
ARGS = parser.parse_args()


def main():
    # Check if DEBUG flag is set
    if ARGS.debug:
        print('DEBUG mode enabled')

    



    # Check the mode flag
    if ARGS.mode == 'training':
        print('Training mode')
    elif ARGS.mode == 'evaluation':
        print('Evaluation mode')
    else:
        print('Please specify a valid mode')

    return

if __name__ == "__main__":
    main()
     
