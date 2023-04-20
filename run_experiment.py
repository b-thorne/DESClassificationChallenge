import argparse
from torchinfo import summary
from src.model import BinaryClassifierCNN
from src.data import DESFitsDataset, load_and_split_dataset

import logging

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

from pathlib import Path

# Create the argparse object
parser = argparse.ArgumentParser(description='DES transient classification')

parser.add_argument('--DEBUG', action='store_true', help='Set DEBUG flag')
parser.add_argument('--mode', choices=['training', 'evaluation'], help='Set mode to training or evaluation')
parser.add_argument('--data-dir', type=Path, required=True, help='Path to DES data directory.')
parser.add_argument('--labels-path', type=Path, required=True, help='Path to labels.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch-size', type=float, default=32, help='Batch size')
parser.add_argument('--train-portion', type=float, default=0.5, help='Train proportion')
parser.add_argument('--test-portion', type=float, default=0.05, help='Test proportion')

# Parse the command-line arguments
ARGS = parser.parse_args()


def main():
    # Check if DEBUG flag is set
    if ARGS.DEBUG:
        print('DEBUG mode enabled')
        train_proportion = ARGS.train_proportion / 100
        test_proportion = ARGS.test_proportion / 100
        
    # Check the mode flag
    if ARGS.mode == 'training':
        print('Training mode')
        model = BinaryClassifierCNN()
        print(summary(model))

        learning_rate = ARGS.learning_rate
        batch_size = ARGS.batch_size
        training_frac = ARGS.training_frac 
        data_dir = ARGS.data_dir
        labels_path = ARGS.labels_path
    
        train_dloader, test_dloader = load_and_split_dataset(data_dir, labels_path, train_proportion, test_proportion)
       logging.debug("Reading input features from: {data_dir}") 
       logging.debug("Reading labels from: {labels_path}")
       print("Starting training ...")
    elif ARGS.mode == 'evaluation':
        print('Evaluation mode')
    else:
        print('Please specify a valid mode')

    return

if __name__ == "__main__":
    main()
     
