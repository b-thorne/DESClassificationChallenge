import argparse
from torchinfo import summary
from src.model import BinaryClassifierCNN
from src.data import load_and_split_dataset
from src.train import do_training
from src.platform import set_device

import logging

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

from pathlib import Path
import wandb

# Create the argparse object
parser = argparse.ArgumentParser(description='DES transient classification')

parser.add_argument('--DEBUG', action='store_true', help='Set DEBUG flag')
parser.add_argument('--mode', choices=['training', 'evaluation'], help='Set mode to training or evaluation')
parser.add_argument('--data-dir', type=Path, required=True, help='Path to DES data directory.')
parser.add_argument('--labels-path', type=Path, required=True, help='Path to labels.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch-size', type=float, default=32, help='Batch size')
parser.add_argument('--train-length', type=int, default=10_000, help='Train length')
parser.add_argument('--test-length', type=int, default=1_000, help='Test length')
parser.add_argument('--val-length', type=int, default=1_000, help='Val length')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')

# Parse the command-line arguments
ARGS = parser.parse_args()


def main():

    # Initialize W&B
    wandb.init(project='des_transient_classification', config=ARGS)

    # Check if DEBUG flag is set
    if ARGS.DEBUG:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

    trn_length = ARGS.train_length
    tst_length = ARGS.test_length
    val_length = ARGS.val_length
        
    # Check the mode flag
    if ARGS.mode == 'training':
        print('Training mode')
        model = BinaryClassifierCNN()
        wandb.watch(model, log='all')

        learning_rate = ARGS.learning_rate
        batch_size = ARGS.batch_size
        data_dir = ARGS.data_dir
        labels_path = ARGS.labels_path
        epochs = ARGS.epochs
    
        trn, tst, val = load_and_split_dataset(data_dir, labels_path, trn_length, tst_length, val_length, batch_size)
        logging.debug(f"Reading input features from: {data_dir}") 
        logging.debug(f"Reading labels from: {labels_path}")
       
        print("Starting training ...")
        logging.debug(f"Learning rate: {learning_rate}" )
        logging.debug(f"Batch size: {batch_size}")

        metric = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        device = set_device()
        logging.debug(f"Using device: {device}")
        model = do_training(model, optimizer, metric, trn, tst, device, epochs)
        wandb.save('checkpoints/final_model.pt')

    elif ARGS.mode == 'evaluation':
        print('Evaluation mode')
    else:
        print('Please specify a valid mode')

    return

if __name__ == "__main__":
    main()
     
