import os

from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import torch
import pandas as pd

from . import logging
from multiprocessing import Pool, cpu_count
from functools import lru_cache

LOGGER = logging.get_logger(__file__)


class DESFitsDataset(Dataset):
    def __init__(self, root_dir, labels_dict):
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.file_dict = {"srch": {}, "diff": {}, "temp": {}}
        traverse_directory(root_dir, self.file_dict)

        self.file_ids = sorted(
            set(self.file_dict["srch"].keys())
            & set(self.file_dict["diff"].keys())
            & set(self.file_dict["temp"].keys())
        )
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.file_ids)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        srch_file = self.file_dict["srch"][file_id]
        diff_file = self.file_dict["diff"][file_id]
        temp_file = self.file_dict["temp"][file_id]

        srch_image = fits.getdata(srch_file).astype(np.float32)
        diff_image = fits.getdata(diff_file).astype(np.float32)
        temp_image = fits.getdata(temp_file).astype(np.float32)

        stacked_image = np.stack((temp_image, srch_image, diff_image), axis=0)
        if (self.mean is not None) and (self.std is not None):
            stacked_image = (stacked_image - self.mean[:, None, None]) / self.std[
                :, None, None
            ]
        stacked_tensor = torch.from_numpy(stacked_image.astype(np.float32))

        target_label = torch.from_numpy(np.array(np.float32(self.labels_dict[file_id])))
        return stacked_tensor, target_label

    def compute_mean_std(self):
        num_workers = cpu_count()

        # Compute the number of samples per worker
        samples_per_worker = len(self) // num_workers

        # Create a list of arguments for each worker
        worker_args = [
            (
                self,
                i * samples_per_worker,
                (i + 1) * samples_per_worker,
            )
            for i in range(num_workers)
        ]

        # Handle the case when the dataset length is not evenly divisible by the number of workers
        if len(self) % num_workers != 0:
            worker_args[-1] = (
                self,
                (num_workers - 1) * samples_per_worker,
                len(self),
            )

        # Run the _compute_mean_std_chunk function in parallel using a process pool
        with Pool(num_workers) as pool:
            results = list(pool.imap_unordered(_compute_mean_std_chunk, worker_args))

        # Combine the results from all workers
        mean = np.zeros(3)
        std = np.zeros(3)
        n_pixels = 0

        for res_mean, res_std, res_n_pixels in results:
            mean += res_mean
            std += res_std
            n_pixels += res_n_pixels

        mean /= n_pixels
        std /= n_pixels
        std -= mean**2
        std = np.sqrt(std)

        return mean, std


def _compute_mean_std_chunk(args):
    dataset, start, end = args
    mean = np.zeros(3)
    std = np.zeros(3)
    n_pixels = 0

    for idx in range(start, end):
        stacked_tensor, _ = dataset[idx]
        stacked_array = stacked_tensor.numpy()
        n_pixels += stacked_array.size // 3
        mean += stacked_array.sum(axis=(1, 2))
        std += (stacked_array**2).sum(axis=(1, 2))

    return mean, std, n_pixels


def traverse_directory(directory, file_dict):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and ".fits" in entry.name:
                filename = entry.name
                file_id = int(filename.split(".")[0][4:])
                file_type = filename[:4]
                full_path = entry.path

                if file_type in file_dict:
                    file_dict[file_type][file_id] = full_path
            elif entry.is_dir():
                traverse_directory(entry.path, file_dict)


def load_labels(filepath):
    df = pd.read_csv(filepath, usecols=["ID", "OBJECT_TYPE"])
    return dict(zip(df["ID"], df["OBJECT_TYPE"]))


def load_and_split_dataset(
    data_dir,
    labels_path,
    trn_length,
    tst_length,
    val_length,
    batch_size,
    split_seed=1234,
    num_workers=0,
):
    # First load the target labels
    labels_dict = load_labels(labels_path)
    # load the feature dataset
    dataset = DESFitsDataset(data_dir, labels_dict)
    mean, std = dataset.compute_mean_std()
    dataset.mean = mean
    dataset.std = std
    # If we are requesting only a part of the dataset, then randomly select
    # a subset of the correct length.
    total_dataset_size = trn_length + tst_length + val_length
    if len(dataset) > total_dataset_size:
        inds = torch.randperm(len(dataset))[:total_dataset_size]
        dataset = torch.utils.data.Subset(dataset, inds)
    LOGGER.debug(f"Samples in dataset: {len(dataset)}")
    # Define a generator to seed the random splitting function and split
    # the dataset
    generator = torch.Generator().manual_seed(split_seed)
    trn_set, tst_set, val_set = random_split(
        dataset, [trn_length, tst_length, val_length], generator=generator
    )
    # Define the data loaders to be returned with batching
    trn_data_loader = DataLoader(
        trn_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    tst_data_loader = DataLoader(
        tst_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_data_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return trn_data_loader, tst_data_loader, val_data_loader
