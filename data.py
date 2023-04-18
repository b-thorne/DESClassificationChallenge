import os 
import numpy as np
import torch
from astropy.io import fits 
from torch.utils.data import Dataset

def traverse_directory(directory, file_dict):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and '.fits' in entry.name:
                filename = entry.name
                file_id = int(filename.split('.')[0][4:])
                file_type = filename[:4]
                full_path = entry.path

                if file_type in file_dict:
                    file_dict[file_type][file_id] = full_path
            elif entry.is_dir():
                traverse_directory(entry.path, file_dict)

class DESFitsDataset(Dataset):
    def __init__(self, root_dir, labels_dict):
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.file_dict = {'srch': {}, 'diff': {}, 'temp': {}}
        traverse_directory(root_dir, self.file_dict)

        self.file_ids = sorted(set(self.file_dict['srch'].keys()) &
                               set(self.file_dict['diff'].keys()) &
                               set(self.file_dict['temp'].keys()))

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        srch_file = self.file_dict['srch'][file_id]
        diff_file = self.file_dict['diff'][file_id]
        temp_file = self.file_dict['temp'][file_id]

        srch_image = fits.getdata(srch_file).astype(np.float32)
        diff_image = fits.getdata(diff_file).astype(np.float32)
        temp_image = fits.getdata(temp_file).astype(np.float32)

        stacked_image = np.stack((temp_image, srch_image, diff_image), axis=0)
        stacked_tensor = torch.from_numpy(stacked_image)

        target_label = torch.from_numpy(np.array(np.float32(self.labels_dict[file_id])))
        
        return stacked_tensor, target_label