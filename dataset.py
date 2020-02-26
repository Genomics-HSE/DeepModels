import os
import torch
from torch.utils import data


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path, list_ids):
        'Initialization'
        self.path_x = os.path.join(path, "x")
        self.path_y = os.path.join(path, "y")
        self.list_ids = list_ids
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ids)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_ids[index]
        
        # Load data and get label
        X = torch.load(os.path.join(self.path_x, ID + '.pt'))
        y = torch.load(os.path.join(self.path_y, ID + '.pt'))
        
        return X, y