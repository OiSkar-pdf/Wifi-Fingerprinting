import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pandas
from torch.utils.data import Dataset, DataLoader
from sigmf import SigMFFile, sigmffile
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, folder_dir, segment_len=128):
        self.folder_dir = folder_dir
        self.transform = ToTensor()
        self.meta_files = [f for f in os.listdir(folder_dir) if f.endswith('.sigmf-meta')]

    def __len__(self):
        return len(self.meta_files)

    def __getitem__(self, idx):
        meta_file = self.meta_files[idx]
        meta_path = os.path.join(self.folder_dir, meta_file)
        data_path = meta_path.replace('.sigmf-meta', '.sigmf-data')
        meta_data = pandas.read_json(meta_path)

        d_type = meta_data['global']['core:datatype']
        if d_type == 'cf64':
            dtype = np.complex64
        
        samples = np.fromfile(data_path, dtype=dtype)

        return samples


data = CustomImageDataset(f'C:/Users/acer/Documents/GitHub/CNN/KRI-16Devices-RawData/Batch/2ft')

for i in range(0, 10):
    sample = data[i]
    print(sample.shape)