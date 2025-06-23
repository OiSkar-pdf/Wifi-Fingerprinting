import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sigmf import SigMFFile, sigmffile
import numpy as np

def sliding_window(data, window_size, stride):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        segment = data[i:i + window_size]
        real = segment.real
        imag = segment.imag
        windows.append(np.stack([real, imag], axis=0))
    return np.array(windows, dtype=np.float32)


class CustomImageDataset(Dataset):
    def __init__(self, iq_sequences, labels, dist_dir, window_size=128, stride=1):
        self.iq_sequences = iq_sequences
        self.labels = labels
        self.dist_dir = dist_dir
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self.targets = []

        for seq, label in zip(self.iq_sequences, self.labels):
            window = sliding_window(seq, self.window_size, self.stride)
            self.samples.extend(window)
            self.targets.extend([label] * len(window))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)