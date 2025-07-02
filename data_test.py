import os
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, window_size=128, stride=16):
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        for path, label in file_paths:
            file_size = os.path.getsize(path) // 8  # complex64 = 8 bytes
            n_windows = (file_size - window_size) // stride + 1
            for i in range(n_windows):
                start = i * stride
                self.samples.append((path, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, label = self.samples[idx]
        iq = np.memmap(path, dtype=np.complex64, mode='r', offset=start * 8, shape=(self.window_size,))
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            iq = iq / np.sqrt(power)
        x = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)  # [2, window_size]
        x = np.expand_dims(x, axis=0)  # [1, 2, window_size] if needed by your model
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)