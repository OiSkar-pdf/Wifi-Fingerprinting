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
from data_test import CustomImageDataset
from classifier import classifier

def load_iq_data(data_path, meta_path, dtype=np.complex128):
    meta = pd.read_json(meta_path)
    num_samples = meta['captures'][0]['core:sample_count']
    iq = np.fromfile(data_path, dtype=dtype)

    return iq if num_samples == None else iq[:num_samples]

def load_from_folder(folder_dir):
    signals = []
    labels = []
    label_map = {}
    label_counter = 0

    for filename in os.listdir(folder_dir):
        if filename.endswith(".sigmf-data"):
            data_path = os.path.join(folder_dir, filename)
            meta_path = os.path.join(folder_dir, filename.replace(".sigmf-data", ".sigmf-meta"))
            iq = load_iq_data(data_path, meta_path)
            signals.append(iq)
            meta = pd.read_json(meta_path)
            label = meta['annotations'][0]['genesys:transmitter']['model']
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1
            labels.append(label_map[label])
    
    return np.array(signals), np.array(labels), label_map

if __name__ == '__main__':
    batch_size = 3
    window_size = 128
    stride = 16
    epochs = 3
    learning_rate = 0.001
    num_classes = 16
    all_folders = os.listdir("batch")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, 3):
        folder = os.path.join("batch", all_folders[i])
        signals, labels, label_map = load_from_folder(folder)

        dataset = CustomImageDataset(signals, labels, folder, window_size=window_size, stride=stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if model is None:
            model = classifier(num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for X_batch, Y_batch in dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
    
    torch.save(model.state_dict(), "model.pt")