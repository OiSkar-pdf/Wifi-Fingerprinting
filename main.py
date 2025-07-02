import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_test import CustomImageDataset
from classifier import classifier

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_iq_data(data_path, meta_path, dtype=np.complex128):
    try:
        meta = pd.read_json(meta_path)
        iq = np.fromfile(data_path, dtype=dtype)
        return iq
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return None

def load_from_folder(folder_dir):
    file_label_pairs = []
    label_map = {}
    label_counter = 0

    for filename in os.listdir(folder_dir):
        if filename.endswith(".sigmf-data"):
            data_path = os.path.join(folder_dir, filename)
            meta_path = os.path.join(folder_dir, filename.replace(".sigmf-data", ".sigmf-meta"))
            if not os.path.exists(meta_path):
                continue

            base = filename.replace(".sigmf-data", "")
            parts = base.split('_')
            label = parts[2] + "_" + parts[3] if len(parts) >= 4 else base

            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1

            file_label_pairs.append((data_path, label_map[label]))

    return file_label_pairs, label_map

def train_model():
    batch_size = 8
    window_size = 128
    stride = 16
    epochs = 4
    learning_rate = 0.001
    num_classes = 16

    batch_dir = "/common/users/oi66/Wifi-Fingerprinting/KRI-16Devices-RawData"
    if not os.path.exists(batch_dir):
        print(f"Error: Directory '{batch_dir}' not found")
        return

    all_file_label_pairs = []

    for folder_name in os.listdir(batch_dir):
        folder = os.path.join(batch_dir, folder_name)
        if not os.path.isdir(folder):
            continue
        file_label_pairs, _ = load_from_folder(folder)
        all_file_label_pairs.extend(file_label_pairs)

    all_file_label_pairs = np.array(all_file_label_pairs, dtype=object)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0

    for train_index, val_index in kf.split(all_file_label_pairs):
        print(f"\n===== Fold {fold + 1} / 5 =====")

        train_pairs = all_file_label_pairs[train_index]
        val_pairs = all_file_label_pairs[val_index]

        train_dataset = CustomImageDataset(train_pairs, None, window_size=window_size, stride=stride)
        val_dataset = CustomImageDataset(val_pairs, None, window_size=window_size, stride=stride)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = classifier(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == Y_batch).sum().item()
                total += Y_batch.size(0)

            avg_loss = epoch_loss / len(train_loader)
            train_acc = 100 * correct / total
            print(f"  Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == Y_batch).sum().item()
                total += Y_batch.size(0)

        val_acc = 100 * correct / total
        print(f"Fold {fold+1} Validation Accuracy: {val_acc:.2f}%")

        torch.save(model.state_dict(), f"model_fold_{fold+1}.pt")
        fold += 1

if __name__ == '__main__':
    train_model()
