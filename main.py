import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from data_test import CustomImageDataset
from classifier import classifier
from torch.utils.data import Subset
from collections import defaultdict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

batch_dir = "/common/users/oi66/Wifi-Fingerprinting/KRI_16Devices_RawData"
labels = []
 
for folder in os.listdir(batch_dir):
    labels.append(folder)

label_map = {label: idx for idx, label in enumerate(labels)}
#train 4 devices at a time
epochs = 20
batch_size = 1024
window_size = 128
stride = 64
optimizer = torch.optim.Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = []
train_labels = labels[:4]
#create a dataset with the first 4 devices
for label in train_labels:
    folder_path = os.path.join(batch_dir, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.sigmf-data'):
            label_idx = label_map[label]
                # Assuming the run is part of the filename
            file_path = os.path.join(folder_path, file_name)
            data.append((file_path, label_idx))


# Create the full windowed dataset for your selected devices
full_dataset = CustomImageDataset(data, window_size=window_size, stride=stride)

# Group indices by device label
label_to_indices = defaultdict(list)
for idx, (_, _, label) in enumerate(full_dataset.samples):
    label_to_indices[label].append(idx)


random.seed(42)

train_size = 200_000
val_size = 10_000
test_size = 50_000

train_indices = []
val_indices = []
test_indices = []

for label, indices in label_to_indices.items():
    indices = indices.copy()
    random.shuffle(indices)

    train_split = indices[:train_size]
    val_split = indices[train_size:train_size + val_size]
    test_split = indices[train_size + val_size:train_size + val_size + test_size]

    train_indices.extend(train_split)
    val_indices.extend(val_split)
    test_indices.extend(test_split)

train_dataset = Subset(full_dataset, train_indices)  # Print the first 50 samples to check the data
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)


dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = classifier(num_classes=len(train_labels))
model.to(device)
optimizer = optimizer(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, optimizer, loss
model = classifier(num_classes=len(train_labels))
model.to(device)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # --- Training ---
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Optional: classification report per epoch
    print("Validation classification report:")
    print(classification_report(all_val_labels, all_val_preds, target_names=train_labels))

# --- Testing after training is complete ---
model.eval()
all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

print("Test classification report:")
print(classification_report(all_test_labels, all_test_preds, target_names=train_labels))

# Save the trained model
torch.save(model.state_dict(), "/common/users/oi66/Wifi-Fingerprinting/Fingerprinter(1-4).pth")
