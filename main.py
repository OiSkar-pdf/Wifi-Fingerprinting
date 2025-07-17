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
import seaborn as sns
from sklearn.metrics import confusion_matrix

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Create plots directory
os.makedirs("plots", exist_ok=True)

batch_dir = "/common/users/oi66/Wifi-Fingerprinting/KRI_16Devices_RawData"
labels = []
 
for folder in os.listdir(batch_dir):
    labels.append(folder)

label_map = {label: idx for idx, label in enumerate(labels)}
#train 4 devices at a time
epochs = 60
batch_size = 512  # Increase from 1024
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

train_indices = []
val_indices = []
test_indices = []

# Paper's methodology: 200k train, 50k test, 10k val per class
TRAIN_SAMPLES_PER_CLASS = 200000
TEST_SAMPLES_PER_CLASS = 50000
VAL_SAMPLES_PER_CLASS = 10000

for label, indices in label_to_indices.items():
    indices = indices.copy()
    random.shuffle(indices)
    
    # Calculate required samples
    total_needed = TRAIN_SAMPLES_PER_CLASS + VAL_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS
    
    if len(indices) < total_needed:
        print(f"Warning: Label {label} has only {len(indices)} samples, need {total_needed}")
        # Use proportional scaling if not enough samples
        scale_factor = len(indices) / total_needed
        train_size = int(TRAIN_SAMPLES_PER_CLASS * scale_factor)
        val_size = int(VAL_SAMPLES_PER_CLASS * scale_factor)
        test_size = len(indices) - train_size - val_size
    else:
        train_size = TRAIN_SAMPLES_PER_CLASS
        val_size = VAL_SAMPLES_PER_CLASS
        test_size = TEST_SAMPLES_PER_CLASS
    
    # Split according to paper methodology
    train_split = indices[:train_size]
    val_split = indices[train_size:train_size + val_size]
    test_split = indices[train_size + val_size:train_size + val_size + test_size]
    
    train_indices.extend(train_split)
    val_indices.extend(val_split)
    test_indices.extend(test_split)
    
    print(f"Label {label}: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test")

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

from collections import Counter
# Fix the distribution counting
def get_label_distribution(dataset):
    if hasattr(dataset, 'dataset'):  # It's a Subset
        return Counter([dataset.dataset.samples[i][2] for i in dataset.indices])
    else:  # It's the full dataset
        return Counter([label for _, _, label in dataset.samples])

print("Training label distribution:", get_label_distribution(train_dataset))
print("Validation label distribution:", get_label_distribution(val_dataset))
print("Testing label distribution:", get_label_distribution(test_dataset))

# Initialize tracking variables
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
weight_norms = []

def plot_weight_distributions(model, epoch):
    """Plot weight distributions for all layers"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2', 'out']
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name and i < len(axes):
            weights = param.data.cpu().numpy().flatten()
            axes[i].hist(weights, bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'{name} - Epoch {epoch}')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Density')
            axes[i].grid(True)
    
    # Remove empty subplots
    for j in range(len(layer_names), len(axes)):
        axes[j].remove()
    
    plt.tight_layout()
    plt.savefig(f'plots/weight_distributions_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weight_norms(model, epoch):
    """Track weight norms over training"""
    norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            norms[name] = param.data.norm().item()
    return norms

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, current_epoch):
    # Use the actual length of your data, not a fixed range
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    if val_losses:  # Only plot if we have validation data
        ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    if val_accuracies:  # Only plot if we have validation data
        ax2.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_curves_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, class_names, epoch, phase='val'):
    """Plot confusion matrix"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {phase.capitalize()} - Epoch {epoch}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{phase}_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Initialize model, optimizer, loss
model = classifier(num_classes=len(train_labels))
model.apply(init_weights)
model.to(device)

# Create the optimizer properly
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Decreased LR
criterion = nn.CrossEntropyLoss()

# Add before training
for i, (inputs, labels) in enumerate(train_dataloader):
    print(f"Input shape: {inputs.shape}")
    print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"Labels: {labels.unique()}")
    break

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # --- Training ---
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1

        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    
    # Calculate average training loss
    avg_train_loss = running_loss / num_batches

    # Calculate training accuracy
    train_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}", flush=True)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (preds == labels).sum().item()
            
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = 100 * correct_val / total_val
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%", flush=True)

    # Store metrics
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    # Track weight norms
    epoch_weight_norms = plot_weight_norms(model, epoch + 1)
    weight_norms.append(epoch_weight_norms)
    
    # Generate plots every 5 epochs or on the last epoch
    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        plot_weight_distributions(model, epoch + 1)
        plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1)
        plot_confusion_matrix(all_val_labels, all_val_preds, train_labels, epoch + 1, 'validation')
    
    # Optional: classification report per epoch
    print("Validation classification report:", flush=True)
    print(classification_report(all_val_labels, all_val_preds, target_names=train_labels), flush=True)
    
# Plot weight norm evolution
if weight_norms:
    plt.figure(figsize=(12, 6))
    for layer_name in weight_norms[0].keys():
        norms = [epoch_norms[layer_name] for epoch_norms in weight_norms]
        plt.plot(range(1, len(norms) + 1), norms, label=layer_name, marker='o')
    
    plt.title('Weight Norms Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Norm')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/weight_norms_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

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

print("Test classification report:", flush=True)
print(classification_report(all_test_labels, all_test_preds, target_names=train_labels), flush=True )

# Final plots
plot_confusion_matrix(all_test_labels, all_test_preds, train_labels, epochs, 'test')
plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, epochs)

# Save final metrics
metrics_df = pd.DataFrame({
    'epoch': range(1, epochs + 1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
})
metrics_df.to_csv('plots/training_metrics.csv', index=False)

# Save the trained model
torch.save(model.state_dict(), "/common/users/oi66/Wifi-Fingerprinting/Fingerprinter(1-4).pth")

print("Training complete! Check the 'plots' directory for visualizations.")