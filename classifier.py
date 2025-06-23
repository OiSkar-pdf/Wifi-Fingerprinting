import torch
import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 50, 7, 1, 3)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(2, 50, 7, 1, 3)
        self.pool2 = nn.MaxPool1d(2, 2)


        self.fc1 = nn.Linear(50 * 32, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 80)
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(80, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x