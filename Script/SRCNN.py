import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(output)
        output = self.conv2(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.conv3(output)
        output += x  
        return output