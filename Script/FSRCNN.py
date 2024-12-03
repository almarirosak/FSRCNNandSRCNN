import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels=1, out_channels=56, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv3d(in_channels=56, out_channels=12, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv_7 = nn.Conv3d(in_channels=12, out_channels=56, kernel_size=1, padding=0)
        self.de_conv_1 = nn.ConvTranspose3d(in_channels=56, out_channels=1, kernel_size=9, stride=3, padding=3, output_padding=0)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        output = self.conv_1(x)
        output = F.relu(output)
        output = self.conv_2(output)
        output = F.relu(output)
        output = self.conv_3(output)
        output = F.relu(output)
        output = self.conv_4(output)
        output = F.relu(output)
        output = self.conv_5(output)
        output = F.relu(output)
        output = self.conv_6(output)
        output = F.relu(output)
        output = self.conv_7(output)
        output = self.dropout(output)
        output = self.de_conv_1(output)
        output = F.relu(output)
        return output