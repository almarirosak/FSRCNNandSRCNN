import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

class VolumeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.volume_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.nii', '.nii.gz'))]

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        volume_path = self.volume_files[idx]
        volume = nib.load(volume_path).get_fdata()
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        high_res_volume = F.interpolate(volume.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False).squeeze(0)
        low_res_volume = F.interpolate(high_res_volume.unsqueeze(0), size=(80, 80, 80), mode='trilinear', align_corners=False).squeeze(0)

        if self.transform:
            low_res_volume = self.transform(low_res_volume)

        return low_res_volume, high_res_volume
