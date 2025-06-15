import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ScanObjectNNDataset(Dataset):
    def __init__(self, root_dir="data/ScanObjNN/h5_files/main_split", split="train", num_points=1024):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        if split == 'train':
            h5_path = os.path.join(root_dir, 'training_objectdataset.h5')
        elif split == 'test':
            h5_path = os.path.join(root_dir, 'test_objectdataset.h5')
        else:
            raise ValueError(f"Invalid split: {split}")

        with h5py.File(h5_path, 'r') as f:
            self.data = f['data'][:]
            self.label = f['label'][:]

        print(f'[ScanObjectNNDataset] Loaded {self.data.shape[0]} samples ({split}) from {root_dir}')

    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        label = self.label[idx]

        choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        pointcloud = torch.from_numpy(pointcloud).float()
        label = torch.tensor(label).long()

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
