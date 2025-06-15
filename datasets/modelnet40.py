import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir="data/ModeNet40/modelnet40_ply_hdf5_2048", split="train", num_points=1024):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        self.data, self.label = self.load_data()

        print(f'[ModelNet40Dataset] Loaded {self.data.shape[0]} samples ({split})')

    def load_data(self):
        all_data = []
        all_label = []
        for file in glob.glob(os.path.join(self.root_dir, f'*{self.split}*.h5')):
            f = h5py.File(file, 'r')
            data = f['data'][:]
            label = f['label'][:]
            f.close()
            all_data.append(data)
            all_label.append(label)
        return np.concatenate(all_data, axis=0), np.concatenate(all_label, axis=0)

    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        label = self.label[idx][0]

        choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        pointcloud = torch.from_numpy(pointcloud).float()
        label = torch.tensor(label).long()

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
