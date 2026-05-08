import os
import numpy as np
import torch
from torch.utils.data import Dataset


# DATASET
class NPZDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npz')])

        for f in files:
            data = np.load(os.path.join(folder, f))
            A = data['A']
            B = data['B']
            L = data['L']

            for i in range(A.shape[0]):
                self.samples.append((A[i],B[i], L[i]))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        A, B, L = self.samples[idx]
    
        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0)  #original shape is [101,161] ; after unsqueeze [1,101,161] - CNN expcts channel dim
    
        return A, B, L