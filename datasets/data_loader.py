"""
Multimodal Data Loaders for Pathomic Fusion.

Implements PyTorch Dataset and DataLoader classes for:
- Histology images (PIL Image → transforms)
- Cell spatial graphs (PyG Data .pt objects)
- Genomic profiles (CNV, mutation, RNA-Seq vectors)
- Survival targets (censored status, survival months)
- Grade targets (WHO histologic grade)

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.helpers import mixed_collate


class PathgraphomicDatasetLoader(Dataset):
    """
    Standard multimodal dataset that reads raw images from disk.
    
    Returns tuple:
        (x_path, x_grph, x_omic, censored, survival_months, grade)
    """

    def __init__(self, opt, data, split, mode='omic'):
        self.opt = opt
        self.mode = mode
        split_data = data[split]

        self.X_path = split_data['x_path']
        self.X_grph = split_data['x_grph']
        self.X_omic = split_data['x_omic']
        self.e = split_data['e']
        self.t = split_data['t']
        self.g = split_data['g']

        # Histopathology image augmentation
        crop_size = getattr(opt, 'input_size_path', 512)
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(crop_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.e)

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index], dtype=torch.float32)
        single_t = torch.tensor(self.t[index], dtype=torch.float32)
        single_g = torch.tensor(self.g[index], dtype=torch.long)

        # 1. Path Image
        single_X_path = 0
        if self.mode in ('path', 'pathpath', 'pathomic', 'pathgraph', 'pathgraphomic'):
            if isinstance(self.X_path[index], str) and os.path.exists(self.X_path[index]):
                img = Image.open(self.X_path[index]).convert('RGB')
                single_X_path = self.transforms(img)
            elif isinstance(self.X_path[index], torch.Tensor):
                single_X_path = self.X_path[index]
            elif isinstance(self.X_path[index], np.ndarray):
                single_X_path = torch.from_numpy(self.X_path[index]).float()

        # 2. Graph
        single_X_grph = 0
        if self.mode in ('graph', 'graphgraph', 'graphomic', 'pathgraph', 'pathgraphomic'):
            if isinstance(self.X_grph[index], str) and os.path.exists(self.X_grph[index]):
                single_X_grph = torch.load(self.X_grph[index], weights_only=False)
            elif hasattr(self.X_grph[index], 'x'):
                single_X_grph = self.X_grph[index]

        # 3. Omic Vector
        single_X_omic = 0
        if self.mode in ('omic', 'omicomic', 'pathomic', 'graphomic', 'pathgraphomic'):
            single_X_omic = torch.tensor(self.X_omic[index], dtype=torch.float32)

        return (single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)


class PathgraphomicFastDatasetLoader(Dataset):
    """
    Fast multimodal dataset loader that uses pre-extracted features or tensors.
    """

    def __init__(self, opt, data, split, mode='omic'):
        self.opt = opt
        self.mode = mode
        split_data = data[split]

        self.X_path = split_data['x_path']
        self.X_grph = split_data['x_grph']
        self.X_omic = split_data['x_omic']
        self.e = split_data['e']
        self.t = split_data['t']
        self.g = split_data['g']

    def __len__(self):
        return len(self.e)

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index], dtype=torch.float32)
        single_t = torch.tensor(self.t[index], dtype=torch.float32)
        single_g = torch.tensor(self.g[index], dtype=torch.long)

        # 1. Path Features
        single_X_path = 0
        if self.mode in ('path', 'pathpath', 'pathomic', 'pathgraph', 'pathgraphomic'):
            if isinstance(self.X_path[index], torch.Tensor):
                single_X_path = self.X_path[index].squeeze(0).float()
            elif isinstance(self.X_path[index], np.ndarray):
                single_X_path = torch.from_numpy(self.X_path[index]).float().squeeze(0)
            elif isinstance(self.X_path[index], str) and os.path.exists(self.X_path[index]):
                img = Image.open(self.X_path[index]).convert('RGB')
                tform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                single_X_path = tform(img)

        # 2. Graph
        single_X_grph = 0
        if self.mode in ('graph', 'graphgraph', 'graphomic', 'pathgraph', 'pathgraphomic'):
            if isinstance(self.X_grph[index], str) and os.path.exists(self.X_grph[index]):
                single_X_grph = torch.load(self.X_grph[index], weights_only=False)
            elif hasattr(self.X_grph[index], 'x'):
                single_X_grph = self.X_grph[index]

        # 3. Omic
        single_X_omic = 0
        if self.mode in ('omic', 'omicomic', 'pathomic', 'graphomic', 'pathgraphomic'):
            single_X_omic = torch.tensor(self.X_omic[index], dtype=torch.float32)

        return (single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)


def build_dataloader(opt, data, split='train', mode='omic', batch_size=32, shuffle=True):
    """Factory function to build DataLoader with mixed collation."""
    use_fast = getattr(opt, 'use_vgg_features', 0) == 1
    loader_cls = PathgraphomicFastDatasetLoader if use_fast else PathgraphomicDatasetLoader

    dataset = loader_cls(opt, data, split, mode=mode)
    has_graph = mode in ('graph', 'graphgraph', 'graphomic', 'pathgraph', 'pathgraphomic')
    collate_fn = mixed_collate if has_graph else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=(split == 'train' and len(dataset) > batch_size)
    )
    return dataloader
