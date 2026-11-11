"""
Multimodal Data Loaders for Pathomic Fusion.

Two loading paths:
- `TensorBatchLoader`: fast in-memory loader used whenever every modality of
  the chosen mode is already a numeric array (genomic vectors, pre-extracted
  histology embeddings). The whole split lives on the training device, so an
  epoch has no per-sample Python overhead.
- `PathgraphomicDatasetLoader`: standard PyTorch Dataset used when raw ROI
  images or cell-graph files (PyG `.pt`) have to be read from disk.

Both yield the tuple
    (x_path, x_grph, x_omic, censored, survival_months, grade)
with `0` in place of a modality that the mode does not use.

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.helpers import mixed_collate

PATH_MODES = ('path', 'pathpath', 'pathomic', 'pathgraph', 'pathgraphomic')
GRAPH_MODES = ('graph', 'graphgraph', 'graphomic', 'pathgraph', 'pathgraphomic')
OMIC_MODES = ('omic', 'omicomic', 'pathomic', 'graphomic', 'pathgraphomic')


# ── Index helpers ─────────────────────────────────────────────────────

def valid_indices(split_data, task='surv'):
    """Rows usable for the task (grade task drops rows with missing grade)."""
    n = len(split_data['e'])
    if task == 'grad':
        return np.where(np.asarray(split_data['g']) >= 0)[0]
    return np.arange(n)


def patient_ids(names):
    """
    Patient identifiers of the rows of a split.

    `x_patname` holds either the patient barcode or a longer ROI / slide
    name; a TCGA patient barcode is its first 12 characters (TCGA-XX-XXXX).
    """
    return np.array([str(n)[:12] for n in names])


def first_row_per_patient(names, indices=None):
    """Indices of the first row of every patient (within `indices`)."""
    names = patient_ids(names)
    if indices is None:
        indices = np.arange(len(names))
    _, first = np.unique(names[indices], return_index=True)
    return indices[np.sort(first)]


def patient_val_split(names, indices, val_frac, seed):
    """
    Split `indices` into train / validation parts at the patient level, so
    that all rows of a patient fall on the same side.

    Returns:
        (train_indices, val_indices)
    """
    names = patient_ids(names)
    patients = np.unique(names[indices])
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_frac)))
    val_patients = set(patients[:n_val])
    is_val = np.array([n in val_patients for n in names[indices]])
    return indices[~is_val], indices[is_val]


def _is_numeric_array(x):
    return isinstance(x, np.ndarray) and x.dtype.kind in 'fiu'


def path_feature_stats(data):
    """
    Mean / std of the pre-extracted histology embeddings, computed on the
    training rows of the fold only (None when x_path holds image paths).

    The stored embeddings are unscaled post-ReLU activations; standardising
    them keeps the output sigmoid and the fusion gates out of saturation.
    """
    x = data['train']['x_path']
    if not _is_numeric_array(x) or len(x) == 0:
        return None
    x = x.reshape(len(x), -1).astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0  # constant (dead) features map to zero
    return mean, std


def _resolve_path(path, dataroot):
    """Resolve a file path stored in a split file against the data root."""
    if os.path.exists(path):
        return path
    alt = os.path.join(dataroot, os.path.basename(os.path.dirname(path)),
                       os.path.basename(path))
    return alt


# ── Cell graphs ───────────────────────────────────────────────────────

_GRAPH_CACHE = {}


def load_graph(path, cache=True, device=None):
    """
    Load a cell graph (PyG `Data`) from disk.

    The graph files were written by an early PyTorch Geometric release, so
    the object is rebuilt as a current `Data` holding x / edge_index /
    edge_attr. Graphs are cached: every fold and epoch re-uses the same
    ~1500 ROI graphs. With `device` set, the cached graphs live on that
    device, so batches are assembled there without any host-to-GPU copy.
    """
    key = (path, str(device))
    if cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    from torch_geometric.data import Data
    raw = torch.load(path, weights_only=False, map_location='cpu')
    fields = raw.__dict__.get('_store', None)
    get = (lambda k: fields.get(k)) if fields is not None else (
        lambda k: raw.__dict__.get(k))
    x, edge_index, edge_attr = get('x'), get('edge_index'), get('edge_attr')
    graph = Data(
        x=x.float(), edge_index=edge_index.long(),
        edge_attr=edge_attr.float() if edge_attr is not None else None)
    if device is not None:
        graph = graph.to(device)
    if cache:
        _GRAPH_CACHE[key] = graph
    return graph


# ── Fast in-memory loader ─────────────────────────────────────────────

class TensorBatchLoader(object):
    """
    Mini-batch iterator over tensors that are held on `device`.

    Args:
        x_path: (N, path_dim) tensor or None
        x_omic: (N, n_omic) tensor or None
        e, t, g: (N,) event indicator, survival time, grade
        batch_size: mini-batch size (<= 0 means full batch)
        shuffle: reshuffle every epoch
        drop_last: drop the final incomplete batch
        seed: seed of the shuffling generator
    """

    def __init__(self, x_path, x_omic, e, t, g, batch_size=32,
                 shuffle=False, drop_last=False, seed=0):
        self.x_path, self.x_omic = x_path, x_omic
        self.e, self.t, self.g = e, t, g
        self.n = len(e)
        self.batch_size = self.n if batch_size <= 0 else min(batch_size, self.n)
        self.shuffle = shuffle
        self.drop_last = drop_last and self.n > self.batch_size
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, generator=self.generator)
        else:
            order = torch.arange(self.n)
        order = order.to(self.e.device)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            yield (
                self.x_path[idx] if self.x_path is not None else 0,
                0,
                self.x_omic[idx] if self.x_omic is not None else 0,
                self.e[idx], self.t[idx], self.g[idx],
            )


# ── Disk-backed dataset (raw images / graphs) ─────────────────────────

class PathgraphomicDatasetLoader(Dataset):
    """
    Multimodal dataset reading ROI images and/or cell graphs from disk.

    Random augmentation is applied to training images only; evaluation
    images are centre-cropped.
    """

    def __init__(self, opt, data, split, mode='omic', indices=None,
                 augment=None, device=None):
        self.opt = opt
        self.mode = mode
        # cached graphs are kept on the training device unless worker
        # processes are used (CUDA tensors cannot be shared with them)
        self.graph_device = (
            device if getattr(opt, 'cache_graphs', 1)
            and int(getattr(opt, 'num_workers', 0)) == 0 else None)
        self.dataroot = getattr(opt, 'dataroot', './data/TCGA_GBMLGG')
        split_data = data[split]
        self.indices = (np.arange(len(split_data['e']))
                        if indices is None else np.asarray(indices))

        self.X_path = split_data['x_path']
        self.X_grph = split_data['x_grph']
        self.X_omic = split_data['x_omic']
        self.e = split_data['e']
        self.t = split_data['t']
        self.g = split_data['g']

        self.path_stats = (path_feature_stats(data)
                           if getattr(opt, 'standardize_path', 1) else None)

        augment = (split == 'train') if augment is None else augment
        crop_size = getattr(opt, 'input_size_path', 512)
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if augment:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomCrop(crop_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                       saturation=0.05, hue=0.01),
            ] + normalize)
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(crop_size)] + normalize
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        single_e = torch.tensor(self.e[index], dtype=torch.float32)
        single_t = torch.tensor(self.t[index], dtype=torch.float32)
        single_g = torch.tensor(self.g[index], dtype=torch.long)

        # 1. Histology: raw image or pre-extracted embedding
        single_X_path = 0
        if self.mode in PATH_MODES:
            item = self.X_path[index]
            if isinstance(item, torch.Tensor):
                item = item.numpy()
            if isinstance(item, np.ndarray) and item.dtype.kind in 'fiu':
                feat = item.reshape(-1).astype(np.float32)
                if self.path_stats is not None:
                    feat = (feat - self.path_stats[0]) / self.path_stats[1]
                single_X_path = torch.from_numpy(feat)
            else:
                img_path = _resolve_path(str(item), self.dataroot)
                img = Image.open(img_path).convert('RGB')
                single_X_path = self.transforms(img)

        # 2. Cell graph
        single_X_grph = 0
        if self.mode in GRAPH_MODES:
            item = self.X_grph[index]
            if hasattr(item, 'x'):
                single_X_grph = item
            else:
                grph_path = _resolve_path(str(item), self.dataroot)
                single_X_grph = load_graph(
                    grph_path, cache=bool(getattr(self.opt, 'cache_graphs', 1)),
                    device=self.graph_device)

        # 3. Genomic vector
        single_X_omic = 0
        if self.mode in OMIC_MODES:
            single_X_omic = torch.tensor(self.X_omic[index],
                                         dtype=torch.float32)

        return (single_X_path, single_X_grph, single_X_omic,
                single_e, single_t, single_g)


class PathgraphomicFastDatasetLoader(PathgraphomicDatasetLoader):
    """Kept for backward compatibility (pre-extracted features are handled
    by `PathgraphomicDatasetLoader` / `TensorBatchLoader`)."""


# ── Factory ───────────────────────────────────────────────────────────

def can_use_tensor_loader(data, split, mode):
    """True when every modality of `mode` is an in-memory numeric array."""
    if mode in GRAPH_MODES:
        return False
    if mode in PATH_MODES and not _is_numeric_array(data[split]['x_path']):
        return False
    return True


def build_dataloader(opt, data, split='train', mode='omic', batch_size=32,
                     shuffle=True, device=None, indices=None, seed=0):
    """
    Build a loader for one split of one CV fold.

    Args:
        data: fold dictionary with 'train' / 'test' entries
        split: 'train' or 'test'
        indices: optional row subset (validation hold-out, de-duplication)
        device: device for the in-memory loader (default: CPU)
    """
    split_data = data[split]
    if indices is None:
        indices = np.arange(len(split_data['e']))
    indices = np.asarray(indices)
    is_train = (split == 'train') and shuffle

    if can_use_tensor_loader(data, split, mode):
        device = device or torch.device('cpu')
        x_path = x_omic = None
        if mode in PATH_MODES:
            arr = np.asarray(split_data['x_path'])[indices]
            arr = arr.reshape(len(indices), -1).astype(np.float32)
            if getattr(opt, 'standardize_path', 1):
                mean, std = path_feature_stats(data)
                arr = (arr - mean) / std
            x_path = torch.from_numpy(arr).to(device)
        if mode in OMIC_MODES:
            arr = np.asarray(split_data['x_omic'])[indices]
            x_omic = torch.from_numpy(arr).float().to(device)
        e = torch.as_tensor(np.asarray(split_data['e'])[indices],
                            dtype=torch.float32, device=device)
        t = torch.as_tensor(np.asarray(split_data['t'])[indices],
                            dtype=torch.float32, device=device)
        g = torch.as_tensor(np.asarray(split_data['g'])[indices],
                            dtype=torch.long, device=device)
        return TensorBatchLoader(x_path, x_omic, e, t, g,
                                 batch_size=batch_size, shuffle=shuffle,
                                 drop_last=is_train, seed=seed)

    dataset = PathgraphomicDatasetLoader(opt, data, split, mode=mode,
                                         indices=indices, augment=is_train,
                                         device=device)
    if batch_size <= 0:
        batch_size = len(dataset)
    num_workers = int(getattr(opt, 'num_workers', 0))
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mixed_collate if mode in GRAPH_MODES else None,
        drop_last=(is_train and len(dataset) > batch_size),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
