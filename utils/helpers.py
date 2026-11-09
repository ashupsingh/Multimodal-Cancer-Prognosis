"""
Helper utilities for Pathomic Fusion.
- Freeze/unfreeze layers
- Custom collate for mixed data (graph + tensor)
- Data preprocessing utilities
"""

import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate

try:
    import torch_geometric
    from torch_geometric.data import Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False


# ── Freeze / Unfreeze ──────────────────────────────────────────────────

def dfs_freeze(model):
    """Recursively freeze all parameters in a model."""
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    """Recursively unfreeze all parameters in a model."""
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def unfreeze_unimodal(opt, model, epoch):
    """
    Unfreeze pre-trained unimodal sub-networks at a specified epoch.
    Following the paper: unfreeze at epoch 5 during fusion training.
    """
    m = model.module if hasattr(model, 'module') else model

    if epoch == 5:
        if opt.mode in ('graphomic', 'pathgraphomic'):
            if hasattr(m, 'omic_net'):
                dfs_unfreeze(m.omic_net)
                print("Unfreezing Omic Network")
            if hasattr(m, 'grph_net'):
                dfs_unfreeze(m.grph_net)
                print("Unfreezing Graph Network")
        elif opt.mode == 'pathomic':
            if hasattr(m, 'omic_net'):
                dfs_unfreeze(m.omic_net)
                print("Unfreezing Omic Network")
        elif opt.mode == 'omicomic':
            if hasattr(m, 'omic_net'):
                dfs_unfreeze(m.omic_net)
                print("Unfreezing Omic Network")
        elif opt.mode == 'graphgraph':
            if hasattr(m, 'grph_net'):
                dfs_unfreeze(m.grph_net)
                print("Unfreezing Graph Network")


# ── Collate ────────────────────────────────────────────────────────────

def mixed_collate(batch):
    """
    Custom collate function that handles mixed data types:
    - PyG Data objects → Batch.from_data_list
    - Standard tensors → default_collate
    """
    elem = batch[0]
    transposed = zip(*batch)

    result = []
    for samples in transposed:
        samples = list(samples)
        if HAS_PYGEOMETRIC and isinstance(samples[0], torch_geometric.data.Data):
            result.append(Batch.from_data_list(samples, []))
        else:
            result.append(default_collate(samples))
    return result


# ── Sorting ────────────────────────────────────────────────────────────

def natural_sort(l):
    """Sort strings with embedded numbers in natural order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# ── Data Preprocessing ────────────────────────────────────────────────

def addHistomolecularSubtype(data):
    """
    Add histomolecular subtype column based on molecular and histological subtypes.
    
    Molecular Subtype: IDHwt=0, IDHmut-non-codel=1, IDHmut-codel=2
    Histology Subtype: astrocytoma=0, oligoastrocytoma=1, oligodendroglioma=2, glioblastoma=3
    """
    subtyped_data = data.copy()
    subtyped_data.insert(
        loc=0, column='Histomolecular subtype', value=np.ones(len(data))
    )

    idhwt_ATC = np.logical_and(
        data['Molecular subtype'] == 0,
        np.logical_or(data['Histology'] == 0, data['Histology'] == 3)
    )
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'

    idhmut_ATC = np.logical_and(
        data['Molecular subtype'] == 1,
        np.logical_or(data['Histology'] == 0, data['Histology'] == 3)
    )
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'

    ODG = np.logical_and(
        data['Molecular subtype'] == 2, data['Histology'] == 2
    )
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'

    return subtyped_data


def changeHistomolecularSubtype(data):
    """Recompute histomolecular subtype (drop and re-add)."""
    data = data.drop(['Histomolecular subtype'], axis=1)
    return addHistomolecularSubtype(data)


def getCleanAllDataset(dataroot='./data/TCGA_GBMLGG/',
                       ignore_missing_moltype=False,
                       ignore_missing_histype=False,
                       use_rnaseq=False):
    """
    Load and preprocess the TCGA-GBMLGG dataset.
    
    Steps:
    1. Join clinical + genomic data
    2. Handle missing molecular subtypes
    3. Handle missing histological data
    4. Add histomolecular subtype labels
    5. Optionally append RNA-Seq features
    
    Returns:
        metadata: list of metadata column names
        all_dataset: preprocessed DataFrame
    """
    metadata = [
        'Histology', 'Grade', 'Molecular subtype',
        'TCGA ID', 'censored', 'Survival months'
    ]

    # Load clinical + genomic data
    all_dataset_path = os.path.join(dataroot, 'all_dataset.csv')
    if not os.path.exists(all_dataset_path):
        # Try alternate filename
        all_dataset_path = os.path.join(dataroot, 'all_datasets.csv')

    all_dataset = pd.read_csv(all_dataset_path)
    if 'indexes' in all_dataset.columns:
        all_dataset = all_dataset.drop('indexes', axis=1)
    all_dataset.index = all_dataset['TCGA ID']

    # Load grade data
    grade_path = os.path.join(dataroot, 'grade_data.csv')
    if os.path.exists(grade_path):
        all_grade = pd.read_csv(grade_path)
        all_grade['Histology'] = all_grade['Histology'].str.replace(
            'astrocytoma (glioblastoma)', 'glioblastoma', regex=False
        )
        all_grade.index = all_grade['TCGA ID']
        all_dataset = all_dataset.join(
            all_grade[['Histology', 'Grade', 'Molecular subtype']],
            how='inner', rsuffix='_grade'
        )
        # Keep joined columns, drop duplicates
        for col in ['Histology', 'Grade', 'Molecular subtype']:
            if f'{col}_grade' in all_dataset.columns:
                all_dataset[col] = all_dataset[f'{col}_grade']
                all_dataset = all_dataset.drop(f'{col}_grade', axis=1)

    # Reorder columns
    cols = all_dataset.columns.tolist()
    meta_cols = [c for c in metadata if c in cols]
    other_cols = [c for c in cols if c not in meta_cols]
    all_dataset = all_dataset[meta_cols + other_cols]

    # Append RNA-Seq data
    if use_rnaseq:
        gbm_path = os.path.join(
            dataroot, 'mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt'
        )
        lgg_path = os.path.join(
            dataroot, 'mRNA_Expression_Zscores_RSEM.txt'
        )

        if os.path.exists(gbm_path) and os.path.exists(lgg_path):
            gbm = pd.read_csv(gbm_path, sep='\t', skiprows=1, index_col=0)
            lgg = pd.read_csv(lgg_path, sep='\t', skiprows=1, index_col=0)
            gbm = gbm[gbm.columns[~gbm.isnull().all()]]
            lgg = lgg[lgg.columns[~lgg.isnull().all()]]
            glioma_RNAseq = gbm.join(lgg, how='inner').T
            glioma_RNAseq = glioma_RNAseq.dropna(axis=1)
            glioma_RNAseq.columns = [
                gene + '_rnaseq' for gene in glioma_RNAseq.columns
            ]
            glioma_RNAseq.index = [
                patname[:12] for patname in glioma_RNAseq.index
            ]
            glioma_RNAseq = glioma_RNAseq.iloc[
                ~glioma_RNAseq.index.duplicated()
            ]
            glioma_RNAseq.index.name = 'TCGA ID'
            all_dataset = all_dataset.join(glioma_RNAseq, how='inner')

    # Handle missing data
    if 'Molecular subtype' in all_dataset.columns:
        pat_missing_moltype = all_dataset[
            all_dataset['Molecular subtype'].isna()
        ].index
        print(f"# Missing Molecular Subtype: {len(pat_missing_moltype)}")

        if ignore_missing_moltype:
            all_dataset = all_dataset[
                all_dataset['Molecular subtype'].isna() == False
            ]

    # Impute missing genomic data with median
    for col in all_dataset.columns:
        if col not in metadata:
            if 'Molecular subtype' in all_dataset.columns:
                all_dataset['Molecular subtype'] = all_dataset[
                    'Molecular subtype'
                ].fillna('Missing')
            all_dataset[col] = all_dataset[col].fillna(
                all_dataset[col].median() if all_dataset[col].dtype != object 
                else 'Missing'
            )

    # Handle missing histological data
    if 'Histology' in all_dataset.columns:
        if ignore_missing_histype:
            all_dataset = all_dataset[
                all_dataset['Histology'].isna() == False
            ]
        else:
            if 'Grade' in all_dataset.columns:
                all_dataset['Grade'] = all_dataset['Grade'].fillna(1)
            all_dataset['Histology'] = all_dataset['Histology'].fillna(
                'Missing'
            )

    # Convert to numeric codes
    if 'Grade' in all_dataset.columns:
        all_dataset['Grade'] = all_dataset['Grade'] - 2

    if 'Molecular subtype' in all_dataset.columns:
        ms2int = {
            'Missing': -1, 'IDHwt': 0,
            'IDHmut-non-codel': 1, 'IDHmut-codel': 2
        }
        all_dataset['Molecular subtype'] = all_dataset[
            'Molecular subtype'
        ].map(lambda s: ms2int.get(s, s))

    if 'Histology' in all_dataset.columns:
        hs2int = {
            'Missing': -1, 'astrocytoma': 0,
            'oligoastrocytoma': 1, 'oligodendroglioma': 2,
            'glioblastoma': 3
        }
        all_dataset['Histology'] = all_dataset['Histology'].map(
            lambda s: hs2int.get(s, s)
        )

    # Add histomolecular subtype
    all_dataset = addHistomolecularSubtype(all_dataset)
    metadata.extend(['Histomolecular subtype'])

    # Invert censoring (1=event occurred)
    if 'censored' in all_dataset.columns:
        all_dataset['censored'] = 1 - all_dataset['censored']

    return metadata, all_dataset
