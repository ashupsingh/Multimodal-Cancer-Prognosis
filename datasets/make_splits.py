"""
15-Fold Cross-Validation Split Generator for Pathomic Fusion.

Aligns multimodal data (histology ROI image paths / features,
cell spatial graph paths, and genomic vectors) across patients
and splits them into 15 cross-validation folds.

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import copy
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torchvision import transforms
from PIL import Image

from utils.helpers import getCleanAllDataset
from models.networks import define_net


def get_vgg_features(model, device, img_path):
    """Extract intermediate features from a histology ROI using PathNet."""
    if model is None:
        return img_path
    x_path = Image.open(img_path).convert('RGB')
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x_path = torch.unsqueeze(normalize(x_path), dim=0)
    with torch.no_grad():
        features, hazard = model(x_path=x_path.to(device))
    return features.cpu().numpy()


def load_fold_path_model(opt, k, device):
    """Histology CNN trained end-to-end on the training patients of fold k."""
    ckpt = os.path.join(opt.checkpoints_dir, opt.exp_name, 'path',
                        f"path_{k}_s0.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"{ckpt} not found. Train the histology CNN on raw ROIs first "
            "(python train.py --mode path --model_name path "
            "--use_vgg_features 0) before extracting embeddings."
        )
    path_opt = copy.copy(opt)
    path_opt.mode, path_opt.use_vgg_features = 'path', 0
    model = define_net(path_opt)
    model.load_state_dict(torch.load(
        ckpt, map_location=device, weights_only=False)['model_state_dict'])
    return model.eval()


def getAlignedMultimodalData(opt, model, device, all_dataset, metadata, pat_split, pat2img):
    """
    Construct aligned lists of patient names, path inputs, graph paths,
    genomic vectors, survival events, survival times, and grade labels.
    """
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []

    graph_dir = os.path.join(opt.dataroot, f"{opt.roi_dir}_{opt.graph_feat_type}")

    for pat_name in pat_split:
        if pat_name not in all_dataset.index:
            continue

        if pat_name not in pat2img:
            continue

        for img_fname in pat2img[pat_name]:
            grph_fname = os.path.splitext(img_fname)[0] + '.pt'
            grph_path = os.path.join(graph_dir, grph_fname)

            pat_row = all_dataset[all_dataset['TCGA ID'] == pat_name]
            if pat_row.shape[0] == 0:
                continue

            x_patname.append(pat_name)
            img_full_path = os.path.join(opt.dataroot, opt.roi_dir, img_fname)
            x_path.append(get_vgg_features(model, device, img_full_path))
            x_grph.append(grph_path)
            x_omic.append(np.array(pat_row.drop(metadata, axis=1)))
            e.append(int(pat_row['censored'].iloc[0]))
            t.append(float(pat_row['Survival months'].iloc[0]))
            if 'Grade' in pat_row.columns:
                g.append(int(pat_row['Grade'].iloc[0]))
            else:
                g.append(0)

    return x_patname, x_path, x_grph, x_omic, e, t, g


def create_splits(opt):
    """Generate 15-fold CV splits dictionary and save to pickle."""
    os.makedirs(os.path.join(opt.dataroot, 'splits'), exist_ok=True)
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')

    metadata, all_dataset = getCleanAllDataset(
        dataroot=opt.dataroot,
        ignore_missing_moltype=bool(opt.ignore_missing_moltype),
        ignore_missing_histype=bool(opt.ignore_missing_histype),
        use_rnaseq=bool(opt.use_rnaseq)
    )

    roi_path = os.path.join(opt.dataroot, opt.roi_dir)
    if os.path.exists(roi_path):
        img_fnames = os.listdir(roi_path)
        pat2img = {}
        for img_fname in img_fnames:
            pat = img_fname[:12]
            if pat not in pat2img:
                pat2img[pat] = []
            pat2img[pat].append(img_fname)
    else:
        pat2img = {}

    pnas_splits_file = os.path.join(opt.dataroot, 'pnas_splits.csv')
    if os.path.exists(pnas_splits_file):
        pnas_splits = pd.read_csv(pnas_splits_file)
        pnas_splits.columns = ['TCGA ID'] + [str(k) for k in range(1, 16)]
        pnas_splits.index = pnas_splits['TCGA ID']
        pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)
    else:
        raise FileNotFoundError(f"Splits file {pnas_splits_file} not found.")

    cv_splits = {}

    for k in pnas_splits.columns:
        print(f"Creating Split {k}...")
        # Histology embeddings must come from the CNN trained on the training
        # patients of *this* fold; an untrained or shared network would give
        # meaningless features or leak test patients.
        model = None
        if getattr(opt, 'use_vgg_features', 0):
            model = load_fold_path_model(opt, k, device)
        pat_train = pnas_splits.index[pnas_splits[k] == 'Train']
        pat_test = pnas_splits.index[pnas_splits[k] == 'Test']

        tr_name, tr_path, tr_grph, tr_omic, tr_e, tr_t, tr_g = getAlignedMultimodalData(
            opt, model, device, all_dataset, metadata, pat_train, pat2img
        )
        te_name, te_path, te_grph, te_omic, te_e, te_t, te_g = getAlignedMultimodalData(
            opt, model, device, all_dataset, metadata, pat_test, pat2img
        )

        tr_omic = np.array(tr_omic).squeeze(axis=1) if len(tr_omic) > 0 else np.zeros((0, opt.input_size_omic))
        te_omic = np.array(te_omic).squeeze(axis=1) if len(te_omic) > 0 else np.zeros((0, opt.input_size_omic))

        if len(tr_omic) > 0:
            scaler = preprocessing.StandardScaler().fit(tr_omic)
            tr_omic = scaler.transform(tr_omic)
            te_omic = scaler.transform(te_omic)

        cv_splits[int(k)] = {
            'train': {
                'x_patname': tr_name, 'x_path': np.array(tr_path),
                'x_grph': tr_grph, 'x_omic': tr_omic,
                'e': np.array(tr_e, dtype=np.float64),
                't': np.array(tr_t, dtype=np.float64),
                'g': np.array(tr_g, dtype=np.int64)
            },
            'test': {
                'x_patname': te_name, 'x_path': np.array(te_path),
                'x_grph': te_grph, 'x_omic': te_omic,
                'e': np.array(te_e, dtype=np.float64),
                't': np.array(te_t, dtype=np.float64),
                'g': np.array(te_g, dtype=np.int64)
            }
        }

    out_file = os.path.join(
        opt.dataroot, 'splits',
        f"gbmlgg15cv_{opt.roi_dir}_{opt.ignore_missing_moltype}_{opt.ignore_missing_histype}_{opt.use_vgg_features}{'_rnaseq' if opt.use_rnaseq else ''}.pkl"
    )
    data_dict = {'data_pd': all_dataset, 'cv_splits': cv_splits}
    with open(out_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved splits to {out_file}")
    return out_file


if __name__ == '__main__':
    from config import parse_args
    create_splits(parse_args())
