"""
Build an ROI-level split with histology embeddings from a patch-level split.

The trimodal network (CNN ⊗ GCN ⊗ SNN) works on 1024-px ROIs: one cell graph
and one histology embedding per ROI. The Mahmood Lab archive ships such a
split for survival (`gbmlgg15cv_all_st_1_0_1_rnaseq.pkl`) but not for grade
classification. This script derives it: for every fold, the embedding of an
ROI is the mean of the embeddings of its nine 512-px patches, taken from the
patch-level split of the *same* fold (so the embeddings still come from the
histology CNN trained on that fold's training patients only).

Usage (grade classification, the default):
    python -m datasets.make_roi_feature_split
"""

import os
import pickle
import argparse
import numpy as np


def roi_name(path):
    """ROI identifier of a graph / image path (file name without extension)."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def build(roi_split_file, patch_split_file, out_file):
    with open(roi_split_file, 'rb') as f:
        roi_data = pickle.load(f)
    with open(patch_split_file, 'rb') as f:
        patch_splits = pickle.load(f)['cv_splits']

    for k, fold in roi_data['cv_splits'].items():
        for split in ('train', 'test'):
            patches = patch_splits[k][split]
            feats = np.asarray(patches['x_path'])
            feats = feats.reshape(len(feats), -1)
            # patch graph name = <roi>_<x>_<y>
            patch_roi = np.array([roi_name(g).rsplit('_', 2)[0]
                                  for g in patches['x_grph']])
            order = np.argsort(patch_roi, kind='stable')
            uniq, start, counts = np.unique(patch_roi[order],
                                            return_index=True,
                                            return_counts=True)
            sums = np.add.reduceat(feats[order], start, axis=0)
            mean_by_roi = dict(zip(uniq, sums / counts[:, None]))

            rois = [roi_name(g) for g in fold[split]['x_grph']]
            missing = [r for r in rois if r not in mean_by_roi]
            if missing:
                raise KeyError(
                    f"fold {k} {split}: {len(missing)} ROIs have no patch "
                    f"embeddings, e.g. {missing[0]}")
            fold[split]['x_path'] = np.stack(
                [mean_by_roi[r] for r in rois]).astype(np.float32)[:, None, :]

    with open(out_file, 'wb') as f:
        pickle.dump(roi_data, f)
    print(f"Saved {out_file}")


if __name__ == '__main__':
    splits = './data/TCGA_GBMLGG/splits'
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_split', default=os.path.join(
        splits, 'gbmlgg15cv_all_st_0_1_0.pkl'))
    parser.add_argument('--patch_split', default=os.path.join(
        splits, 'gbmlgg15cv_all_st_patches_512_0_1_1.pkl'))
    parser.add_argument('--out', default=os.path.join(
        splits, 'gbmlgg15cv_all_st_0_1_1.pkl'))
    args = parser.parse_args()
    build(args.roi_split, args.patch_split, args.out)
