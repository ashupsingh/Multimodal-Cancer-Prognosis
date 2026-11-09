"""
Evaluation and Testing Pipeline for Pathomic Fusion.

Loads trained model checkpoints across 15-fold cross-validation
and calculates final evaluation metrics:
- Survival: Concordance Index (C-Index), log-rank p-value, Cox accuracy
- Grade: Multiclass AUC, Average Precision (AP), Micro-F1, Grade IV F1
- 95% Confidence Intervals across folds

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import parse_args, get_device, print_options
from models.networks import define_net
from datasets.data_loader import build_dataloader
from train import resolve_split_file, evaluate
from utils.metrics import CI_interval


def run_testing(opt):
    """Evaluate trained model checkpoints across cross-validation folds."""
    device = get_device(opt)
    print_options(opt)

    split_file = resolve_split_file(opt)
    print(f"Loading CV splits from: {split_file}")
    with open(split_file, 'rb') as f:
        split_dict = pickle.load(f)

    cv_splits = split_dict['cv_splits']
    total_folds = len(cv_splits)

    target_folds = getattr(opt, 'folds', None)
    if target_folds is not None and target_folds != 'all':
        fold_indices = [int(x.strip()) for x in str(target_folds).split(',')]
    else:
        fold_indices = list(cv_splits.keys())

    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    results = []

    print("\n" + "=" * 60)
    print(f" Testing: {opt.model_name} | Mode: {opt.mode} | Task: {opt.task}")
    print("=" * 60 + "\n")

    for k in fold_indices:
        ckpt_path = os.path.join(checkpoint_dir, f"{opt.model_name}_{k}_best.pt")
        if not os.path.exists(ckpt_path):
            # Try alternate naming
            ckpt_path = os.path.join(checkpoint_dir, f"{opt.model_name}_{k}.pt")

        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint for fold {k} not found at {ckpt_path}. Skipping.")
            continue

        print(f"Evaluating fold {k} using checkpoint: {ckpt_path}")
        model = define_net(opt)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

        test_loader = build_dataloader(
            opt, cv_splits[k], split='test', mode=opt.mode,
            batch_size=opt.batch_size, shuffle=False
        )

        test_metrics, test_preds = evaluate(opt, model, test_loader, device)
        results.append({'fold': k, **test_metrics})

        if opt.task == 'surv':
            print(f"Fold {k}: C-Index = {test_metrics['cindex']:.4f} | Log-rank p = {test_metrics['pvalue']:.4e}")
        else:
            print(f"Fold {k}: Accuracy = {test_metrics.get('accuracy', 0):.4f} | AUC = {test_metrics.get('auc', 0):.4f} | F1 = {test_metrics.get('f1_micro', 0):.4f}")

    if not results:
        print("No evaluations completed. Ensure checkpoints exist.")
        return None

    df_results = pd.DataFrame(results)
    results_dir = getattr(opt, 'results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"test_{opt.exp_name}_{opt.model_name}_{opt.mode}_{opt.task}.csv")
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 65)
    print(f"  Final Evaluation Across {len(results)} Folds: {opt.model_name}")
    print("=" * 65)
    for col in df_results.columns:
        if col != 'fold':
            vals = df_results[col].dropna()
            mean_v = vals.mean()
            std_v = vals.std() if len(vals) > 1 else 0
            low_ci, high_ci = CI_interval(vals) if len(vals) > 1 else (mean_v, mean_v)
            print(f"  {col:>15}: {mean_v:.4f} +/- {std_v:.4f}  [95% CI: {low_ci:.4f} - {high_ci:.4f}]")
    print("=" * 65)
    print(f"Saved results to: {out_csv}\n")

    return df_results


if __name__ == '__main__':
    opt = parse_args()
    run_testing(opt)
