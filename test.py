"""
Evaluation Pipeline for Pathomic Fusion.

Re-loads the trained checkpoints of every cross-validation fold, averages the
predictions of the per-fold ensemble members and reports:
- Survival: concordance index per ROI row and per patient, patient-level
  log-rank p-value
- Grade: AUC, Average Precision, micro-F1, Grade IV F1 (per ROI and per patient)
- Mean, standard error and 95% confidence interval across folds

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import sys
import glob
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
from train import (
    load_splits, get_fold_indices, predict_split, compute_metrics
)
from utils.metrics import CI_interval


def fold_checkpoints(checkpoint_dir, model_name, k):
    """Checkpoints of all ensemble members of fold k."""
    return sorted(glob.glob(
        os.path.join(checkpoint_dir, f"{model_name}_{k}_s*.pt")))


def run_testing(opt):
    """Evaluate the trained checkpoints across cross-validation folds."""
    device = get_device(opt)
    split_file, cv_splits = load_splits(opt)
    print_options(opt)
    print(f"Loading CV splits from: {split_file}")

    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.exp_name,
                                  opt.model_name)
    results = []
    for k in get_fold_indices(opt, cv_splits):
        ckpts = fold_checkpoints(checkpoint_dir, opt.model_name, k)
        if not ckpts:
            print(f"[Warning] No checkpoint for fold {k} in {checkpoint_dir}. Skipping.")
            continue

        preds, out, names = [], None, None
        for ckpt_path in ckpts:
            model = define_net(opt)
            state = torch.load(ckpt_path, map_location=device,
                               weights_only=False)
            model.load_state_dict(state.get('model_state_dict', state))
            out, names = predict_split(opt, model, cv_splits[k], device)
            preds.append(out['pred'])

        ensemble = np.mean(np.stack(preds, axis=0), axis=0)
        metrics = compute_metrics(opt, ensemble, out['censor'], out['time'],
                                  out['grade'], names)
        results.append({'fold': k, 'n_models': len(ckpts), **metrics})
        if opt.task == 'surv':
            print(f"Fold {k:>2} ({len(ckpts)} models): c-index ROI = "
                  f"{metrics['cindex']:.4f} | patient = "
                  f"{metrics['cindex_patient']:.4f}")
        else:
            print(f"Fold {k:>2} ({len(ckpts)} models): AUC = "
                  f"{metrics['auc']:.4f} | F1 = {metrics['f1_micro']:.4f}")

    if not results:
        print("No evaluations completed. Ensure checkpoints exist.")
        return None

    df_results = pd.DataFrame(results)
    os.makedirs(opt.results_dir, exist_ok=True)
    out_csv = os.path.join(
        opt.results_dir,
        f"test_{opt.exp_name}_{opt.model_name}_{opt.mode}_{opt.task}.csv")
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print(f"  Evaluation over {len(results)} folds: {opt.model_name}")
    print("=" * 70)
    for col in df_results.columns:
        if col in ('fold', 'n_models') or 'pvalue' in col:
            continue
        vals = df_results[col].dropna()
        low, high = CI_interval(vals)
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {col:>20}: {vals.mean():.4f} ± {se:.4f} (SE)  "
              f"[95% CI: {low:.4f} - {high:.4f}]")
    print("=" * 70)
    print(f"Saved results to: {out_csv}\n")
    return df_results


if __name__ == '__main__':
    run_testing(parse_args())
