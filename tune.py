"""
Hyper-parameter selection on inner validation splits.

For every fold, a share of the *training* patients is held out as an inner
validation set. Each configuration of the grid is trained on the remaining
training patients and scored on that inner validation set; the configuration
with the best mean score over the folds is written to
`results/tuning/best_<task>_<mode>.json`.

The test folds are never read here. Fusion models are tuned with a genomic
network that is itself pre-trained on the inner training patients only.

Caveats:
- One configuration is shared by all folds (not fully nested CV): a
  patient held out for inner validation in one fold can be a test patient
  of another fold.
- The histology embeddings stored in the split files come from a CNN that
  was trained on all training rows of the fold, so inner validation is
  optimistic for histology and may favour configurations that lean on it.

Usage:
    python tune.py --task surv --mode omic
    python tune.py --task surv --mode pathomic
"""

import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import parse_args, get_device
from datasets.data_loader import (
    valid_indices, first_row_per_patient, patient_val_split, patient_ids
)
from experiments import (
    GRIDS, FUSION_MODES, TUNING_DIR, experiment_argv, overrides_to_argv,
    best_config_path
)
import train as T


def primary_metric(task):
    return 'cindex_patient' if task == 'surv' else 'auc_patient'


_CV_CACHE = {}


def _score_config(job):
    """Worker entry point: score one configuration (runs in a subprocess)."""
    import torch
    task, mode, overrides, folds, val_frac, gpu_ids = job
    if gpu_ids == '-1':
        torch.set_num_threads(1)
    mean, se = inner_validation_score(task, mode, overrides, _CV_CACHE,
                                      folds, val_frac, gpu_ids)
    return overrides, mean, se


def inner_validation_score(task, mode, overrides, cv_cache, folds, val_frac,
                           gpu_ids='-1'):
    """Mean inner-validation score of one configuration over the folds."""
    device_argv = ['--gpu_ids', gpu_ids]
    opt = parse_args(experiment_argv(task, mode, tuned=False)
                     + overrides_to_argv(overrides) + device_argv)
    device = get_device(opt)
    if mode not in cv_cache:
        cv_cache[mode] = T.load_splits(opt)[1]
    else:
        first = cv_cache[mode][sorted(cv_cache[mode].keys())[0]]
        opt.input_size_omic = int(np.asarray(first['train']['x_omic']).shape[1])
    cv_splits = cv_cache[mode]

    opt_omic = None
    if mode in FUSION_MODES and opt.load_unimodal:
        opt_omic = parse_args(experiment_argv(task, 'omic', tuned=True)
                              + device_argv)
        opt_omic.input_size_omic = opt.input_size_omic

    scores = []
    for k in folds:
        fold_data = cv_splits[k]
        names = patient_ids(fold_data['train']['x_patname'])
        idx = valid_indices(fold_data['train'], task)
        tr_idx, va_idx = patient_val_split(names, idx, val_frac, seed=k)
        seed = opt.seed + 1000 * k

        states = None
        if opt_omic is not None:
            omic_model, _ = T.fit(opt_omic, fold_data, k, seed, device,
                                  train_idx=first_row_per_patient(names, tr_idx))
            states = {'net_omic': omic_model.state_dict()}

        train_idx = (first_row_per_patient(names, tr_idx)
                     if mode == 'omic' and opt.dedup_patients else tr_idx)
        model, _ = T.fit(opt, fold_data, k, seed, device,
                         train_idx=train_idx, unimodal_states=states)
        out, va_names = T.predict_split(opt, model, fold_data, device,
                                        split='train', indices=va_idx)
        metrics = T.compute_metrics(opt, out['pred'], out['censor'],
                                    out['time'], out['grade'], va_names)
        scores.append(metrics[primary_metric(task)])
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='surv', choices=['surv', 'grad'])
    parser.add_argument('--mode', default='omic', choices=sorted(GRIDS))
    parser.add_argument('--val_frac', type=float, default=0.25)
    parser.add_argument('--folds', default='all')
    parser.add_argument('--workers', type=int, default=6,
                        help='Configurations scored in parallel')
    parser.add_argument('--gpu_ids', default='-1',
                        help="Device for tuning ('-1' = CPU, which is faster "
                             "for the small feature-based models)")
    args = parser.parse_args()

    os.makedirs(TUNING_DIR, exist_ok=True)
    grid = GRIDS[args.mode]
    keys = list(grid)
    configs = [dict(zip(keys, vals))
               for vals in itertools.product(*(grid[key] for key in keys))]

    probe = parse_args(experiment_argv(args.task, args.mode, tuned=False)
                       + ['--gpu_ids', '-1'])
    folds = (sorted(T.load_splits(probe)[1]) if args.folds == 'all'
             else [int(x) for x in args.folds.split(',')])

    print(f"Tuning {args.task}/{args.mode}: {len(configs)} configurations, "
          f"{len(folds)} folds, inner validation share {args.val_frac}, "
          f"{args.workers} workers", flush=True)
    jobs = [(args.task, args.mode, overrides, folds, args.val_frac,
             args.gpu_ids) for overrides in configs]
    rows, start = [], time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(args.workers) as pool:
            results = pool.imap_unordered(_score_config, jobs)
            for i, (overrides, mean, se) in enumerate(results, 1):
                rows.append({**overrides, 'val_score': mean, 'val_se': se})
                print(f"[{i:>2}/{len(configs)}] {overrides} -> "
                      f"{mean:.4f} ± {se:.4f}  ({time.time() - start:.0f}s)",
                      flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            overrides, mean, se = _score_config(job)
            rows.append({**overrides, 'val_score': mean, 'val_se': se})
            print(f"[{i:>2}/{len(configs)}] {overrides} -> "
                  f"{mean:.4f} ± {se:.4f}  ({time.time() - start:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows).sort_values('val_score', ascending=False)
    df.to_csv(os.path.join(TUNING_DIR, f'grid_{args.task}_{args.mode}.csv'),
              index=False)
    best = df.iloc[0]
    overrides = {key: (best[key].item() if hasattr(best[key], 'item')
                       else best[key]) for key in keys}
    for key in keys:  # keep integer-valued options integer
        if all(isinstance(v, int) for v in grid[key]):
            overrides[key] = int(overrides[key])
    with open(best_config_path(args.task, args.mode), 'w') as f:
        json.dump({'task': args.task, 'mode': args.mode,
                   'metric': primary_metric(args.task),
                   'val_score': float(best['val_score']),
                   'val_se': float(best['val_se']),
                   'val_frac': args.val_frac, 'n_folds': len(folds),
                   'overrides': overrides}, f, indent=2)
    print(f"\nBest: {overrides} (inner validation "
          f"{primary_metric(args.task)} = {best['val_score']:.4f})")


if __name__ == '__main__':
    main()
