"""
Training Pipeline for Pathomic Fusion.
Multimodal Cancer Prognosis using Histopathology Images and Genomic Features.

Supports:
- Unimodal models: CNN (Path), GCN (Graph), SNN (Omic)
- Multimodal fusion: CNN⊗SNN, GCN⊗SNN, CNN⊗GCN⊗SNN
- Tasks: Survival Outcome Prediction (Cox loss), Grade Classification (NLL)

Evaluation protocol (15-fold Monte-Carlo cross-validation):
- Every model is trained for a fixed number of epochs and the *final* epoch
  is evaluated on the test fold. The test fold is never used for model
  selection, early stopping, or hyper-parameter choice.
- Fusion networks load the per-fold pre-trained unimodal networks, keep them
  frozen for the first epochs, then fine-tune them (Chen et al., 2022).
- `--n_seeds N` trains N models per fold and averages their predictions.
- Metrics are reported per ROI row (as in the paper) and per patient
  (predictions of a patient's ROIs averaged).

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import sys
import copy
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import parse_args, get_device, print_options, save_options
from models.networks import (
    define_net, define_optimizer, define_scheduler, define_reg
)
from datasets.data_loader import (
    build_dataloader, valid_indices, first_row_per_patient, patient_ids
)
from utils.helpers import freeze_unimodal, unfreeze_unimodal
from utils.loss import CoxLoss
from utils.metrics import (
    CIndex_lifeline, cox_log_rank, grade_metrics, aggregate_by_patient
)

FUSION_MODES = ('pathomic', 'graphomic', 'pathgraphomic')
UNIMODAL_SOURCES = {'net_omic': 'unimodal_name',
                    'net_grph': 'unimodal_grph_name'}


def set_seed(seed=2019):
    """Seed every random number generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ── Split files ───────────────────────────────────────────────────────

def resolve_split_file(opt):
    """
    Find the split file matching the experiment options.

    File name pattern (Mahmood Lab splits):
        gbmlgg15cv_<roi_dir>_<ignore_missing_moltype>_<ignore_missing_histype>_<use_vgg_features>[_rnaseq].pkl
    Survival uses patients with a known molecular subtype (1_0); grade
    classification uses patients with a known grade (0_1, or 1_1 with RNA-Seq).
    """
    if getattr(opt, 'split_path', '') and os.path.exists(opt.split_path):
        return opt.split_path

    splits_dir = os.path.join(opt.dataroot, 'splits')
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"Splits directory {splits_dir} does not exist.")

    rnaseq = '_rnaseq' if getattr(opt, 'use_rnaseq', 0) else ''
    vgg = 1 if getattr(opt, 'use_vgg_features', 0) else 0
    if opt.task == 'grad':
        moltype, histype = (1, 1) if rnaseq else (0, 1)
    else:
        moltype, histype = 1, 0

    if opt.mode in ('path', 'pathpath', 'pathomic'):
        # one row per 512-px patch, histology embedding per patch
        name = f"gbmlgg15cv_all_st_patches_512_{moltype}_{histype}_{vgg}{rnaseq}.pkl"
    elif opt.mode == 'pathgraphomic':
        # one row per 1024-px ROI: ROI-level histology embedding + cell graph
        name = f"gbmlgg15cv_all_st_{moltype}_{histype}_{vgg}{rnaseq}.pkl"
    else:
        # one row per 1024-px ROI: genomic vector and/or cell graph
        name = f"gbmlgg15cv_all_st_{moltype}_{histype}_0{rnaseq}.pkl"

    path = os.path.join(splits_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file {path} not found. Pass --split_path explicitly."
        )
    return path


def load_splits(opt):
    """Load the CV splits and set the genomic input size from the data."""
    split_file = resolve_split_file(opt)
    with open(split_file, 'rb') as f:
        cv_splits = pickle.load(f)['cv_splits']
    first = cv_splits[sorted(cv_splits.keys())[0]]
    opt.input_size_omic = int(np.asarray(first['train']['x_omic']).shape[1])
    return split_file, cv_splits


def get_fold_indices(opt, cv_splits):
    """Folds selected by --folds ('all' or a comma separated list)."""
    target = str(getattr(opt, 'folds', 'all'))
    if target in ('all', 'None', ''):
        return sorted(cv_splits.keys())
    return [int(x.strip()) for x in target.split(',')]


def default_train_indices(opt, fold_data):
    """
    Training rows of a fold.

    In unimodal omic mode every row of a patient carries the same genomic
    vector, so `--dedup_patients 1` keeps one row per patient: each patient
    then has equal weight in the loss.
    """
    idx = valid_indices(fold_data['train'], opt.task)
    if opt.mode == 'omic' and getattr(opt, 'dedup_patients', 1):
        idx = first_row_per_patient(fold_data['train']['x_patname'], idx)
    return idx


# ── One optimisation epoch / prediction ───────────────────────────────

def _unpack(batch, device):
    x_path, x_grph, x_omic, censor, survtime, grade = batch
    inputs = {}
    if isinstance(x_path, torch.Tensor):
        inputs['x_path'] = x_path.to(device)
    if hasattr(x_grph, 'to'):
        inputs['x_grph'] = x_grph.to(device)
    if isinstance(x_omic, torch.Tensor):
        inputs['x_omic'] = x_omic.to(device)
    return inputs, censor.to(device), survtime.to(device), grade.to(device)


def train_epoch(opt, model, dataloader, optimizer, device):
    """Run one training epoch. Returns the mean training loss."""
    model.train()
    loss_total, n_total = 0.0, 0

    for batch in dataloader:
        inputs, censor, survtime, grade = _unpack(batch, device)

        optimizer.zero_grad()
        _, pred = model(**inputs)

        if opt.task == 'surv':
            loss = opt.lambda_cox * CoxLoss(survtime, censor, pred)
        else:
            loss = opt.lambda_nll * F.nll_loss(pred, grade)
        loss = loss + define_reg(opt, model)

        loss.backward()
        optimizer.step()

        loss_total += loss.item() * len(censor)
        n_total += len(censor)

    return loss_total / max(1, n_total)


@torch.no_grad()
def predict(opt, model, dataloader, device):
    """
    Predict a whole split.

    Returns:
        dict with 'pred' (risk scores (N,) for survival, class probabilities
        (N, C) for grade), 'censor', 'time', 'grade'
    """
    model.eval()
    preds, censors, times, grades = [], [], [], []
    for batch in dataloader:
        inputs, censor, survtime, grade = _unpack(batch, device)
        _, pred = model(**inputs)
        if opt.task == 'grad':
            pred = torch.exp(pred) if opt.act_type == 'LSM' else pred
            preds.append(pred.cpu().numpy())
        else:
            preds.append(pred.view(-1).cpu().numpy())
        censors.append(censor.cpu().numpy())
        times.append(survtime.cpu().numpy())
        grades.append(grade.cpu().numpy())
    return {
        'pred': np.concatenate(preds, axis=0),
        'censor': np.concatenate(censors),
        'time': np.concatenate(times),
        'grade': np.concatenate(grades),
    }


def compute_metrics(opt, pred, censor, time, grade, names=None):
    """
    Metrics for one set of predictions, per ROI row and (when patient names
    are given) per patient.
    """
    out = {}
    if opt.task == 'surv':
        out['cindex'] = CIndex_lifeline(pred, censor, time)
        if names is not None:
            _, p, c, t = aggregate_by_patient(names, pred, censor, time)
            out['cindex_patient'] = CIndex_lifeline(p, c, t)
            out['pvalue_patient'] = cox_log_rank(p, c, t)
    else:
        out.update(grade_metrics(pred, grade, n_classes=opt.label_dim))
        if names is not None:
            _, p, g = aggregate_by_patient(names, pred, grade)
            out.update({f"{key}_patient": val for key, val in
                        grade_metrics(p, g, n_classes=opt.label_dim).items()})
    return out


# ── Pre-trained unimodal networks ─────────────────────────────────────

def unimodal_checkpoint(opt, attr, k, seed_idx):
    """Path of the pre-trained unimodal checkpoint for fold k (or None)."""
    name = getattr(opt, UNIMODAL_SOURCES[attr])
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, name)
    for fname in (f"{name}_{k}_s{seed_idx}.pt", f"{name}_{k}_s0.pt"):
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            return path
    return None


def load_pretrained_unimodal(opt, model, k, seed_idx, device, states=None):
    """
    Load pre-trained unimodal networks into a fusion model and freeze them.

    Args:
        states: optional {attr: state_dict} overriding the checkpoints on
                disk (used for inner-validation tuning)

    Returns:
        list of sub-network names that were loaded and frozen
    """
    loaded = []
    for attr in UNIMODAL_SOURCES:
        net = getattr(model, attr, None)
        if net is None:
            continue
        state = (states or {}).get(attr)
        if state is None:
            path = unimodal_checkpoint(opt, attr, k, seed_idx)
            if path is not None:
                state = torch.load(path, map_location=device,
                                   weights_only=False)['model_state_dict']
        if state is None:
            continue
        net.load_state_dict(state)
        loaded.append(attr)
    freeze_unimodal(model, loaded)
    return loaded


# ── Fit one model ─────────────────────────────────────────────────────

def fit(opt, fold_data, k, seed, device, seed_idx=0, train_idx=None,
        unimodal_states=None):
    """
    Train one model on (a subset of) the training split of a fold.

    Args:
        seed: random seed of this model
        seed_idx: index of the model within the per-fold ensemble
        train_idx: training rows (default: `default_train_indices`)
        unimodal_states: optional pre-trained unimodal state dicts

    Returns:
        (model, info) where info holds the loss history and the list of
        pre-trained sub-networks that were loaded
    """
    set_seed(seed)
    model = define_net(opt)

    pretrained = []
    if opt.mode in FUSION_MODES and getattr(opt, 'load_unimodal', 1):
        pretrained = load_pretrained_unimodal(
            opt, model, k, seed_idx, device, states=unimodal_states)

    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)

    if train_idx is None:
        train_idx = default_train_indices(opt, fold_data)
    train_loader = build_dataloader(
        opt, fold_data, split='train', mode=opt.mode,
        batch_size=opt.batch_size, shuffle=True, device=device,
        indices=train_idx, seed=seed)

    losses = []
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        if opt.finetune:
            unfreeze_unimodal(opt, model, epoch)
        losses.append(train_epoch(opt, model, train_loader, optimizer, device))
        scheduler.step()

    return model, {'losses': losses, 'pretrained': pretrained}


def predict_split(opt, model, fold_data, device, split='test', indices=None):
    """Predict a split of a fold; returns (prediction dict, patient ids)."""
    if indices is None:
        indices = valid_indices(fold_data[split], opt.task)
    loader = build_dataloader(
        opt, fold_data, split=split, mode=opt.mode,
        batch_size=(max(opt.batch_size, 256)
                    if opt.mode in ('omic', 'path', 'pathomic')
                    else max(opt.batch_size, 32)),
        shuffle=False, device=device, indices=indices)
    out = predict(opt, model, loader, device)
    names = patient_ids(fold_data[split]['x_patname'])[indices]
    return out, names


def run_fold(opt, k, fold_data, device):
    """
    Train the per-fold ensemble of fold k and evaluate it on the test split.

    Saves one checkpoint per ensemble member and the test predictions.

    Returns:
        (result row, one-line progress message)
    """
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.exp_name,
                                  opt.model_name)
    primary = 'cindex_patient' if opt.task == 'surv' else 'auc_patient'
    seed_preds, single_metrics, pretrained = [], [], []
    test_out, names = None, None

    for s in range(opt.n_seeds):
        seed = opt.seed + 1000 * k + s
        model, info = fit(opt, fold_data, k, seed, device, seed_idx=s)
        pretrained = info['pretrained']
        test_out, names = predict_split(opt, model, fold_data, device)
        seed_preds.append(test_out['pred'])
        single_metrics.append(compute_metrics(
            opt, test_out['pred'], test_out['censor'], test_out['time'],
            test_out['grade'], names)[primary])

        torch.save({
            'fold': k, 'seed': seed, 'seed_idx': s,
            'model_state_dict': model.state_dict(),
            'opt': {key: val for key, val in vars(opt).items()},
        }, os.path.join(checkpoint_dir, f"{opt.model_name}_{k}_s{s}.pt"))

    ensemble = np.mean(np.stack(seed_preds, axis=0), axis=0)
    metrics = compute_metrics(opt, ensemble, test_out['censor'],
                              test_out['time'], test_out['grade'], names)

    pred_key = 'risk' if opt.task == 'surv' else 'probs'
    with open(os.path.join(checkpoint_dir,
                           f"{opt.model_name}_{k}_pred_test.pkl"), 'wb') as pf:
        pickle.dump({
            pred_key: ensemble, 'per_seed': np.stack(seed_preds, axis=0),
            'censor': test_out['censor'], 'time': test_out['time'],
            'grade': test_out['grade'], 'patname': names,
        }, pf)

    row = {'fold': k, 'n_rows': len(names),
           'n_patients': len(np.unique(names)), **metrics,
           f'{primary}_single_model_mean': float(np.mean(single_metrics))}
    note = f" | pretrained: {','.join(pretrained)}" if pretrained else ''
    if opt.task == 'surv':
        message = (f"Fold {k:>2}: c-index ROI = {metrics['cindex']:.4f} | "
                   f"patient = {metrics['cindex_patient']:.4f} | "
                   f"log-rank p = {metrics['pvalue_patient']:.2e}{note}")
    else:
        message = (f"Fold {k:>2}: AUC = {metrics['auc']:.4f} | "
                   f"AP = {metrics['ap']:.4f} | F1 = {metrics['f1_micro']:.4f} | "
                   f"patient AUC = {metrics['auc_patient']:.4f}{note}")
    return row, message


_WORKER_SPLITS = {}


def _fold_worker(job):
    """Process-pool entry point: run one fold in a worker process."""
    opt, k = job
    torch.set_num_threads(1)
    key = resolve_split_file(opt)
    if key not in _WORKER_SPLITS:
        _WORKER_SPLITS[key] = load_splits(opt)[1]
    return run_fold(opt, k, _WORKER_SPLITS[key][k], get_device(opt))


# ── Cross-validation ──────────────────────────────────────────────────

def run_cross_validation(opt):
    """Run the cross-validation experiment described by `opt`."""
    device = get_device(opt)
    split_file, cv_splits = load_splits(opt)
    print_options(opt)
    save_options(opt)
    print(f"Loading CV splits from: {split_file}")

    fold_indices = get_fold_indices(opt, cv_splits)
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" Experiment: {opt.exp_name} | Model: {opt.model_name} | "
          f"Mode: {opt.mode} | Task: {opt.task}")
    print(f" Folds: {fold_indices} | Models per fold: {opt.n_seeds} | "
          f"Epochs: {opt.niter + opt.niter_decay}")
    print("=" * 70)

    jobs = [(opt, k) for k in fold_indices]
    workers = min(int(getattr(opt, 'cv_workers', 1)), len(jobs))
    fold_results = []
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers) as pool:
            for row, message in pool.imap(_fold_worker, jobs):
                print(message, flush=True)
                fold_results.append(row)
    else:
        for k in fold_indices:
            row, message = run_fold(opt, k, cv_splits[k], device)
            print(message, flush=True)
            fold_results.append(row)

    df_results = pd.DataFrame(fold_results)
    results_dir = getattr(opt, 'results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(
        results_dir,
        f"{opt.exp_name}_{opt.model_name}_{opt.mode}_{opt.task}_summary.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print(f"  Summary over {len(df_results)} folds: {opt.model_name} ({opt.mode})")
    print("=" * 70)
    for col in df_results.columns:
        if col in ('fold', 'n_rows', 'n_patients') or 'pvalue' in col:
            continue
        vals = df_results[col].dropna()
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {col:>34}: {vals.mean():.4f} +/- {se:.4f} (SE)")
    print("=" * 70)
    print(f"Results saved to: {csv_path}\n")
    return df_results


if __name__ == '__main__':
    run_cross_validation(parse_args())
