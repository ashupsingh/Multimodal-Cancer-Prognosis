"""
Training Pipeline for Pathomic Fusion.
Multimodal Cancer Prognosis using Histopathology Images and Genomic Features.

Supports:
- Unimodal models: CNN (Path), GCN (Graph), SNN (Omic)
- Multimodal fusion: CNN⊗SNN, GCN⊗SNN, CNN⊗GCN⊗SNN (Pathomic, Graphomic, Pathgraphomic)
- Ablation baselines: Path-Path, Graph-Graph, Omic-Omic
- Tasks: Survival Outcome Prediction (Cox Loss), Grade Classification (NLL Loss)
- 15-Fold Cross-Validation with linear LR decay, unimodal unfreezing, and C-Index / AUC evaluation

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import parse_args, get_device, print_options, save_options
from models.networks import define_net, define_optimizer, define_scheduler, define_reg
from datasets.data_loader import build_dataloader
from utils.helpers import unfreeze_unimodal
from utils.loss import CoxLoss
from utils.metrics import (
    CIndex_lifeline, cox_log_rank, accuracy_cox,
    calcGradMetrics, CI_interval
)


def set_seed(seed=2019):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def resolve_split_file(opt):
    """Find the best matching split file for the experiment options."""
    if hasattr(opt, 'split_path') and opt.split_path and os.path.exists(opt.split_path):
        return opt.split_path

    splits_dir = os.path.join(opt.dataroot, 'splits')
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"Splits directory {splits_dir} does not exist.")

    # Candidate filename patterns
    use_rnaseq_str = '_rnaseq' if getattr(opt, 'use_rnaseq', 0) else ''
    vgg_feat = getattr(opt, 'use_vgg_features', 0)

    preferred_names = []
    if opt.mode in ('pathomic', 'pathgraphomic') or vgg_feat:
        preferred_names.append(f"gbmlgg15cv_all_st_patches_512_1_0_1{use_rnaseq_str}.pkl")
        preferred_names.append(f"gbmlgg15cv_all_st_patches_512_0_0_1.pkl")
        preferred_names.append(f"gbmlgg15cv_all_st_patches_512_1_1_1{use_rnaseq_str}.pkl")
    elif opt.mode == 'omic':
        preferred_names.append(f"gbmlgg15cv_all_st_1_0_0{use_rnaseq_str}.pkl")
        preferred_names.append(f"gbmlgg15cv_all_st_1_1_0{use_rnaseq_str}.pkl")
        preferred_names.append(f"gbmlgg15cv_all_st_patches_512_1_0_1{use_rnaseq_str}.pkl")
    else:
        preferred_names.append(f"gbmlgg15cv_all_st_0_0_0.pkl")
        preferred_names.append(f"gbmlgg15cv_all_st_patches_512_0_0_0.pkl")

    for name in preferred_names:
        p = os.path.join(splits_dir, name)
        if os.path.exists(p):
            return p

    # Fallback: any pkl in splits
    all_pkls = [f for f in os.listdir(splits_dir) if f.endswith('.pkl')]
    if all_pkls:
        return os.path.join(splits_dir, all_pkls[0])

    raise FileNotFoundError(f"No split files found in {splits_dir}.")


def train_epoch(opt, model, dataloader, optimizer, device):
    """Run one training epoch."""
    model.train()
    loss_total = 0.0
    risk_all, censor_all, time_all, grade_all = [], [], [], []
    grad_acc_total = 0
    total_samples = 0

    for batch in dataloader:
        x_path, x_grph, x_omic, censor, survtime, grade = batch

        # Move to device
        censor = censor.to(device)
        survtime = survtime.to(device)
        grade = grade.to(device)

        inputs = {}
        if isinstance(x_path, torch.Tensor):
            inputs['x_path'] = x_path.to(device)
        if hasattr(x_grph, 'to'):
            inputs['x_grph'] = x_grph.to(device)
        if isinstance(x_omic, torch.Tensor):
            inputs['x_omic'] = x_omic.to(device)

        # Forward
        optimizer.zero_grad()
        feat, pred = model(**inputs)

        # Loss calculation
        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == 'surv' else torch.tensor(0.0, device=device)
        loss_nll = F.nll_loss(pred, grade) if opt.task == 'grad' else torch.tensor(0.0, device=device)
        loss_reg = define_reg(opt, model)

        loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll + loss_reg
        loss.backward()
        optimizer.step()

        batch_sz = len(censor)
        loss_total += loss.item() * batch_sz
        total_samples += batch_sz

        if opt.task == 'surv':
            risk_all.extend(pred.detach().cpu().view(-1).numpy().tolist())
            censor_all.extend(censor.detach().cpu().view(-1).numpy().tolist())
            time_all.extend(survtime.detach().cpu().view(-1).numpy().tolist())
        elif opt.task == 'grad':
            cls_pred = pred.argmax(dim=1)
            grad_acc_total += (cls_pred == grade).sum().item()

    avg_loss = loss_total / max(1, total_samples)

    metrics = {'loss': avg_loss}
    if opt.task == 'surv':
        metrics['cindex'] = CIndex_lifeline(np.array(risk_all), np.array(censor_all), np.array(time_all))
        metrics['pvalue'] = cox_log_rank(np.array(risk_all), np.array(censor_all), np.array(time_all))
        metrics['surv_acc'] = accuracy_cox(np.array(risk_all), np.array(censor_all))
    elif opt.task == 'grad':
        metrics['accuracy'] = grad_acc_total / max(1, total_samples)

    return metrics


@torch.no_grad()
def evaluate(opt, model, dataloader, device):
    """Evaluate model on validation or test split."""
    model.eval()
    loss_total = 0.0
    risk_all, censor_all, time_all, grade_all = [], [], [], []
    probs_all = []
    total_samples = 0

    for batch in dataloader:
        x_path, x_grph, x_omic, censor, survtime, grade = batch

        censor = censor.to(device)
        survtime = survtime.to(device)
        grade = grade.to(device)

        inputs = {}
        if isinstance(x_path, torch.Tensor):
            inputs['x_path'] = x_path.to(device)
        if hasattr(x_grph, 'to'):
            inputs['x_grph'] = x_grph.to(device)
        if isinstance(x_omic, torch.Tensor):
            inputs['x_omic'] = x_omic.to(device)

        feat, pred = model(**inputs)

        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == 'surv' else torch.tensor(0.0, device=device)
        loss_nll = F.nll_loss(pred, grade) if opt.task == 'grad' else torch.tensor(0.0, device=device)
        loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll

        batch_sz = len(censor)
        loss_total += loss.item() * batch_sz
        total_samples += batch_sz

        if opt.task == 'surv':
            risk_all.extend(pred.detach().cpu().view(-1).numpy().tolist())
            censor_all.extend(censor.detach().cpu().view(-1).numpy().tolist())
            time_all.extend(survtime.detach().cpu().view(-1).numpy().tolist())
        elif opt.task == 'grad':
            probs = torch.exp(pred) if isinstance(model.act, torch.nn.LogSoftmax) else pred
            probs_all.append(probs.detach().cpu().numpy())
            grade_all.extend(grade.detach().cpu().view(-1).numpy().tolist())

    avg_loss = loss_total / max(1, total_samples)
    metrics = {'loss': avg_loss}

    if opt.task == 'surv':
        r, c, t = np.array(risk_all), np.array(censor_all), np.array(time_all)
        metrics['cindex'] = CIndex_lifeline(r, c, t)
        metrics['pvalue'] = cox_log_rank(r, c, t)
        metrics['surv_acc'] = accuracy_cox(r, c)
        predictions = {'risk': r, 'censor': c, 'time': t}
    elif opt.task == 'grad':
        probs_mat = np.concatenate(probs_all, axis=0) if probs_all else np.array([])
        targets = np.array(grade_all)
        grad_metrics = calcGradMetrics(probs_mat, targets)
        metrics.update(grad_metrics)
        predictions = {'probs': probs_mat, 'grade': targets}

    return metrics, predictions


def run_cross_validation(opt):
    """Execute complete 15-fold cross-validation experiment."""
    set_seed(2019)
    device = get_device(opt)
    print_options(opt)
    save_options(opt)

    split_file = resolve_split_file(opt)
    print(f"Loading CV splits from: {split_file}")
    with open(split_file, 'rb') as f:
        split_dict = pickle.load(f)

    cv_splits = split_dict['cv_splits']
    total_folds = len(cv_splits)
    print(f"Found {total_folds} cross-validation folds.")

    # Select folds to train
    target_folds = getattr(opt, 'folds', None)
    if target_folds is not None and target_folds != 'all':
        fold_indices = [int(x.strip()) for x in str(target_folds).split(',')]
    else:
        fold_indices = list(cv_splits.keys())

    fold_results = []
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f" Starting Experiment: {opt.exp_name} | Mode: {opt.mode} | Task: {opt.task}")
    print(f" Training on folds: {fold_indices}")
    print("=" * 60 + "\n")

    for k in fold_indices:
        print(f"\n-- Fold [{k}/{total_folds}] ------------------------------")
        model = define_net(opt)
        optimizer = define_optimizer(opt, model)
        scheduler = define_scheduler(opt, optimizer)

        train_loader = build_dataloader(opt, cv_splits[k], split='train', mode=opt.mode, batch_size=opt.batch_size, shuffle=True)
        test_loader = build_dataloader(opt, cv_splits[k], split='test', mode=opt.mode, batch_size=opt.batch_size, shuffle=False)

        best_metric = -float('inf') if opt.task == 'surv' else 0.0
        best_epoch = 0
        best_test_metrics = None
        best_predictions = None

        total_epochs = opt.niter + opt.niter_decay

        pbar = tqdm(range(opt.epoch_count, total_epochs + 1), desc=f"Fold {k}")
        for epoch in pbar:
            # Unfreeze unimodal layers at epoch 5 for fusion networks
            if opt.finetune:
                unfreeze_unimodal(opt, model, epoch)

            train_metrics = train_epoch(opt, model, train_loader, optimizer, device)
            scheduler.step()

            test_metrics, test_preds = evaluate(opt, model, test_loader, device)

            # Track primary metric
            current_metric = test_metrics['cindex'] if opt.task == 'surv' else test_metrics.get('accuracy', 0.0)

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_test_metrics = test_metrics
                best_predictions = test_preds

                # Save best checkpoint for this fold
                ckpt_path = os.path.join(checkpoint_dir, f"{opt.model_name}_{k}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': test_metrics
                }, ckpt_path)

            if opt.task == 'surv':
                pbar.set_postfix({
                    'Tr_Loss': f"{train_metrics['loss']:.3f}",
                    'Tr_CIdx': f"{train_metrics.get('cindex', 0):.3f}",
                    'Te_CIdx': f"{test_metrics.get('cindex', 0):.3f}",
                    'Best': f"{best_metric:.3f} (ep {best_epoch})"
                })
            else:
                pbar.set_postfix({
                    'Tr_Loss': f"{train_metrics['loss']:.3f}",
                    'Tr_Acc': f"{train_metrics.get('accuracy', 0):.3f}",
                    'Te_Acc': f"{test_metrics.get('accuracy', 0):.3f}",
                    'Best': f"{best_metric:.3f} (ep {best_epoch})"
                })

        # Save test predictions for fold
        pred_path = os.path.join(checkpoint_dir, f"{opt.model_name}_{k}_pred_test.pkl")
        with open(pred_path, 'wb') as pf:
            pickle.dump(best_predictions, pf)

        print(f"Fold {k} finished! Best epoch: {best_epoch} | Best {('C-Index' if opt.task == 'surv' else 'Accuracy')}: {best_metric:.4f}")
        fold_results.append({
            'fold': k,
            'best_epoch': best_epoch,
            **best_test_metrics
        })

    # Summary across folds
    df_results = pd.DataFrame(fold_results)
    results_dir = getattr(opt, 'results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{opt.exp_name}_{opt.model_name}_{opt.mode}_{opt.task}_summary.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"  Cross-Validation Results Summary: {opt.model_name} ({opt.mode.upper()})")
    print("=" * 60)
    for col in df_results.columns:
        if col not in ('fold', 'best_epoch'):
            vals = df_results[col].dropna()
            mean_v, std_v = vals.mean(), vals.std()
            se_v = std_v / np.sqrt(len(vals)) if len(vals) > 1 else 0
            print(f"  {col:>15}: Mean = {mean_v:.4f} +/- {std_v:.4f} (SE = {se_v:.4f})")
    print("=" * 60)
    print(f"Results saved to: {csv_path}\n")

    return df_results


if __name__ == '__main__':
    opt = parse_args()
    run_cross_validation(opt)
