"""
Evaluation metrics for Pathomic Fusion.
- Concordance Index (c-Index) for survival prediction
- AUC, AP, F1 for grade classification
- Log-rank test for survival stratification
"""

import numpy as np
import scipy
import scipy.stats
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score
)


def accuracy(output, labels):
    """Compute classification accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    """
    Compute accuracy based on estimated survival events against true events.
    Dichotomize hazards at median threshold.
    """
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    """
    Perform log-rank test to compare survival between predicted risk groups.
    
    Returns:
        p-value from log-rank test
    """
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    return results.p_value


def CIndex_lifeline(hazards, labels, survtime_all):
    """
    Compute concordance index using lifelines.
    
    Args:
        hazards: Predicted risk scores
        labels: Censoring indicators (1=event)
        survtime_all: Observed survival times
        
    Returns:
        Concordance index (c-Index)
    """
    return concordance_index(survtime_all, -hazards, labels)


def CI_pm(data, confidence=0.95):
    """Compute mean ± confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return "{0:.4f} ± {1:.3f}".format(m, h)


def CI_interval(data, confidence=0.95):
    """Compute confidence interval bounds as (lower, upper) float tuple."""
    a = 1.0 * np.array(data)
    n = len(a)
    if n <= 1:
        m = float(np.mean(a)) if n == 1 else 0.0
        return (m, m)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return (float(m - h), float(m + h))


def CI_interval_str(data, confidence=0.95):
    """Compute confidence interval bounds as formatted string."""
    low, high = CI_interval(data, confidence)
    return f"{low:.3f}, {high:.3f}"


def aggregate_by_patient(names, pred, *labels):
    """
    Collapse row-level (per-ROI) predictions to one prediction per patient.

    Predictions are averaged over a patient's rows; labels (identical within
    a patient) take the first value.

    Returns:
        (patient_ids, pred_patient, *labels_patient)
    """
    names = np.asarray(names)
    pred = np.asarray(pred)
    uniq, first_idx, inv = np.unique(
        names, return_index=True, return_inverse=True
    )
    counts = np.bincount(inv).astype(float)
    if pred.ndim == 1:
        agg = np.bincount(inv, weights=pred) / counts
    else:
        agg = np.zeros((len(uniq), pred.shape[1]))
        np.add.at(agg, inv, pred)
        agg = agg / counts[:, None]
    out_labels = [np.asarray(lab)[first_idx] for lab in labels]
    return (uniq, agg, *out_labels)


def grade_metrics(probs, targets, n_classes=3):
    """
    Grade classification metrics for one set of predictions.

    Args:
        probs: (N, n_classes) class probabilities
        targets: (N,) integer labels (0 = Grade II, 1 = III, 2 = IV)

    Returns:
        dict with accuracy, auc, ap, f1_micro, f1_gradeIV
        (auc / ap micro-averaged over one-hot labels, as in Chen et al.)
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets).astype(int)
    classes = np.arange(n_classes)
    onehot = (targets[:, None] == classes[None, :]).astype(int)
    pred_cls = probs.argmax(axis=1)

    out = {'accuracy': float((pred_cls == targets).mean())}
    try:
        out['auc'] = float(roc_auc_score(onehot, probs, average='micro'))
        out['ap'] = float(
            average_precision_score(onehot, probs, average='micro')
        )
    except ValueError:
        out['auc'], out['ap'] = float('nan'), float('nan')
    out['f1_micro'] = float(f1_score(targets, pred_cls, average='micro'))
    per_class = f1_score(targets, pred_cls, average=None, labels=classes,
                         zero_division=0)
    out['f1_gradeIV'] = float(per_class[2]) if n_classes > 2 else float('nan')
    return out


def calcGradMetrics(predictions_list, avg='micro'):
    """
    Grade metrics across CV folds.

    Args:
        predictions_list: list of (probs, targets) tuples, one per fold

    Returns:
        dict with mean ± CI strings and the raw per-fold lists
    """
    per_fold = [grade_metrics(p, t) for p, t in predictions_list]
    out = {}
    for key in ('auc', 'ap', 'f1_micro', 'f1_gradeIV', 'accuracy'):
        vals = [m[key] for m in per_fold]
        out[key] = CI_pm(vals) if len(vals) > 1 else f"{vals[0]:.4f}"
        out[key + '_raw'] = vals
    return out
