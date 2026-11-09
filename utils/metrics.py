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
from sklearn.preprocessing import LabelBinarizer


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


def calcGradMetrics(predictions_list, avg='micro'):
    """
    Calculate grade classification metrics across all CV folds.
    
    Args:
        predictions_list: List of (grade_pred, grade_true) tuples per fold
        avg: Averaging method for F1 score
        
    Returns:
        dict with 'auc', 'ap', 'f1', 'f1_gradeIV' as CI_pm strings
    """
    auc_all, ap_all, f1_all, f1_gradeIV_all = [], [], [], []

    for grade_pred, grade in predictions_list:
        grade_pred = np.array(grade_pred)
        grade = np.array(grade)

        enc = LabelBinarizer()
        enc.fit(grade)
        grade_oh = enc.transform(grade)

        rocauc = roc_auc_score(grade_oh, grade_pred, average=avg)
        ap = average_precision_score(grade_oh, grade_pred, average=avg)
        f1 = f1_score(grade_pred.argmax(axis=1), grade, average=avg)
        f1_gradeIV = f1_score(
            grade_pred.argmax(axis=1), grade, average=None
        )
        # Grade IV is index 2 (Grade II=0, III=1, IV=2)
        f1_iv = f1_gradeIV[2] if len(f1_gradeIV) > 2 else 0.0

        auc_all.append(rocauc)
        ap_all.append(ap)
        f1_all.append(f1)
        f1_gradeIV_all.append(f1_iv)

    return {
        'auc': CI_pm(auc_all),
        'ap': CI_pm(ap_all),
        'f1': CI_pm(f1_all),
        'f1_gradeIV': CI_pm(f1_gradeIV_all),
        'auc_raw': auc_all,
        'ap_raw': ap_all,
        'f1_raw': f1_all,
        'f1_gradeIV_raw': f1_gradeIV_all,
    }
