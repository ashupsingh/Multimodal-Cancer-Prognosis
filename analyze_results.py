"""
Analysis and Reporting Pipeline for Pathomic Fusion.

Everything here is computed from the files written by `train.py` /
`run_cox_baselines.py` (per-fold summaries and saved test predictions):
1. Survival and grade comparison tables (mean ± standard error over folds).
   The numbers reported by Chen et al. (2022) are listed in a separate
   column for reference only.
2. Late-fusion ensemble of the unimodal models, scored from their saved
   out-of-sample predictions.
3. Kaplan-Meier risk stratification from out-of-sample patient predictions.
4. Modality comparison bar plot.

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from utils.metrics import (
    CIndex_lifeline, aggregate_by_patient, grade_metrics
)

RESULTS_DIR = './results'
CHECKPOINTS_DIR = './checkpoints/TCGA_GBMLGG'
SURV_EXP, GRAD_EXP = 'surv_15_rnaseq', 'grad_15'

# (mode, display name, modality group)
MODELS = [
    ('path', 'Histology CNN (VGG19-BN embeddings)', 'Unimodal'),
    ('graph', 'Histology GCN (cell graph)', 'Unimodal'),
    ('omic', 'Genomic SNN', 'Unimodal'),
    ('pathomic', 'CNN ⊗ SNN (Pathomic Fusion)', 'Bimodal'),
    ('graphomic', 'GCN ⊗ SNN (Graphomic Fusion)', 'Bimodal'),
    ('pathgraphomic', 'CNN ⊗ GCN ⊗ SNN (Trimodal Fusion)', 'Trimodal'),
]

# Numbers published in Chen et al. (2022), Tables I and V (TCGA-GBMLGG).
# Reference only: they are not results of this project, and were obtained
# with the authors' own training runs and per-ROI scoring.
PAPER_REPORTED = {
    'surv': {
        'cox_grade': 0.738, 'cox_moltype': 0.760, 'cox_grade_moltype': 0.777,
        'path': 0.792, 'graph': 0.746, 'omic': 0.808,
        'pathomic': 0.820, 'graphomic': 0.812, 'pathgraphomic': 0.826,
    },
    'grad': {
        'path': {'auc': 0.883, 'ap': 0.793, 'f1_micro': 0.717, 'f1_gradeIV': 0.873},
        'graph': {'auc': 0.849, 'ap': 0.764, 'f1_micro': 0.665, 'f1_gradeIV': 0.849},
        'omic': {'auc': 0.853, 'ap': 0.729, 'f1_micro': 0.652, 'f1_gradeIV': 0.857},
        'pathomic': {'auc': 0.905, 'ap': 0.833, 'f1_micro': 0.730, 'f1_gradeIV': 0.913},
        'graphomic': {'auc': 0.897, 'ap': 0.812, 'f1_micro': 0.714, 'f1_gradeIV': 0.902},
        'pathgraphomic': {'auc': 0.908, 'ap': 0.828, 'f1_micro': 0.749, 'f1_gradeIV': 0.920},
    },
}


# ── Helpers ───────────────────────────────────────────────────────────

def mean_se(values):
    vals = pd.Series(values).dropna()
    if len(vals) == 0:
        return float('nan'), float('nan')
    se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return float(vals.mean()), float(se)


def fmt(values):
    m, se = mean_se(values)
    return '—' if np.isnan(m) else f"{m:.3f} ± {se:.3f}"


def summary_path(exp, mode, task):
    return os.path.join(RESULTS_DIR, f"{exp}_{mode}_{mode}_{task}_summary.csv")


def load_fold_predictions(exp, mode):
    """{fold: prediction dict} from the saved test predictions of a model."""
    out = {}
    pattern = os.path.join(CHECKPOINTS_DIR, exp, mode, f"{mode}_*_pred_test.pkl")
    for path in glob.glob(pattern):
        fold = int(os.path.basename(path).split('_')[-3])
        with open(path, 'rb') as f:
            out[fold] = pickle.load(f)
    return out


def patient_level(pred_dict, key):
    """Patient-level predictions and labels of one fold as a DataFrame."""
    names, pred, censor, time, grade = aggregate_by_patient(
        pred_dict['patname'], pred_dict[key], pred_dict['censor'],
        pred_dict['time'], pred_dict['grade'])
    if pred.ndim == 1:
        df = pd.DataFrame({'pred': pred}, index=names)
    else:
        df = pd.DataFrame(pred, index=names,
                          columns=[f'p{i}' for i in range(pred.shape[1])])
    df['censor'], df['time'], df['grade'] = censor, time, grade
    return df


# ── Late fusion of unimodal models ────────────────────────────────────

def late_fusion(exp, task, modes):
    """
    Late-fusion ensemble: per patient, the mean of the unimodal models'
    predictions (risk scores for survival, class probabilities for grade).
    No weights are fitted and no test statistics are used.

    Returns:
        DataFrame with one row of metrics per fold, or None
    """
    key = 'risk' if task == 'surv' else 'probs'
    # only models whose cross-validation run has completed
    if not all(os.path.exists(summary_path(exp, m, task)) for m in modes):
        return None
    preds = {m: load_fold_predictions(exp, m) for m in modes}
    if any(len(p) == 0 for p in preds.values()):
        return None
    folds = sorted(set.intersection(*(set(p) for p in preds.values())))
    rows = []
    for k in folds:
        frames = [patient_level(preds[m][k], key) for m in modes]
        common = frames[0].index
        for fr in frames[1:]:
            common = common.intersection(fr.index)
        if task == 'surv':
            risk = np.mean([fr.loc[common, 'pred'].values for fr in frames], axis=0)
            ref = frames[0].loc[common]
            rows.append({'fold': k, 'cindex_patient': CIndex_lifeline(
                risk, ref['censor'].values, ref['time'].values)})
        else:
            cols = [c for c in frames[0].columns if c.startswith('p')]
            probs = np.mean([fr.loc[common, cols].values for fr in frames], axis=0)
            m = grade_metrics(probs, frames[0].loc[common, 'grade'].values)
            rows.append({'fold': k, **{f'{a}_patient': b for a, b in m.items()}})
    return pd.DataFrame(rows)


# ── Tables ────────────────────────────────────────────────────────────

def build_survival_table():
    rows = []
    cox_csv = os.path.join(RESULTS_DIR, 'cox_baselines', 'cox_baselines_summary.csv')
    if os.path.exists(cox_csv):
        for _, r in pd.read_csv(cox_csv).iterrows():
            rows.append({
                'Category': 'Cox baseline', 'Model': r['Model'],
                'c-Index per patient': f"{r['Mean C-Index']:.3f} ± {r['SE C-Index']:.3f}",
                'c-Index per ROI': '—', 'Folds': int(r['Num Folds']),
                'Paper (reported)': PAPER_REPORTED['surv'].get(r['Key'], np.nan),
                '_mean': r['Mean C-Index'],
            })
    for mode, name, group in MODELS:
        path = summary_path(SURV_EXP, mode, 'surv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        rows.append({
            'Category': f'{group} deep model', 'Model': name,
            'c-Index per patient': fmt(df['cindex_patient']),
            'c-Index per ROI': fmt(df['cindex']), 'Folds': len(df),
            'Paper (reported)': PAPER_REPORTED['surv'].get(mode, np.nan),
            '_mean': mean_se(df['cindex_patient'])[0],
        })
    for modes, name in [(('path', 'omic'), 'Late fusion: CNN + SNN'),
                        (('path', 'graph', 'omic'), 'Late fusion: CNN + GCN + SNN')]:
        lf = late_fusion(SURV_EXP, 'surv', modes)
        if lf is not None:
            lf.to_csv(os.path.join(
                RESULTS_DIR, f"{SURV_EXP}_latefusion_{'_'.join(modes)}_surv_summary.csv"),
                index=False)
            rows.append({
                'Category': 'Late-fusion ensemble', 'Model': name,
                'c-Index per patient': fmt(lf['cindex_patient']),
                'c-Index per ROI': '—', 'Folds': len(lf),
                'Paper (reported)': np.nan,
                '_mean': mean_se(lf['cindex_patient'])[0],
            })
    return pd.DataFrame(rows)


def build_grade_table():
    rows = []
    metrics = [('auc', 'AUC'), ('ap', 'AP'), ('f1_micro', 'Micro-F1'),
               ('f1_gradeIV', 'Grade IV F1')]
    for mode, name, group in MODELS:
        path = summary_path(GRAD_EXP, mode, 'grad')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        row = {'Model': name, 'Folds': len(df)}
        for key, label in metrics:
            row[f'{label} per patient'] = fmt(df[f'{key}_patient'])
        for key, label in metrics:
            row[f'{label} per ROI'] = fmt(df[key])
        paper = PAPER_REPORTED['grad'].get(mode, {})
        row['Paper AUC (reported)'] = paper.get('auc', np.nan)
        row['_mean'] = mean_se(df['auc_patient'])[0]
        rows.append(row)
    lf = late_fusion(GRAD_EXP, 'grad', ('path', 'omic'))
    if lf is not None:
        row = {'Model': 'Late fusion: CNN + SNN', 'Folds': len(lf)}
        for key, label in metrics:
            row[f'{label} per patient'] = fmt(lf[f'{key}_patient'])
            row[f'{label} per ROI'] = '—'
        row['Paper AUC (reported)'] = np.nan
        row['_mean'] = mean_se(lf['auc_patient'])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison_table(output_dir=RESULTS_DIR):
    """Write the survival and grade comparison tables (CSV) and print them."""
    os.makedirs(output_dir, exist_ok=True)
    df_surv, df_grad = build_survival_table(), build_grade_table()

    if len(df_surv):
        df_surv.drop(columns='_mean').to_csv(
            os.path.join(output_dir, 'survival_comparison_table.csv'), index=False)
        print("\n" + "=" * 100)
        print("  Survival outcome prediction — c-Index, mean ± SE over folds")
        print("=" * 100)
        print(df_surv.drop(columns='_mean').to_string(index=False))
    if len(df_grad):
        df_grad.drop(columns='_mean').to_csv(
            os.path.join(output_dir, 'grade_classification_comparison_table.csv'),
            index=False)
        print("\n" + "=" * 100)
        print("  Grade classification — mean ± SE over folds")
        print("=" * 100)
        print(df_grad.drop(columns='_mean').to_string(index=False))
    return df_surv, df_grad


# ── Kaplan-Meier ──────────────────────────────────────────────────────

def out_of_sample_patient_risk(exp, mode):
    """
    One out-of-sample risk per patient: within every test fold the patient
    risks are converted to percentile ranks, which are then averaged over
    the folds in which the patient was a test patient.
    """
    frames = []
    for k, pred in load_fold_predictions(exp, mode).items():
        df = patient_level(pred, 'risk')
        df['rank'] = df['pred'].rank(pct=True)
        frames.append(df[['rank', 'censor', 'time']])
    if not frames:
        return None
    allf = pd.concat(frames)
    return allf.groupby(level=0).agg(
        risk=('rank', 'mean'), censor=('censor', 'first'), time=('time', 'first'))


def plot_kaplan_meier(risk_scores, censors, survtimes,
                      model_name='Pathomic Fusion',
                      save_path='./results/kaplan_meier_curve.png'):
    """
    Kaplan-Meier curves of the high / low risk halves (median split of the
    predicted risk) with a two-sided log-rank test.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    censors = np.asarray(censors).reshape(-1).astype(int)
    survtimes = np.asarray(survtimes, dtype=float).reshape(-1)

    valid = survtimes > 0
    risk_scores, censors, survtimes = risk_scores[valid], censors[valid], survtimes[valid]

    ranks = np.argsort(risk_scores)
    high = np.zeros(len(risk_scores), dtype=bool)
    high[ranks[len(risk_scores) // 2:]] = True
    low = ~high

    p_val = logrank_test(survtimes[high], survtimes[low],
                         event_observed_A=censors[high],
                         event_observed_B=censors[low]).p_value

    plt.figure(figsize=(9, 7), dpi=200)
    kmf_low, kmf_high = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_low.fit(survtimes[low], event_observed=censors[low],
                label=f'Low risk (N={low.sum()})')
    ax = kmf_low.plot_survival_function(color='#2980b9', linewidth=2.5, ci_show=True)
    kmf_high.fit(survtimes[high], event_observed=censors[high],
                 label=f'High risk (N={high.sum()})')
    kmf_high.plot_survival_function(ax=ax, color='#e74c3c', linewidth=2.5, ci_show=True)

    plt.title(f'Survival risk stratification — {model_name}',
              fontsize=13, fontweight='bold', pad=15)
    # the dataset column is called 'Survival months' but holds days
    plt.xlabel('Time (days)', fontsize=12, fontweight='bold')
    plt.ylabel('Overall survival probability', fontsize=12, fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0, max(survtimes) * 1.02])
    p_str = f"Log-rank p = {p_val:.1e}" if p_val < 0.001 else f"Log-rank p = {p_val:.4f}"
    plt.text(0.05, 0.15, p_str, transform=ax.transAxes, fontsize=12,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#95a5a6', alpha=0.9))
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Kaplan-Meier plot saved to: {save_path} (log-rank p = {p_val:.2e})")
    return p_val


# ── Modality comparison plot ──────────────────────────────────────────

def plot_modality_ablation(df_surv,
                           save_path='./results/modality_ablation_cindex.png'):
    """
    Bar chart of the patient-level c-Index (mean ± SE over folds) of every
    model trained in this project, with the strongest Cox baseline as a
    reference line.
    """
    deep = df_surv[df_surv['Category'].str.contains('deep model|Late-fusion')]
    if deep.empty:
        return
    palette = {'Unimodal deep model': '#95a5a6', 'Bimodal deep model': '#3498db',
               'Trimodal deep model': '#2ecc71', 'Late-fusion ensemble': '#f39c12'}
    labels, means, ses, colors = [], [], [], []
    for _, r in deep.iterrows():
        mean, se = r['c-Index per patient'].split(' ± ')
        labels.append(r['Model'].replace('⊗', '×')
                      .replace(' (', '\n(').replace(': ', ':\n'))
        means.append(float(mean))
        ses.append(float(se))
        colors.append(palette.get(r['Category'], '#95a5a6'))

    plt.figure(figsize=(max(9, 1.9 * len(labels)), 6), dpi=200)
    bars = plt.bar(labels, means, yerr=ses, capsize=6, color=colors,
                   edgecolor='black', alpha=0.9, width=0.6)
    cox = df_surv[df_surv['Category'] == 'Cox baseline']
    if len(cox):
        best = cox.loc[cox['_mean'].idxmax()]
        plt.axhline(best['_mean'], color='#c0392b', linestyle='--', linewidth=1.5,
                    label=f"Best Cox baseline: {best['Model']} ({best['_mean']:.3f})")
        plt.legend(loc='upper left', frameon=True, fontsize=10)
    lo = min(means) - 0.05
    plt.ylim([max(0.5, lo), max(means) + 0.04])
    plt.ylabel('c-Index per patient (mean ± SE, 15 folds)', fontsize=12,
               fontweight='bold')
    plt.title('Survival prediction by modality — TCGA-GBMLGG',
              fontsize=13, fontweight='bold', pad=15)
    for bar, val, err in zip(bars, means, ses):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + err + 0.003,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Modality comparison plot saved to: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────

def run_full_analysis():
    """Build every table and figure from the results on disk."""
    df_surv, _ = build_comparison_table()
    if len(df_surv):
        plot_modality_ablation(df_surv)

    # Kaplan-Meier curve of the most complete fusion model that was trained
    for mode, name, _ in reversed(MODELS):
        if mode not in ('pathomic', 'graphomic', 'pathgraphomic'):
            continue
        risk = out_of_sample_patient_risk(SURV_EXP, mode)
        if risk is not None:
            plot_kaplan_meier(
                risk['risk'].values, risk['censor'].values, risk['time'].values,
                model_name=f"{name.replace('⊗', '×')}\nout-of-sample "
                           f"predictions, {len(risk)} patients",
                save_path=os.path.join(RESULTS_DIR, 'kaplan_meier_curve.png'))
            break


if __name__ == '__main__':
    run_full_analysis()
