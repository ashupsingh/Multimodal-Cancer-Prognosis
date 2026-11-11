"""
Cox Proportional Hazards Baselines for Pathomic Fusion.

Implements standard clinical and statistical baselines across 15 folds:
1. Genomic Omic Only (CNV + Mutation)
2. Molecular Subtype Only (IDH mutation + 1p/19q codeletion)
3. Histological Subtype Only (astrocytoma, oligodendroglioma, glioblastoma)
4. Histologic Grade Only (WHO Grade II, III, IV)
5. Grade + Molecular Subtype
6. Multimodal Clinical + Genomic (All available features)

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import pickle
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from utils.metrics import CI_pm, CI_interval


def run_cox_baselines(split_file='./data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_1_0_0_rnaseq.pkl',
                      output_dir='./results/cox_baselines'):
    """Run all 6 Cox PH baseline models across 15 cross-validation folds."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(split_file):
        # Find any split file in data/TCGA_GBMLGG/splits
        splits_dir = os.path.dirname(split_file)
        pkls = [f for f in os.listdir(splits_dir) if f.endswith('.pkl')]
        if not pkls:
            raise FileNotFoundError(f"No split files found in {splits_dir}")
        split_file = os.path.join(splits_dir, pkls[0])

    print(f"Loading data from split file: {split_file}")
    with open(split_file, 'rb') as f:
        data_dict = pickle.load(f)

    all_dataset = data_dict['data_pd'].copy()
    cv_splits = data_dict['cv_splits']

    # Ensure censored is 1=event
    if 'Survival months' not in all_dataset.columns and 't' in all_dataset.columns:
        all_dataset['Survival months'] = all_dataset['t']

    model_configs = {
        'cox_grade': {
            'features': ['Grade'],
            'penalizer': 0.0,
            'desc': 'Histologic Grade Only'
        },
        'cox_moltype': {
            'features': [c for c in ['codeletion', 'idh mutation'] if c in all_dataset.columns],
            'penalizer': 0.0,
            'desc': 'Molecular Subtype Only (IDH + codeletion)'
        },
        'cox_histype': {
            'features': [c for c in ['Histology'] if c in all_dataset.columns],
            'penalizer': 0.0,
            'desc': 'Histological Subtype Only'
        },
        'cox_grade_moltype': {
            'features': [c for c in ['Grade', 'codeletion', 'idh mutation'] if c in all_dataset.columns],
            'penalizer': 0.0,
            'desc': 'Grade + Molecular Subtype'
        },
        'cox_omic': {
            'features': 'omic_only',
            'penalizer': 0.1,
            'desc': 'Genomic Features (CNV + Mutation)'
        },
        'cox_clinical_omic': {
            'features': 'all_feats',
            'penalizer': 0.1,
            'desc': 'Clinical + Genomic Features'
        }
    }

    metadata_cols = [
        'Histomolecular subtype', 'Histology', 'Grade', 'Molecular subtype',
        'TCGA ID', 'censored', 'Survival months'
    ]
    omic_cols = [c for c in all_dataset.columns if c not in metadata_cols]

    baseline_summary = []
    per_fold_rows = []

    print("\n" + "=" * 70)
    print(" Running Cox Proportional Hazards Baselines (15 Folds)")
    print("=" * 70)

    for model_name, cfg in model_configs.items():
        print(f"\nEvaluating Baseline: {cfg['desc']} ({model_name})...")
        penalizer = cfg['penalizer']

        if cfg['features'] == 'omic_only':
            selected_feats = omic_cols[:80]  # First 80 CNV/mutation
        elif cfg['features'] == 'all_feats':
            selected_feats = [c for c in ['Grade', 'Histology', 'Molecular subtype'] if c in all_dataset.columns] + omic_cols[:80]
        else:
            selected_feats = cfg['features']

        fold_cindices = []

        for fold_k, fold_data in cv_splits.items():
            tr_names = fold_data['train']['x_patname']
            te_names = fold_data['test']['x_patname']

            # Unique patient IDs
            tr_pats = list(set(tr_names).intersection(all_dataset.index))
            te_pats = list(set(te_names).intersection(all_dataset.index))

            if not tr_pats or not te_pats:
                # Fallback to index-based split if patnames not aligned
                n = len(all_dataset)
                fold_idx = fold_k - 1
                test_start = (fold_idx * n) // 15
                test_end = ((fold_idx + 1) * n) // 15
                te_pats = all_dataset.index[test_start:test_end].tolist()
                tr_pats = [p for p in all_dataset.index if p not in te_pats]

            fit_cols = selected_feats + ['Survival months', 'censored']
            train_df = all_dataset.loc[tr_pats, fit_cols].dropna()
            test_df = all_dataset.loc[te_pats, fit_cols].dropna()

            # Ensure numeric
            train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Drop non-varying columns in train
            var_cols = [c for c in selected_feats if train_df[c].nunique() > 1]
            if not var_cols:
                continue

            try:
                cph = CoxPHFitter(penalizer=penalizer)
                cph.fit(train_df[var_cols + ['Survival months', 'censored']],
                        duration_col='Survival months', event_col='censored',
                        show_progress=False)

                risk_scores = cph.predict_partial_hazard(test_df[var_cols])
                # Lower survival time = higher risk, so we negate risk for concordance_index
                cin = concordance_index(
                    test_df['Survival months'].values,
                    -risk_scores.values.squeeze(),
                    test_df['censored'].values
                )
                fold_cindices.append(cin)
                per_fold_rows.append({'Key': model_name, 'fold': fold_k,
                                      'cindex_patient': cin})
            except Exception as e:
                print(f"  [Warning] fold {fold_k} failed for {model_name}: {e}")

        if fold_cindices:
            mean_c = np.mean(fold_cindices)
            std_c = np.std(fold_cindices)
            se_c = std_c / np.sqrt(len(fold_cindices)) if len(fold_cindices) > 1 else 0
            ci_low, ci_high = CI_interval(fold_cindices) if len(fold_cindices) > 1 else (mean_c, mean_c)
            print(f"  {cfg['desc']:<35}: C-Index = {mean_c:.4f} ± {std_c:.4f} (SE = {se_c:.4f}) [95% CI: {ci_low:.4f} - {ci_high:.4f}]")
            baseline_summary.append({
                'Model': cfg['desc'],
                'Key': model_name,
                'Mean C-Index': mean_c,
                'Std C-Index': std_c,
                'SE C-Index': se_c,
                '95% CI Low': ci_low,
                '95% CI High': ci_high,
                'Num Folds': len(fold_cindices)
            })

    pd.DataFrame(per_fold_rows).to_csv(
        os.path.join(output_dir, 'cox_baselines_per_fold.csv'), index=False)
    df_summary = pd.DataFrame(baseline_summary)
    out_csv = os.path.join(output_dir, 'cox_baselines_summary.csv')
    df_summary.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"Saved Cox baselines summary to: {out_csv}")
    print("=" * 70 + "\n")
    return df_summary


if __name__ == '__main__':
    run_cox_baselines()
