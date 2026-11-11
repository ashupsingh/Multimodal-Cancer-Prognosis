# Baseline snapshot (before the optimisation work of 2026-11-11)

Exact copy of `results/` and `checkpoints/TCGA_GBMLGG/surv_15_rnaseq/` as they were at git commit `087997f`.

- Only fold 1 of 15 was trained, for 5 epochs.
- Reported c-index (per ROI row): omic 0.8032, pathomic 0.8489.
- Re-scored per patient (100 test patients): omic 0.787, pathomic 0.828.
- The grade table, modality-ablation plot, KM curve and explainability outputs in this
  snapshot were placeholders (paper numbers / untrained network / random inputs), not model results.
- The checkpoints here only load with the code at commit `087997f`.
