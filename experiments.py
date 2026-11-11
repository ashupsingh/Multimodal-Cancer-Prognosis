"""
Experiment definitions for the TCGA-GBMLGG benchmark.

Single source of truth for:
- the base arguments of every model (per task),
- the hyper-parameter grids searched by `tune.py`,
- the tuned settings written by `tune.py` and consumed by `run_all.py`.

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import json

TUNING_DIR = './results/tuning'

# Arguments shared by every model of a task
TASK_ARGS = {
    'surv': ['--task', 'surv', '--use_rnaseq', '1'],
    'grad': ['--task', 'grad'],
}

# Base arguments per mode (model_name == mode)
MODE_ARGS = {
    'omic': ['--reg_type', 'omic', '--batch_size', '64'],
    'path': ['--use_vgg_features', '1', '--reg_type', 'none',
             '--batch_size', '64', '--lr', '1e-3', '--niter_decay', '20'],
    # GCN settings as documented by the authors (dataset README)
    'graph': ['--reg_type', 'none', '--batch_size', '32', '--lr', '2e-3',
              '--niter_decay', '50'],
    'pathomic': ['--use_vgg_features', '1', '--reg_type', 'none',
                 '--batch_size', '64', '--niter_decay', '30',
                 '--fusion_type', 'pofusion', '--mmhid', '64'],
    'graphomic': ['--reg_type', 'none',
                  '--batch_size', '32', '--niter_decay', '30',
                  '--fusion_type', 'pofusion', '--mmhid', '64'],
    'pathgraphomic': ['--use_vgg_features', '1', '--reg_type', 'none',
                      '--batch_size', '32', '--niter_decay', '30',
                      '--mmhid', '64'],
}

# Fusion architecture options that differ by task, as documented by the
# authors (dataset README): which modality is left un-gated, which embedding
# is halved before the Kronecker product, and the trilinear gating variant.
TASK_MODE_ARGS = {
    ('surv', 'graphomic'): ['--omic_gate', '0', '--grph_scale', '2'],
    ('grad', 'graphomic'): ['--grph_gate', '0', '--omic_scale', '2'],
    ('surv', 'pathgraphomic'): ['--fusion_type', 'pofusion_A',
                                '--omic_gate', '0', '--grph_scale', '2'],
    ('grad', 'pathgraphomic'): ['--fusion_type', 'pofusion_B',
                                '--path_gate', '0'],
}

# Hyper-parameter grids (searched on inner validation splits only)
GRIDS = {
    'omic': {
        'lr': [2e-3, 5e-4],
        'niter_decay': [30, 60],
        'dropout_rate': [0.25, 0.5],
        'lambda_reg': [3e-4, 1e-3],
        'batch_size': [32, 64],
    },
    'pathomic': {
        'lr': [1e-4, 5e-4, 2e-3],
        'unfreeze_epoch': [5, 0],
        'omic_gate': [0, 1],
    },
}

# Graph-based fusion models are not tuned separately (a grid over them costs
# hours); they re-use these settings selected for the pathomic fusion model.
INHERIT_FROM_PATHOMIC = ('lr', 'unfreeze_epoch')

GRAPH_MODES = ('graph', 'graphomic', 'pathgraphomic')
FUSION_MODES = ('pathomic', 'graphomic', 'pathgraphomic')


def overrides_to_argv(overrides):
    """{'lr': 0.001} -> ['--lr', '0.001']"""
    argv = []
    for key, val in overrides.items():
        argv += [f'--{key}', str(val)]
    return argv


def best_config_path(task, mode):
    return os.path.join(TUNING_DIR, f'best_{task}_{mode}.json')


def load_best_overrides(task, mode):
    """Tuned hyper-parameters of a model ({} when it was not tuned)."""
    path = best_config_path(task, mode)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)['overrides']
    if mode in ('graphomic', 'pathgraphomic'):
        inherited = load_best_overrides(task, 'pathomic')
        return {k: v for k, v in inherited.items()
                if k in INHERIT_FROM_PATHOMIC}
    return {}


def experiment_argv(task, mode, extra=None, tuned=True):
    """Full argument list of an experiment."""
    argv = (['--mode', mode, '--model_name', mode]
            + TASK_ARGS[task] + MODE_ARGS[mode]
            + TASK_MODE_ARGS.get((task, mode), []))
    if tuned:
        argv += overrides_to_argv(load_best_overrides(task, mode))
    return argv + list(extra or [])
