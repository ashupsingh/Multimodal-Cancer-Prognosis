"""
Run the complete TCGA-GBMLGG benchmark.

Trains every model that the data on disk allows, for both tasks, over all
15 cross-validation folds, using the settings in `experiments.py` and the
hyper-parameters selected by `tune.py`:

    genomic SNN -> histology -> pathomic fusion
    (+ cell-graph GCN -> graphomic -> trimodal fusion when the graph files
     are present under <dataroot>/all_st_cpc; fetch them with
     `python download_data.py --target gbmlgg_graphs`)

Unimodal networks are trained first because the fusion networks load them.

Usage:
    python run_all.py                       # everything
    python run_all.py --tasks surv --modes omic,pathomic --n_seeds 5
"""

import os
import sys
import glob
import time
import argparse

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import parse_args
from experiments import experiment_argv, GRAPH_MODES
from train import run_cross_validation

MODE_ORDER = ['omic', 'path', 'graph', 'pathomic', 'graphomic',
              'pathgraphomic']


def graph_data_available(dataroot='./data/TCGA_GBMLGG'):
    """True when the ROI cell-graph files are on disk."""
    return len(glob.glob(os.path.join(dataroot, 'all_st_cpc', '*.pt'))) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='surv,grad')
    parser.add_argument('--modes', default='all')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Ensemble members per fold')
    parser.add_argument('--graph_n_seeds', type=int, default=3,
                        help='Ensemble members per fold for graph models')
    parser.add_argument('--folds', default='all')
    parser.add_argument('--gpu_ids', default='0',
                        help="Device ('-1' = CPU)")
    parser.add_argument('--cv_workers', type=int, default=1,
                        help='Folds trained in parallel (feature-based models)')
    parser.add_argument('--graph_cv_workers', type=int, default=1,
                        help='Folds trained in parallel (graph models, on GPU; '
                             'every worker keeps its own graph cache in RAM)')
    args = parser.parse_args()

    modes = MODE_ORDER if args.modes == 'all' else [
        m for m in MODE_ORDER if m in args.modes.split(',')]
    has_graphs = graph_data_available()

    for task in args.tasks.split(','):
        for mode in modes:
            if mode in GRAPH_MODES and not has_graphs:
                print(f"[skip] {task}/{mode}: cell-graph files not found.")
                continue
            n_seeds = args.graph_n_seeds if mode in GRAPH_MODES else args.n_seeds
            cv_workers = (args.graph_cv_workers if mode in GRAPH_MODES
                          else args.cv_workers)
            gpu_ids = '0' if mode in GRAPH_MODES else args.gpu_ids
            opt = parse_args(experiment_argv(
                task, mode,
                extra=['--n_seeds', str(n_seeds), '--folds', args.folds,
                       '--cv_workers', str(cv_workers),
                       '--gpu_ids', gpu_ids]))
            start = time.time()
            run_cross_validation(opt)
            print(f"[done] {task}/{mode} in {time.time() - start:.0f}s\n")


if __name__ == '__main__':
    main()
