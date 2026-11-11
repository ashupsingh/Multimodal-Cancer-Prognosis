"""
Central Configuration for Pathomic Fusion
Multimodal Cancer Prognosis using Histopathology Images and Genomic Features

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import argparse
import os
import torch


def parse_args(argv=None):
    """Parse command line arguments for training and evaluation.

    Args:
        argv: optional list of arguments (defaults to sys.argv). Unknown
              arguments raise an error instead of being silently ignored.
    """
    parser = argparse.ArgumentParser(
        description='Pathomic Fusion: Multimodal Cancer Prognosis'
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--dataroot', type=str,
                        default='./data/TCGA_GBMLGG',
                        help='Root directory of the dataset')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='./checkpoints/TCGA_GBMLGG',
                        help='Directory to save model checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (checkpoint subdirectory). '
                             'Default: surv_15_rnaseq (surv) or grad_15 (grad)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save evaluation results')

    # ── Hardware ──────────────────────────────────────────────────────────
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs (e.g. 0, 0,1,2). Use -1 for CPU')

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument('--mode', type=str, default='omic',
                        choices=['path', 'graph', 'omic',
                                 'pathomic', 'graphomic', 'pathgraphomic',
                                 'pathpath', 'graphgraph', 'omicomic'],
                        help='Modality mode')
    parser.add_argument('--model_name', type=str, default='omic',
                        help='Name of the model (used for checkpoint naming)')
    parser.add_argument('--use_vgg_features', type=int, default=0,
                        help='If 1, use pre-extracted VGG19 features instead of raw images')
    parser.add_argument('--use_rnaseq', type=int, default=0,
                        help='If 1, include RNA-Seq features in genomic data')

    # ── Task ──────────────────────────────────────────────────────────────
    parser.add_argument('--task', type=str, default='surv',
                        choices=['surv', 'grad'],
                        help='Task: surv=survival prediction, grad=grade classification')
    parser.add_argument('--act_type', type=str, default=None,
                        choices=['Sigmoid', 'Tanh', 'ReLU', 'LSM', 'none'],
                        help='Output activation (default: Sigmoid for surv, LSM for grad)')
    parser.add_argument('--label_dim', type=int, default=None,
                        help='Output dimension (1 for survival, 3 for grade)')

    # ── Dimensions ────────────────────────────────────────────────────────
    parser.add_argument('--input_size_omic', type=int, default=80,
                        help='Genomic input feature dimension '
                             '(set from the split file at train time)')
    parser.add_argument('--input_size_path', type=int, default=512,
                        help='Histopathology image crop size')
    parser.add_argument('--path_dim', type=int, default=32,
                        help='Path network embedding dimension')
    parser.add_argument('--grph_dim', type=int, default=32,
                        help='Graph network embedding dimension')
    parser.add_argument('--omic_dim', type=int, default=32,
                        help='Omic network embedding dimension')

    # ── Fusion ────────────────────────────────────────────────────────────
    parser.add_argument('--fusion_type', type=str, default='pofusion',
                        help='Fusion method: pofusion, pofusion_A, pofusion_B')
    parser.add_argument('--skip', type=int, default=0,
                        help='Use skip connections in fusion')
    parser.add_argument('--use_bilinear', type=int, default=1,
                        help='Use bilinear gating in fusion')
    parser.add_argument('--path_gate', type=int, default=1,
                        help='Enable gating for path modality')
    parser.add_argument('--grph_gate', type=int, default=1,
                        help='Enable gating for graph modality')
    parser.add_argument('--omic_gate', type=int, default=1,
                        help='Enable gating for omic modality')
    parser.add_argument('--path_scale', type=int, default=1,
                        help='Dimension scaling for path in fusion')
    parser.add_argument('--grph_scale', type=int, default=1,
                        help='Dimension scaling for graph in fusion')
    parser.add_argument('--omic_scale', type=int, default=1,
                        help='Dimension scaling for omic in fusion')
    parser.add_argument('--mmhid', type=int, default=64,
                        help='Multimodal hidden dimension')

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument('--optimizer_type', type=str, default='adam',
                        choices=['adam', 'adagrad'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--weight_decay', type=float, default=4e-4,
                        help='L2 weight decay')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        choices=['linear', 'exp', 'step', 'plateau', 'cosine'],
                        help='Learning rate schedule policy')
    parser.add_argument('--niter', type=int, default=0,
                        help='Number of epochs at starting LR')
    parser.add_argument('--niter_decay', type=int, default=25,
                        help='Number of epochs to linearly decay LR to zero')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='Starting epoch count')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (0 = full batch)')
    parser.add_argument('--num_splits', type=int, default=15,
                        help='Number of cross-validation folds')

    # ── Regularization ────────────────────────────────────────────────────
    parser.add_argument('--reg_type', type=str, default='omic',
                        choices=['none', 'path', 'mm', 'all', 'omic'],
                        help='Regularization type')
    parser.add_argument('--lambda_cox', type=float, default=1.0,
                        help='Weight for Cox loss')
    parser.add_argument('--lambda_reg', type=float, default=3e-4,
                        help='Weight for regularization loss')
    parser.add_argument('--lambda_nll', type=float, default=1.0,
                        help='Weight for NLL loss (grade classification)')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                        help='Dropout rate')

    # ── Initialization ────────────────────────────────────────────────────
    parser.add_argument('--init_type', type=str, default='none',
                        choices=['normal', 'xavier', 'kaiming', 'orthogonal',
                                 'max', 'none'],
                        help='Network weight initialization method')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='Scaling factor for initialization')

    # ── Graph ─────────────────────────────────────────────────────────────
    parser.add_argument('--use_edges', type=int, default=1,
                        help='Use edge attributes in GCN')
    parser.add_argument('--pooling_ratio', type=float, default=0.2,
                        help='SAGPool pooling ratio')
    parser.add_argument('--GNN', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'SAG'],
                        help='Graph conv mode for pooling')

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument('--verbose', type=int, default=1,
                        help='Print training progress')
    parser.add_argument('--print_every', type=int, default=0,
                        help='Print frequency (0 = end of epoch only)')
    parser.add_argument('--save_at', type=int, default=20,
                        help='Save checkpoint at this epoch')
    parser.add_argument('--measure', type=int, default=1,
                        help='Compute metrics during training')
    parser.add_argument('--patience', type=float, default=0.005,
                        help='Early stopping patience threshold')
    parser.add_argument('--finetune', type=int, default=1,
                        help='Enable fine-tuning of pretrained layers')

    # ── Cross-validation protocol ─────────────────────────────────────────
    parser.add_argument('--folds', type=str, default='all',
                        help="Folds to run: 'all' or comma list, e.g. 1,2,3")
    parser.add_argument('--seed', type=int, default=2019,
                        help='Base random seed')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='Models trained per fold with different seeds; '
                             'their predictions are averaged (ensemble)')
    parser.add_argument('--cv_workers', type=int, default=1,
                        help='Folds trained in parallel processes (use with '
                             '--gpu_ids -1 for the small feature-based models)')
    parser.add_argument('--split_path', type=str, default='',
                        help='Explicit split .pkl (default: auto-resolved)')
    parser.add_argument('--standardize_path', type=int, default=1,
                        help='Standardise pre-extracted histology embeddings '
                             'with training-fold statistics')
    parser.add_argument('--dedup_patients', type=int, default=1,
                        help='Omic-only training: keep one row per patient '
                             '(rows of a patient are identical in this mode)')

    # ── Unimodal pre-training for fusion networks ─────────────────────────
    parser.add_argument('--load_unimodal', type=int, default=1,
                        help='Fusion modes: load the per-fold pre-trained '
                             'omic network and freeze it')
    parser.add_argument('--unimodal_name', type=str, default='omic',
                        help='model_name of the pre-trained omic run')
    parser.add_argument('--unimodal_grph_name', type=str, default='graph',
                        help='model_name of the pre-trained graph run')
    parser.add_argument('--cache_graphs', type=int, default=1,
                        help='Keep loaded cell graphs in memory')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (graph / raw-image modes)')
    parser.add_argument('--unfreeze_epoch', type=int, default=5,
                        help='Epoch at which frozen unimodal nets are '
                             'unfrozen (0 = never unfreeze)')

    # ── Split generation (datasets/make_splits.py) ────────────────────────
    parser.add_argument('--roi_dir', type=str, default='all_st_patches_512',
                        help='ROI image sub-directory under dataroot')
    parser.add_argument('--graph_feat_type', type=str, default='cpc',
                        help='Graph feature type suffix of the graph dir')
    parser.add_argument('--ignore_missing_moltype', type=int, default=1)
    parser.add_argument('--ignore_missing_histype', type=int, default=0)
    parser.add_argument('--lr_decay_iters', type=int, default=10,
                        help='Step size for --lr_policy step')

    opt = parser.parse_args(argv)
    opt = finalize_options(opt)
    return opt


def finalize_options(opt):
    """Fill task-dependent defaults and parse GPU ids."""
    if opt.exp_name is None:
        opt.exp_name = 'grad_15' if opt.task == 'grad' else 'surv_15_rnaseq'
    if opt.act_type is None:
        opt.act_type = 'LSM' if opt.task == 'grad' else 'Sigmoid'
    if opt.label_dim is None:
        opt.label_dim = 3 if opt.task == 'grad' else 1
    opt = _parse_gpu_ids(opt)
    return opt


def _parse_gpu_ids(opt):
    """Parse GPU IDs from comma-separated string."""
    if isinstance(opt.gpu_ids, (list, tuple)):
        return opt
    str_ids = str(opt.gpu_ids).split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            opt.gpu_ids.append(gpu_id)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt


def get_device(opt):
    """Get the compute device based on configuration."""
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{opt.gpu_ids[0]}')
    return torch.device('cpu')


def print_options(opt):
    """Print all options to console."""
    message = '\n' + '=' * 60 + '\n'
    message += '  Pathomic Fusion — Configuration\n'
    message += '=' * 60 + '\n'
    for k, v in sorted(vars(opt).items()):
        message += f'  {k:>25}: {str(v):<30}\n'
    message += '=' * 60
    print(message)


def save_options(opt):
    """Save options to a text file in the checkpoint directory."""
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, 'train_opt.txt')
    with open(file_name, 'wt') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')
