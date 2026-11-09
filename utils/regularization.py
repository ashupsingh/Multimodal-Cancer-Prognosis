"""
Regularization utilities for Pathomic Fusion.
L1 regularization on model weights to prevent overfitting.
"""

import torch


def regularize_weights(model, reg_type=None):
    """L1 regularization on all model parameters."""
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg


def regularize_path_weights(model, reg_type=None):
    """L1 regularization on path classifier and linear layers only."""
    l1_reg = None

    # Handle DataParallel wrapper
    m = model.module if hasattr(model, 'module') else model

    if hasattr(m, 'classifier'):
        for W in m.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if hasattr(m, 'linear'):
        for W in m.linear.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg if l1_reg is not None else torch.tensor(0.0)


def regularize_MM_weights(model, reg_type=None):
    """L1 regularization on multimodal fusion components."""
    l1_reg = None
    m = model.module if hasattr(model, 'module') else model

    target_attrs = [
        'omic_net', 'fusion', 'classifier',
        'linear_h_path', 'linear_h_omic', 'linear_h_grph',
        'linear_z_path', 'linear_z_omic', 'linear_z_grph',
        'linear_o_path', 'linear_o_omic', 'linear_o_grph',
        'encoder1', 'encoder2',
    ]

    for attr_name in target_attrs:
        if hasattr(m, attr_name):
            for W in getattr(m, attr_name).parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg if l1_reg is not None else torch.tensor(0.0)


def regularize_MM_omic(model, reg_type=None):
    """L1 regularization on omic network only."""
    l1_reg = None
    m = model.module if hasattr(model, 'module') else model

    if hasattr(m, 'omic_net'):
        for W in m.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg if l1_reg is not None else torch.tensor(0.0)
