"""
Multimodal Network Architectures and Model Factory for Pathomic Fusion.

Implements unimodal, bimodal (CNN⊗SNN, GCN⊗SNN), and trimodal (CNN⊗GCN⊗SNN)
fusion architectures, along with optimizer, scheduler, and regularization factories.

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import lr_scheduler

from models.path_net import PathNet, get_vgg
from models.graph_net import GraphNet
from models.omic_net import MaxNet
from models.fusion import BilinearFusion, TrilinearFusion_A, TrilinearFusion_B
from utils.initialization import init_weights, init_max_weights


# ── Activation Layer Factory ──────────────────────────────────────────

def define_act_layer(act_type='Sigmoid'):
    """Return activation layer based on type string."""
    if act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    elif act_type == 'ReLU':
        return nn.ReLU()
    elif act_type == 'LSM':
        return nn.LogSoftmax(dim=1)
    elif act_type == 'none' or act_type is None:
        return None
    else:
        raise NotImplementedError(f"Activation function [{act_type}] is not recognized.")


# ── Fusion Factories ──────────────────────────────────────────────────

def define_bifusion(skip=1, use_bilinear=1, gate1=1, gate2=1,
                    dim1=32, dim2=32, scale_dim1=1, scale_dim2=1,
                    mmhid=64, dropout_rate=0.25):
    """Instantiate a BilinearFusion module."""
    return BilinearFusion(
        skip=skip, use_bilinear=use_bilinear,
        gate1=gate1, gate2=gate2,
        dim1=dim1, dim2=dim2,
        scale_dim1=scale_dim1, scale_dim2=scale_dim2,
        mmhid=mmhid, dropout_rate=dropout_rate
    )


def define_trifusion(fusion_type='pofusion_A', skip=1, use_bilinear=1,
                     gate1=1, gate2=1, gate3=1,
                     dim1=32, dim2=32, dim3=32,
                     scale_dim1=1, scale_dim2=1, scale_dim3=1,
                     mmhid=96, dropout_rate=0.25):
    """Instantiate a TrilinearFusion module (variant A or B)."""
    if fusion_type == 'pofusion_B':
        return TrilinearFusion_B(
            skip=skip, use_bilinear=use_bilinear,
            gate1=gate1, gate2=gate2, gate3=gate3,
            dim1=dim1, dim2=dim2, dim3=dim3,
            scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3,
            mmhid=mmhid, dropout_rate=dropout_rate
        )
    else:
        return TrilinearFusion_A(
            skip=skip, use_bilinear=use_bilinear,
            gate1=gate1, gate2=gate2, gate3=gate3,
            dim1=dim1, dim2=dim2, dim3=dim3,
            scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3,
            mmhid=mmhid, dropout_rate=dropout_rate
        )


# ── Multimodal Networks ───────────────────────────────────────────────

class PathomicNet(nn.Module):
    """
    Bimodal Fusion Network: Histology CNN ⊗ Genomic SNN.
    
    Extracts features from histopathology ROI (VGG19-BN) and genomic vector (SNN),
    then fuses representations using gated bilinear Kronecker product.
    """

    def __init__(self, opt, act=None):
        super(PathomicNet, self).__init__()
        self.opt = opt
        self.act = act

        # Modality Sub-networks
        if getattr(opt, 'use_vgg_features', 0):
            # Pre-extracted path_dim-d CNN embeddings are fed straight
            # into the fusion module; no path network is needed.
            self.net_path = None
        else:
            self.net_path = get_vgg(
                arch='vgg19_bn', cfg='E', act=None,
                batch_norm=True, path_dim=opt.path_dim,
                label_dim=opt.label_dim, pretrained=True
            )

        self.net_omic = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=None,
            label_dim=opt.label_dim,
            init_max=True
        )

        # Bilinear Fusion Module
        self.fusion = define_bifusion(
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate, gate2=opt.omic_gate,
            dim1=opt.path_dim, dim2=opt.omic_dim,
            scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )

        # Classification / Hazard Head
        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        if getattr(self.opt, 'use_vgg_features', 0):
            x_p = kwargs['x_path']
            if x_p.dim() > 2:
                x_p = x_p.view(x_p.size(0), -1)
            if x_p.shape[-1] != self.opt.path_dim:
                raise ValueError(
                    f"use_vgg_features=1 expects {self.opt.path_dim}-d path "
                    f"embeddings, got {x_p.shape[-1]}-d"
                )
            feat_p = x_p
        else:
            feat_p, _ = self.net_path(**kwargs)

        feat_o, _ = self.net_omic(**kwargs)

        fused = self.fusion(feat_p, feat_o)
        hazard = self.classifier(fused)

        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return fused, hazard


class GraphomicNet(nn.Module):
    """
    Bimodal Fusion Network: Cell Graph GCN ⊗ Genomic SNN.
    
    Extracts features from cell graph (SAGEConv GCN) and genomic vector (SNN),
    then fuses representations using gated bilinear Kronecker product.
    """

    def __init__(self, opt, act=None):
        super(GraphomicNet, self).__init__()
        self.opt = opt
        self.act = act

        self.net_grph = GraphNet(
            features=1036, nhid=128, grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate, GNN=opt.GNN,
            use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio,
            act=None, label_dim=opt.label_dim, init_max=True
        )

        self.net_omic = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=None,
            label_dim=opt.label_dim,
            init_max=True
        )

        self.fusion = define_bifusion(
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.grph_gate, gate2=opt.omic_gate,
            dim1=opt.grph_dim, dim2=opt.omic_dim,
            scale_dim1=opt.grph_scale, scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )

        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        feat_g, _ = self.net_grph(**kwargs)
        feat_o, _ = self.net_omic(**kwargs)

        fused = self.fusion(feat_g, feat_o)
        hazard = self.classifier(fused)

        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return fused, hazard


class PathgraphomicNet(nn.Module):
    """
    Trimodal Fusion Network: Histology CNN ⊗ Cell Graph GCN ⊗ Genomic SNN.
    
    Integrates histopathology morphological features, cell spatial network topology,
    and molecular genomic profiles via 3-way gated Kronecker product fusion.
    """

    def __init__(self, opt, act=None):
        super(PathgraphomicNet, self).__init__()
        self.opt = opt
        self.act = act

        # 1. Path Network (Histology CNN)
        if getattr(opt, 'use_vgg_features', 0):
            # Pre-extracted path_dim-d CNN embeddings are fed straight
            # into the fusion module; no path network is needed.
            self.net_path = None
        else:
            self.net_path = get_vgg(
                arch='vgg19_bn', cfg='E', act=None,
                batch_norm=True, path_dim=opt.path_dim,
                label_dim=opt.label_dim, pretrained=True
            )

        # 2. Graph Network (Cell Spatial GCN)
        self.net_grph = GraphNet(
            features=1036, nhid=128, grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate, GNN=opt.GNN,
            use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio,
            act=None, label_dim=opt.label_dim, init_max=True
        )

        # 3. Omic Network (Genomic SNN)
        self.net_omic = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=None,
            label_dim=opt.label_dim,
            init_max=True
        )

        # Trilinear Fusion Module
        fusion_variant = getattr(opt, 'fusion_type', 'pofusion_A')
        self.fusion = define_trifusion(
            fusion_type=fusion_variant,
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate, gate2=opt.grph_gate, gate3=opt.omic_gate,
            dim1=opt.path_dim, dim2=opt.grph_dim, dim3=opt.omic_dim,
            scale_dim1=opt.path_scale, scale_dim2=opt.grph_scale, scale_dim3=opt.omic_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )

        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        if getattr(self.opt, 'use_vgg_features', 0):
            x_p = kwargs['x_path']
            if x_p.dim() > 2:
                x_p = x_p.view(x_p.size(0), -1)
            if x_p.shape[-1] != self.opt.path_dim:
                raise ValueError(
                    f"use_vgg_features=1 expects {self.opt.path_dim}-d path "
                    f"embeddings, got {x_p.shape[-1]}-d"
                )
            feat_p = x_p
        else:
            feat_p, _ = self.net_path(**kwargs)

        feat_g, _ = self.net_grph(**kwargs)
        feat_o, _ = self.net_omic(**kwargs)

        fused = self.fusion(feat_p, feat_g, feat_o)
        hazard = self.classifier(fused)

        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return fused, hazard


# ── Dual-Same-Modality Ablation Networks ──────────────────────────────

class PathpathNet(nn.Module):
    """Dual Path ablation network (fuses two path views/augmentations)."""
    def __init__(self, opt, act=None):
        super(PathpathNet, self).__init__()
        self.net_path = get_vgg(arch='vgg19_bn', cfg='E', act=None,
                                batch_norm=True, path_dim=opt.path_dim,
                                label_dim=opt.label_dim, pretrained=True)
        self.fusion = define_bifusion(
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate, gate2=opt.path_gate,
            dim1=opt.path_dim, dim2=opt.path_dim,
            scale_dim1=opt.path_scale, scale_dim2=opt.path_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )
        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        feat_p, _ = self.net_path(**kwargs)
        fused = self.fusion(feat_p, feat_p)
        hazard = self.classifier(fused)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return fused, hazard


class GraphgraphNet(nn.Module):
    """Dual Graph ablation network."""
    def __init__(self, opt, act=None):
        super(GraphgraphNet, self).__init__()
        self.net_grph = GraphNet(
            features=1036, nhid=128, grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate, GNN=opt.GNN,
            use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio,
            act=None, label_dim=opt.label_dim, init_max=True
        )
        self.fusion = define_bifusion(
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.grph_gate, gate2=opt.grph_gate,
            dim1=opt.grph_dim, dim2=opt.grph_dim,
            scale_dim1=opt.grph_scale, scale_dim2=opt.grph_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )
        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        feat_g, _ = self.net_grph(**kwargs)
        fused = self.fusion(feat_g, feat_g)
        hazard = self.classifier(fused)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return fused, hazard


class OmicomicNet(nn.Module):
    """Dual Omic ablation network."""
    def __init__(self, opt, act=None):
        super(OmicomicNet, self).__init__()
        self.net_omic = MaxNet(
            input_dim=opt.input_size_omic, omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate, act=None,
            label_dim=opt.label_dim, init_max=True
        )
        self.fusion = define_bifusion(
            skip=opt.skip, use_bilinear=opt.use_bilinear,
            gate1=opt.omic_gate, gate2=opt.omic_gate,
            dim1=opt.omic_dim, dim2=opt.omic_dim,
            scale_dim1=opt.omic_scale, scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid, dropout_rate=opt.dropout_rate
        )
        self.classifier = nn.Linear(opt.mmhid, opt.label_dim)
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        init_max_weights(self.classifier)

    def forward(self, **kwargs):
        feat_o, _ = self.net_omic(**kwargs)
        fused = self.fusion(feat_o, feat_o)
        hazard = self.classifier(fused)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return fused, hazard


# ── Histology head on pre-extracted CNN embeddings ────────────────────

class PathFeatNet(nn.Module):
    """
    Histology model for pre-extracted CNN embeddings (use_vgg_features=1).

    The split files store the `path_dim`-d embedding produced by the per-fold
    VGG19-BN histology CNN. This model is that CNN's final prediction layer,
    re-fit on the stored embeddings.
    """

    def __init__(self, path_dim=32, act=None, label_dim=1):
        super(PathFeatNet, self).__init__()
        self.linear = nn.Linear(path_dim, label_dim)
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        init_max_weights(self)

    def forward(self, **kwargs):
        features = kwargs['x_path']
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        out = self.linear(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        return features, out


# ── Model Factory ─────────────────────────────────────────────────────

def define_net(opt, custom_act=None):
    """
    Factory function to instantiate models based on configuration.
    
    Modes:
        'path': Histology CNN (VGG19-BN)
        'graph': Cell Graph GCN (SAGEConv + SAGPooling)
        'omic': Genomic SNN (4-layer ELU + AlphaDropout)
        'pathomic': CNN ⊗ SNN Bilinear Fusion
        'graphomic': GCN ⊗ SNN Bilinear Fusion
        'pathgraphomic': CNN ⊗ GCN ⊗ SNN Trilinear Fusion
        'pathpath', 'graphgraph', 'omicomic': Ablation baselines
    """
    act = custom_act if custom_act is not None else define_act_layer(opt.act_type)

    if opt.mode == 'path' and getattr(opt, 'use_vgg_features', 0):
        net = PathFeatNet(path_dim=opt.path_dim, act=act,
                          label_dim=opt.label_dim)
    elif opt.mode == 'path':
        net = get_vgg(
            arch='vgg19_bn', cfg='E', act=act,
            batch_norm=True, path_dim=opt.path_dim,
            label_dim=opt.label_dim, pretrained=True
        )
    elif opt.mode == 'graph':
        net = GraphNet(
            features=1036, nhid=128, grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate, GNN=opt.GNN,
            use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio,
            act=act, label_dim=opt.label_dim, init_max=True
        )
    elif opt.mode == 'omic':
        net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=True
        )
    elif opt.mode == 'pathomic':
        net = PathomicNet(opt, act=act)
    elif opt.mode == 'graphomic':
        net = GraphomicNet(opt, act=act)
    elif opt.mode == 'pathgraphomic':
        net = PathgraphomicNet(opt, act=act)
    elif opt.mode == 'pathpath':
        net = PathpathNet(opt, act=act)
    elif opt.mode == 'graphgraph':
        net = GraphgraphNet(opt, act=act)
    elif opt.mode == 'omicomic':
        net = OmicomicNet(opt, act=act)
    else:
        raise NotImplementedError(f"Model mode [{opt.mode}] is not implemented.")

    # Apply network initialization if specified
    if getattr(opt, 'init_type', 'none') != 'none':
        init_weights(net, init_type=opt.init_type, init_gain=opt.init_gain)

    # Move to GPU if available
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(opt.gpu_ids[0])

    return net


# ── Optimizer Factory ─────────────────────────────────────────────────

def define_optimizer(opt, model):
    """Factory to create optimizer for the given model and config."""
    # All parameters are registered: frozen ones receive no gradient and are
    # skipped by the optimizer until `unfreeze_unimodal` re-enables them.
    params = list(model.parameters())

    if opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=opt.lr,
            betas=(opt.beta1, opt.beta2),
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(
            params, lr=opt.lr,
            weight_decay=opt.weight_decay,
            initial_accumulator_value=0.1
        )
    else:
        raise NotImplementedError(
            f"Optimizer [{opt.optimizer_type}] is not implemented."
        )
    return optimizer


# ── Scheduler Factory ─────────────────────────────────────────────────

def define_scheduler(opt, optimizer):
    """
    Factory to create learning rate scheduler.
    
    Pathomic Fusion default is linear decay to 0 over `niter_decay` epochs
    starting after `niter` epochs.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return max(0.0, lr_l)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.niter_decay, eta_min=0
        )
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError(
            f"Learning rate policy [{opt.lr_policy}] is not implemented."
        )
    return scheduler


# ── Regularization Factory ────────────────────────────────────────────

def _l1(params):
    total = None
    for p in params:
        if p.requires_grad:
            term = p.abs().sum()
            total = term if total is None else total + term
    return total


def define_reg(opt, model):
    """
    L1 regularisation loss selected by opt.reg_type.

    reg_type options:
        'none': no regularisation
        'omic': omic network (the whole model in unimodal omic mode)
        'path': path network classifier head
        'mm':   fusion module and prediction head
        'all':  every trainable parameter
    """
    device = next(model.parameters()).device
    zero = torch.tensor(0.0, device=device)
    if opt.reg_type == 'none' or opt.lambda_reg == 0:
        return zero

    m = model.module if hasattr(model, 'module') else model
    params = []
    if opt.reg_type == 'omic':
        target = getattr(m, 'net_omic', None)
        if target is None and isinstance(m, MaxNet):
            target = m
        if target is not None:
            params = list(target.parameters())
    elif opt.reg_type == 'path':
        target = getattr(m, 'net_path', None)
        if target is None and isinstance(m, (PathNet, PathFeatNet)):
            target = m
        if target is not None:
            params = [p for n, p in target.named_parameters()
                      if 'classifier' in n or 'linear' in n]
    elif opt.reg_type == 'mm':
        for name in ('fusion', 'classifier'):
            if hasattr(m, name):
                params += list(getattr(m, name).parameters())
    elif opt.reg_type == 'all':
        params = list(m.parameters())

    loss_reg = _l1(params)
    if loss_reg is None:
        return zero
    return opt.lambda_reg * loss_reg
