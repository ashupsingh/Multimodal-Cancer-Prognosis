"""
Multimodal Fusion Modules for Pathomic Fusion.

Implements gating-based attention and Kronecker product fusion:
- BilinearFusion: For bimodal fusion (CNN⊗SNN, GCN⊗SNN)
- TrilinearFusion_A: For trimodal fusion (CNN⊗GCN⊗SNN) — survival
- TrilinearFusion_B: For trimodal fusion (CNN⊗GCN⊗SNN) — grade classification

Reference: Chen et al. (2022), IEEE TMI
"""

import torch
import torch.nn as nn

from utils.initialization import init_max_weights


class BilinearFusion(nn.Module):
    """
    Bilinear (Kronecker product) fusion with gated multimodal units.
    
    Architecture:
    1. Gated attention: Each modality is gated using bilinear interaction
       with the other modality (GMU)
    2. Kronecker product: Outer product of gated representations
    3. Post-fusion encoder: MLP to compress fused representation
    
    Args:
        skip: Use skip connections (concatenate gated outputs)
        use_bilinear: Use bilinear layer for gating (vs linear concat)
        gate1/gate2: Enable gating for each modality
        dim1/dim2: Input dimensions of each modality
        scale_dim1/scale_dim2: Dimension scaling factors
        mmhid: Multimodal hidden dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1,
                 dim1=32, dim2=32, scale_dim1=1, scale_dim2=1,
                 mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og = dim1, dim2
        dim1, dim2 = dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1 + dim2 + 2 if skip else 0

        # Gated Multimodal Unit for modality 1
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Gated Multimodal Unit for modality 2
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Post-fusion encoder
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        init_max_weights(self)

    def forward(self, vec1, vec2):
        """
        Args:
            vec1: (batch, dim1) — e.g., path or graph features
            vec2: (batch, dim2) — e.g., omic features
            
        Returns:
            (batch, mmhid) fused representation
        """
        device = vec1.device

        # ── Gated Multimodal Units ──
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec2)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            )
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            )
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # ── Kronecker Product Fusion ──
        o1 = torch.cat(
            (o1, torch.ones(o1.shape[0], 1, device=device)), dim=1
        )
        o2 = torch.cat(
            (o2, torch.ones(o2.shape[0], 1, device=device)), dim=1
        )
        # Outer product: (B, dim1+1, 1) × (B, 1, dim2+1) → flatten
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat((out, o1, o2), dim=1)

        out = self.encoder2(out)
        return out


class TrilinearFusion_A(nn.Module):
    """
    Trilinear fusion for 3 modalities (variant A — for survival prediction).
    
    Gating strategy:
    - Path gated by Omic
    - Graph gated by Omic
    - Omic gated by Path
    """

    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1,
                 dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1,
                 scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1, self.gate2, self.gate3 = gate1, gate2, gate3

        dim1_og, dim2_og, dim3_og = dim1, dim2, dim3
        dim1 = dim1 // scale_dim1
        dim2 = dim2 // scale_dim2
        dim3 = dim3 // scale_dim3
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # Path GMU (gated by Omic)
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Graph GMU (gated by Omic)
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim3_og, dim2) if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Omic GMU (gated by Path)
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Post-fusion encoder
        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        """
        Args:
            vec1: Path features (batch, dim1)
            vec2: Graph features (batch, dim2)
            vec3: Omic features (batch, dim3)
        """
        device = vec1.device

        # Gate Path with Omic
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3) if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        # Gate Graph with Omic
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec3) if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            )
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # Gate Omic with Path
        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3) if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )
            o3 = self.linear_o3(torch.sigmoid(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        # 3-way Kronecker product
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), dim=1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_B(nn.Module):
    """
    Trilinear fusion variant B — for grade classification.
    
    Gating strategy differs from A:
    - Path gated by Omic
    - Graph gated by Path (instead of Omic)
    - Omic gated by Path
    """

    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1,
                 dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1,
                 scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1, self.gate2, self.gate3 = gate1, gate2, gate3

        dim1_og, dim2_og, dim3_og = dim1, dim2, dim3
        dim1 = dim1 // scale_dim1
        dim2 = dim2 // scale_dim2
        dim3 = dim3 // scale_dim3
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # Path GMU (gated by Omic)
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Graph GMU (gated by Path)
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim1_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Omic GMU (gated by Path)
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid),
            nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        device = vec1.device

        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3) if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec1) if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec1), dim=1))
            )
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3) if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )
            o3 = self.linear_o3(torch.sigmoid(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), dim=1)
        out = self.encoder2(out)
        return out
