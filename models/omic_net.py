"""
Self-Normalizing Network (SNN) for Genomic Features.

Architecture from Pathomic Fusion (Chen et al., 2022):
- 4 fully connected layers with SELU activations and AlphaDropout
- Hidden dims: input → 64 → 48 → 32 → omic_dim
- Self-normalizing initialization (stdv = 1/sqrt(fan_in))
- Processes 320 genomic features (79 CNV + 1 mutation + 240 RNA-Seq)
"""

import torch
import torch.nn as nn
from torch.nn import Parameter

from utils.initialization import init_max_weights


class MaxNet(nn.Module):
    """
    Self-Normalizing Network for genomic feature learning.
    
    Uses ELU activations and AlphaDropout for self-normalizing properties.
    The SNN architecture naturally keeps activations at zero mean and unit 
    variance through training, reducing internal covariate shift.
    
    Args:
        input_dim: Number of input genomic features (default: 80)
        omic_dim: Output embedding dimension (default: 32)
        dropout_rate: AlphaDropout probability (default: 0.25)
        act: Output activation function
        label_dim: Number of output classes
        init_max: Use SNN-style weight initialization
    """

    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25,
                 act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()

        hidden = [64, 48, 32, 32]
        self.act = act

        # 4-layer SNN encoder with ELU + AlphaDropout
        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False)
        )

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False)
        )

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False)
        )

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False)
        )

        self.encoder = nn.Sequential(
            encoder1, encoder2, encoder3, encoder4
        )
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max:
            init_max_weights(self)

        self.output_range = Parameter(
            torch.FloatTensor([6]), requires_grad=False
        )
        self.output_shift = Parameter(
            torch.FloatTensor([-3]), requires_grad=False
        )

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        out = self.classifier(features)

        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out
