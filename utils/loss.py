"""
Loss functions for Pathomic Fusion.
- CoxLoss: Partial likelihood loss for survival prediction
- NLLSurvLoss: Negative log-likelihood for survival (alternative)
"""

import numpy as np
import torch
import torch.nn as nn


def CoxLoss(survtime, censor, hazard_pred, device):
    """
    Cox proportional hazards partial likelihood loss (fully vectorized on GPU).

    Credit: Travers Ching (cox-nnet)
    """
    survtime = survtime.view(-1)
    censor = censor.view(-1)
    theta = hazard_pred.view(-1)

    # Vectorized risk set matrix: R_mat[i, j] = 1 if survtime[j] >= survtime[i]
    R_mat = (survtime.unsqueeze(0) >= survtime.unsqueeze(1)).float()

    exp_theta = torch.exp(theta)
    risk_sum = torch.sum(exp_theta.unsqueeze(0) * R_mat, dim=1)
    # Numerical stability
    risk_sum = torch.clamp(risk_sum, min=1e-7)

    loss_cox = -torch.mean((theta - torch.log(risk_sum)) * censor)
    return loss_cox


class NLLSurvLoss(nn.Module):
    """
    Negative log-likelihood survival loss.
    Alternative to CoxLoss for discrete-time survival models.
    """

    def __init__(self, alpha=0.15):
        super(NLLSurvLoss, self).__init__()
        self.alpha = alpha

    def forward(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            alpha = self.alpha

        batch_size = len(Y)
        Y = Y.view(batch_size, 1)
        c = c.view(batch_size, 1).float()
        S_padded = torch.cat([torch.ones_like(c), S], 1)

        uncensored_loss = -(1 - c) * (
            torch.log(torch.clamp(hazards.gather(1, Y), min=1e-8))
            + torch.log(S_padded.gather(1, Y).clamp(min=1e-8))
        )
        censored_loss = -c * torch.log(S_padded.gather(1, Y + 1).clamp(min=1e-8))

        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss
