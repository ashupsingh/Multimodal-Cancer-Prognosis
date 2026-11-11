"""
Loss functions for Pathomic Fusion.
- CoxLoss: Partial likelihood loss for survival prediction
- NLLSurvLoss: Negative log-likelihood for survival (alternative)
"""

import numpy as np
import torch
import torch.nn as nn


def CoxLoss(survtime, censor, hazard_pred, device=None):
    """
    Cox proportional hazards negative partial log-likelihood.

    Vectorised and numerically stable (log-sum-exp over each risk set).
    Risk set of sample i = all samples j with survtime[j] >= survtime[i].

    Args:
        survtime: (N,) observed times
        censor: (N,) event indicator (1 = event observed)
        hazard_pred: (N,) or (N, 1) predicted log-risk
    """
    survtime = survtime.view(-1)
    censor = censor.view(-1)
    theta = hazard_pred.view(-1)

    in_risk_set = survtime.unsqueeze(0) >= survtime.unsqueeze(1)
    theta_mat = theta.unsqueeze(0).expand(len(theta), -1)
    log_risk = torch.logsumexp(
        theta_mat.masked_fill(~in_risk_set, float('-inf')), dim=1
    )
    return -torch.mean((theta - log_risk) * censor)


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
