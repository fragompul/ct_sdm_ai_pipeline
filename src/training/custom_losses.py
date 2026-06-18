# src/training/custom_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedMSELoss(nn.Module):
    """Pérdida MSE que ignora el zero-padding multiplicando por una máscara booleana."""
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, mask):
        raw_mse = F.mse_loss(y_pred, y_true, reduction='none')
        masked_mse = raw_mse * mask
        valid_elements = mask.sum(dim=1).clamp(min=1.0)
        return (masked_mse.sum(dim=1) / valid_elements).mean()

class MaskedNLLLoss(nn.Module):
    """Negative Log-Likelihood enmascarada para MDN."""
    def __init__(self):
        super(MaskedNLLLoss, self).__init__()

    def forward(self, pi, mu, sigma, y_true, mask):
        y_true_expanded = y_true.unsqueeze(1) 
        var = sigma ** 2
        diff = y_true_expanded - mu
        log_prob = -0.5 * torch.log(2 * math.pi * var) - (diff ** 2) / (2 * var)
        mask_expanded = mask.unsqueeze(1)
        masked_log_prob = log_prob * mask_expanded
        prob_sum_D = torch.sum(masked_log_prob, dim=2) 
        weighted_log_prob = prob_sum_D + torch.log(pi + 1e-8)
        nll = -torch.logsumexp(weighted_log_prob, dim=1)
        return nll.mean()

class MaskedVAELoss(nn.Module):
    """Loss para cVAE: Masked Reconstrucción + KL Divergence."""
    def __init__(self, beta=1.0):
        super(MaskedVAELoss, self).__init__()
        self.masked_mse = MaskedMSELoss()
        self.beta = beta

    def forward(self, recon_y, y_true, mu, logvar, mask):
        recon_loss = self.masked_mse(recon_y, y_true, mask)
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon_loss + self.beta * kl_loss