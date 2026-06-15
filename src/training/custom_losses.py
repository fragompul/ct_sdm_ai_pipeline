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
        # mask tiene 1s en variables pertinentes y 0s en el padding
        raw_mse = F.mse_loss(y_pred, y_true, reduction='none')
        masked_mse = raw_mse * mask
        
        # Promediamos solo sobre los elementos válidos para no sesgar el gradiente
        valid_elements = mask.sum(dim=1)
        # Evitamos división por cero sumando un epsilon
        loss = masked_mse.sum(dim=1) / (valid_elements + 1e-8)
        
        return loss.mean()

class MaskedNLLLoss(nn.Module):
    """Negative Log-Likelihood enmascarada para la Mixture Density Network (MDN)."""
    def __init__(self):
        super(MaskedNLLLoss, self).__init__()

    def forward(self, pi, mu, sigma, y_true, mask):
        # y_true shape: [batch_size, D]
        # Expandimos y_true para que coincida con las mezclas Gaussianas
        y_true_expanded = y_true.unsqueeze(1) # [batch_size, 1, D]
        
        # Calculamos la probabilidad Gaussiana normalizada
        var = sigma ** 2
        diff = y_true_expanded - mu
        log_prob = -0.5 * torch.log(2 * math.pi * var) - (diff ** 2) / (2 * var)
        
        # Multiplicamos por la máscara: [batch_size, 1, D] * [batch_size, 1, D]
        mask_expanded = mask.unsqueeze(1)
        masked_log_prob = log_prob * mask_expanded
        
        # Sumamos sobre las dimensiones de diseño D
        prob_sum_D = torch.sum(masked_log_prob, dim=2) # [batch_size, num_mixtures]
        
        # Ponderamos por pi (mixture weights) y aplicamos log-sum-exp trick por estabilidad
        weighted_log_prob = prob_sum_D + torch.log(pi + 1e-8)
        nll = -torch.logsumexp(weighted_log_prob, dim=1)
        
        return nll.mean()