# src/training/custom_losses.py

import torch
import torch.nn as nn
import math

class MaskedNLLLoss(nn.Module):
    """Negative Log-Likelihood enmascarada para la Mixture Density Network (Fase 3)."""
    def __init__(self):
        super(MaskedNLLLoss, self).__init__()

    def forward(self, pi, mu, sigma, y_true, mask):
        # y_true shape: [batch, D] -> Expandimos para coincidir con mezclas [batch, 1, D]
        y_true_expanded = y_true.unsqueeze(1) 
        
        # Probabilidad Gaussiana logarítmica
        var = sigma ** 2
        diff = y_true_expanded - mu
        log_prob = -0.5 * torch.log(2 * math.pi * var) - (diff ** 2) / (2 * var)
        
        # Aplicar Ecuación 3 del PDF: Multiplicar por la máscara booleana 
        mask_expanded = mask.unsqueeze(1)
        masked_log_prob = log_prob * mask_expanded
        
        # Sumar sobre las dimensiones de diseño D
        prob_sum_D = torch.sum(masked_log_prob, dim=2) 
        
        # Ponderar por pi y aplicar log-sum-exp para estabilidad numérica
        weighted_log_prob = prob_sum_D + torch.log(pi + 1e-8)
        nll = -torch.logsumexp(weighted_log_prob, dim=1)
        
        return nll.mean()
