# src/models/phase4_surrogate.py

import torch
import torch.nn as nn

class SurrogateMLP(nn.Module):
    """Proxy ultra-rápido MLP clásico para NGSpice [cite: 140-144]."""
    def __init__(self, num_topologies=12, super_vector_dim=50, output_metrics=3, hidden_dim=256):
        super(SurrogateMLP, self).__init__()
        input_dim = num_topologies + super_vector_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_metrics)
        )

    def forward(self, x_surrogate):
        return self.network(x_surrogate)

class SurrogateResNet(nn.Module):
    """Arquitectura Residual para datos tabulares (SOTA en Regresión Tabular)."""
    def __init__(self, num_topologies=12, super_vector_dim=50, output_metrics=3, hidden_dim=256, dropout=0.1):
        super(SurrogateResNet, self).__init__()
        input_dim = num_topologies + super_vector_dim
        
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bloque Residual 1
        self.res1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Bloque Residual 2
        self.res2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.out_proj = nn.Linear(hidden_dim, output_metrics)

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return self.out_proj(x)

class PINNSurrogateLoss(nn.Module):
    """Función de pérdida para la PINN [cite: 145-147]."""
    def __init__(self, lambda_physics=0.1):
        super(PINNSurrogateLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics

    def physics_constraints(self, y_design_inputs, y_metrics_preds):
        """
        Placeholder: Se definirá analíticamente cuando tengamos las ecuaciones exactas.
        Devolvemos 0.0 para no interferir con la optimización actual.
        """
        return 0.0 

    def forward(self, y_design_inputs, y_metrics_preds, y_metrics_true):
        l_data = self.mse(y_metrics_preds, y_metrics_true)
        l_physics = self.physics_constraints(y_design_inputs, y_metrics_preds)
        
        if isinstance(l_physics, float) and l_physics == 0.0:
            return l_data # Evita errores del autograd al sumar escalares puros
            
        return l_data + self.lambda_physics * l_physics