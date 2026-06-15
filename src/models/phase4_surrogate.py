# src/models/phase4_surrogate.py

import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

class SurrogateMLP(nn.Module):
    """
    Proxy ultra-rápido para NGSpice.
    Entrada: OneHot(Topología) + Súper-Vector de Diseño.
    Salida: [SNDR, Bw, Power].
    """
    def __init__(self, num_topologies=12, super_vector_dim=50, output_metrics=3):
        super(SurrogateMLP, self).__init__()
        
        input_dim = num_topologies + super_vector_dim
        
        # Uso de activaciones SiLU (Swish) para garantizar un gradiente suave
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, output_metrics)
        )

    def forward(self, x_surrogate):
        return self.network(x_surrogate)

class PINNSurrogateLoss(nn.Module):
    """
    Función de pérdida para la PINN: L_total = L_data + lambda * L_physics.
    """
    def __init__(self, lambda_physics=0.1):
        super(PINNSurrogateLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics

    def physics_constraints(self, y_design_inputs, y_metrics_preds):
        """
        Penaliza predicciones que violen leyes fundamentales.
        Ejemplo conceptual: El ruido térmico (kT/C) impone un límite al SNDR.
        """
        # Supongamos que el índice 0 en y_metrics_preds es SNDR y el índice 5 en y_design es Cint1
        sndr_pred = y_metrics_preds[:, 0]
        cint1 = y_design_inputs[:, 5]
        
        # Lógica física simulada: Si Cint1 es muy pequeño, un SNDR altísimo es físically imposible.
        # constraint_violation = ReLU(SNDR_pred - Max_Theoretical_SNDR_from_Cint)
        # Esto es un placeholder; los expertos en hardware definirían las ecuaciones exactas.
        physical_penalty = torch.mean(torch.relu(sndr_pred - (10 * torch.log10(cint1 + 1e-8) + 200))) 
        return physical_penalty

    def forward(self, y_design_inputs, y_metrics_preds, y_metrics_true):
        l_data = self.mse(y_metrics_preds, y_metrics_true)
        l_physics = self.physics_constraints(y_design_inputs, y_metrics_preds)
        return l_data + self.lambda_physics * l_physics

def build_xgb_surrogate(random_state=42):
    """Fallback no diferenciable usando Multi-output XGBoost."""
    xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    return MultiOutputRegressor(xgb_estimator)