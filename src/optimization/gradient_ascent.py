# src/optimization/gradient_ascent.py

import torch
import torch.optim as optim
import numpy as np

class DifferentiableSearch:
    def __init__(self, surrogate_model, scaler_metrics, device='cpu'):
        self.surrogate = surrogate_model.to(device)
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad = False
            
        self.scaler_metrics = scaler_metrics
        self.device = device

    def _calculate_fom(self, scaled_metrics):
        """Destransforma las métricas para calcular el FoMs real[cite: 143]."""
        # 1. Pasar a CPU y Numpy para usar el scaler de scikit-learn
        metrics_np = scaled_metrics.detach().cpu().numpy()
        real_metrics = self.scaler_metrics.inverse_transform(metrics_np)
        
        # 2. Volver a PyTorch preservando el grafo de computación
        # Esto es un truco avanzado: calculamos la diferencia entre el escalado y el real
        # para aplicar la operación de forma diferenciable.
        means = torch.tensor(self.scaler_metrics.mean_, dtype=torch.float32, device=self.device)
        scales = torch.tensor(self.scaler_metrics.scale_, dtype=torch.float32, device=self.device)
        
        # metrics = (scaled_metrics * scales) + means
        real_metrics_tensor = (scaled_metrics * scales) + means
        
        sndr = real_metrics_tensor[:, 0]
        bw = real_metrics_tensor[:, 1]
        power = real_metrics_tensor[:, 2] # Asumiendo W o un factor constante
        
        fom = sndr + 10 * torch.log10((bw / (power + 1e-12)) + 1e-12)
        return fom

    def optimize(self, initial_y_scaled, topology_onehot, mask, steps=100, lr=0.05):
        """Ascenso del gradiente acotado por la máscara topológica [cite: 148-154]."""
        y_opt = initial_y_scaled.clone().detach().to(self.device).requires_grad_(True)
        topology_t = topology_onehot.clone().detach().to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        optimizer = optim.Adam([y_opt], lr=lr)
        
        best_y = None
        best_fom = -float('inf')

        for step in range(steps):
            optimizer.zero_grad()
            
            x_surrogate = torch.cat([topology_t, y_opt], dim=1)
            predicted_metrics_scaled = self.surrogate(x_surrogate)
            
            foms = self._calculate_fom(predicted_metrics_scaled)
            loss = -torch.mean(foms) # Maximizar FoM
            
            loss.backward()
            
            # Anular gradiente de variables zero-padded
            y_opt.grad *= mask_t
            
            optimizer.step()
            
            current_fom = foms.mean().item()
            if current_fom > best_fom:
                best_fom = current_fom
                best_y = y_opt.clone().detach()

        return best_y, best_fom