# src/optimization/gradient_ascent.py

import torch
import torch.optim as optim

class DifferentiableSearch:
    def __init__(self, surrogate_model, mask_matrix):
        """
        surrogate_model: El modelo MLP de la Fase 4 entrenado y con pesos congelados.
        mask_matrix: Para asegurar que el optimizador no cambie el zero-padding.
        """
        self.surrogate = surrogate_model
        self.surrogate.eval() # Modo inferencia
        for param in self.surrogate.parameters():
            param.requires_grad = False # Congelamos la red
            
        self.mask_matrix = mask_matrix

    def _calculate_fom(self, metrics):
        """Calcula el Figure of Merit de Schreier.
        Asumiendo metrics = [SNDR, Bw, Power]
        FoMs = SNDR + 10 * log10(Bw / Power)
        """
        sndr = metrics[:, 0]
        bw = metrics[:, 1]
        power = metrics[:, 2]
        
        # Prevenir logaritmos negativos
        fom = sndr + 10 * torch.log10((bw / (power + 1e-12)) + 1e-12)
        return fom

    def optimize(self, initial_y_design, topology_onehot, target_specs, steps=100, lr=0.01):
        """
        Ajusta iterativamente initial_y_design para maximizar el FoMs.
        """
        # initial_y_design viene de muestrear la Fase 3
        # Clonamos y habilitamos el gradiente sobre los datos de ENTRADA
        y_opt = initial_y_design.clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([y_opt], lr=lr)
        
        best_y = None
        best_fom = -float('inf')

        for step in range(steps):
            optimizer.zero_grad()
            
            # Concatenamos la topología estática con las variables optimizables
            x_surrogate = torch.cat([topology_onehot, y_opt], dim=1)
            
            # Predicción rápida del surrogate
            predicted_metrics = self.surrogate(x_surrogate)
            
            # Queremos MAXIMIZAR FoM, así que minimizamos -FoM
            foms = self._calculate_fom(predicted_metrics)
            loss = -torch.mean(foms) 
            
            loss.backward()
            
            # Aplicar la máscara al gradiente antes del step para no actualizar el zero-padding
            y_opt.grad *= self.mask_matrix
            
            optimizer.step()
            
            # Opcional: Proyección para asegurar que los componentes (C, R, gm) sean positivos
            with torch.no_grad():
                y_opt.clamp_(min=1e-15) 
            
            # Guardamos el mejor candidato
            current_fom = foms.mean().item()
            if current_fom > best_fom:
                best_fom = current_fom
                best_y = y_opt.clone().detach()

        return best_y, best_fom