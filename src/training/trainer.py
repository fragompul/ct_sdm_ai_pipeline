# src/training/trainer.py

import torch
import torch.optim as optim
import os

class GenericTrainer:
    def __init__(self, model, criterion, lr=0.001, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train_epoch(self, dataloader, is_masked=False):
        self.model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            if is_masked:
                # Fase 3: Entrenamiento MDN
                cond = batch['cond'].to(self.device)
                y_true = batch['y'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                pi, mu, sigma = self.model(cond)
                loss = self.criterion(pi, mu, sigma, y_true, mask)
            else:
                # Fase 4: Entrenamiento Surrogate
                x = batch['x'].to(self.device)
                y_true = batch['y'].to(self.device)
                y_pred = self.model(x)
                
                loss = self.criterion(x, y_pred, y_true) # PINNSurrogateLoss acepta 'x' para límites físicos
                
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
