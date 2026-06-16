# src/training/trainer.py

import torch
import torch.optim as optim
import os
import time

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
                cond, y_true, mask = batch['cond'].to(self.device), batch['y'].to(self.device), batch['mask'].to(self.device)
                pi, mu, sigma = self.model(cond)
                loss = self.criterion(pi, mu, sigma, y_true, mask)
            else:
                x, y_true = batch['x'].to(self.device), batch['y'].to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(x, y_pred, y_true)
                
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)

    def train_full(self, dataloader, epochs, is_masked, logger, phase_name):
        """Ejecuta todo el entrenamiento midiendo tiempos."""
        start_time = time.time()
        final_loss = 0.0
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(dataloader, is_masked=is_masked)
            epoch_time = time.time() - epoch_start
            
            final_loss = loss
            
            if epoch % 50 == 0 or epoch == epochs:
                logger.info(f"[{phase_name}] Epoch {epoch}/{epochs} | Loss: {loss:.6f} | Tiempo Epoch: {epoch_time:.4f}s")
                
        total_time = time.time() - start_time
        return final_loss, total_time

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)