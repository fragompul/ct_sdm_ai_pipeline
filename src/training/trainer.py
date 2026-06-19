# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

class GenerativeTrainer:
    """Entrenador Universal para MDN, cVAE, MC-Dropout, cGAN y Diffusion."""
    def __init__(self, model_name, model, criterion, lr=0.001, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.criterion = criterion
        
        if model_name == "cGAN":
            self.netG = model['G'].to(device)
            self.netD = model['D'].to(device)
            self.optG = optim.Adam(self.netG.parameters(), lr=lr)
            self.optD = optim.Adam(self.netD.parameters(), lr=lr*0.5) # D aprende más despacio
        else:
            self.model = model.to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        if self.model_name != "cGAN": self.model.train()
        else: self.netG.train(); self.netD.train()
            
        epoch_loss = 0.0
        
        for batch in dataloader:
            cond = batch['cond'].to(self.device)
            y_true = batch['y'].to(self.device)
            mask = batch['mask'].to(self.device)
            bs = cond.size(0)
            
            if self.model_name != "cGAN": self.optimizer.zero_grad()
            
            if self.model_name == "MDN":
                pi, mu, sigma = self.model(cond)
                loss = self.criterion(pi, mu, sigma, y_true, mask)
                
            elif self.model_name == "cVAE":
                recon_y, mu, logvar = self.model(y_true, cond)
                loss = self.criterion(recon_y, y_true, mu, logvar, mask)
                
            elif self.model_name == "MCDropout":
                y_pred = self.model(cond)
                loss = self.criterion(y_pred, y_true, mask)
                
            elif self.model_name == "TabularDDPM":
                # Simular ruido aleatorio y tiempo t
                t = torch.rand(bs, 1).to(self.device)
                noise = torch.randn_like(y_true)
                y_noisy = y_true * (1 - t) + noise * t # Interpolación simple DDPM-like
                noise_pred = self.model(y_noisy, cond, t)
                # El objetivo es recuperar el ruido. Usamos MaskedMSE.
                loss = self.criterion(noise_pred, noise, mask)
                
            elif self.model_name == "cGAN":
                # 1. Entrenar Discriminador
                self.optD.zero_grad()
                real_valid = torch.ones(bs, 1).to(self.device)
                fake_valid = torch.zeros(bs, 1).to(self.device)
                
                # y_true_masked es lo que el D debe evaluar
                d_real_loss = nn.BCELoss()(self.netD(y_true * mask, cond), real_valid)
                
                z = torch.randn(bs, self.netG.net[0].in_features - cond.size(1)).to(self.device)
                fake_y = self.netG(z, cond)
                
                d_fake_loss = nn.BCELoss()(self.netD(fake_y.detach() * mask, cond), fake_valid)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optD.step()
                
                # 2. Entrenar Generador
                self.optG.zero_grad()
                g_loss = nn.BCELoss()(self.netD(fake_y * mask, cond), real_valid)
                g_loss.backward()
                self.optG.step()
                
                loss = g_loss # Registramos la pérdida del generador para la curva
                
            if self.model_name != "cGAN":
                loss.backward()
                self.optimizer.step()
                
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)

    def eval_epoch(self, dataloader):
        if self.model_name == "cGAN": return 0.0 # GANs no tienen val loss tradicional representativa
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                cond = batch['cond'].to(self.device)
                y_true = batch['y'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                if self.model_name == "MDN":
                    pi, mu, sigma = self.model(cond)
                    loss = self.criterion(pi, mu, sigma, y_true, mask)
                elif self.model_name == "cVAE":
                    recon_y, mu, logvar = self.model(y_true, cond)
                    loss = self.criterion(recon_y, y_true, mu, logvar, mask)
                elif self.model_name == "MCDropout":
                    y_pred = self.model(cond)
                    loss = self.criterion(y_pred, y_true, mask)
                elif self.model_name == "TabularDDPM":
                    t = torch.rand(cond.size(0), 1).to(self.device)
                    noise = torch.randn_like(y_true)
                    y_noisy = y_true * (1 - t) + noise * t
                    noise_pred = self.model(y_noisy, cond, t)
                    loss = self.criterion(noise_pred, noise, mask)
                    
                val_loss += loss.item()
        return val_loss / len(dataloader)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.model_name == "cGAN":
            torch.save({'G': self.netG.state_dict(), 'D': self.netD.state_dict()}, path)
        else:
            torch.save(self.model.state_dict(), path)

class GenericTrainer:
    """Entrenador Estándar para la Fase 4: Proxy Surrogate y PINN."""
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
            
            # x contiene el Super-Vector y y_true las métricas (SNDR, Bw, Power)
            x = batch['x'].to(self.device)
            y_true = batch['y'].to(self.device)
            
            y_pred = self.model(x)
            
            # La PINNSurrogateLoss requiere (entrada_diseño, predicciones, valor_real)
            loss = self.criterion(x, y_pred, y_true)
                
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
