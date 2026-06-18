# src/models/phase3_gen.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. cVAE 
class ConditionalVAE(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim = spec_dim + num_topologies
        
        self.encoder = nn.Sequential(
            nn.Linear(super_vector_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, super_vector_dim)
        )

    def encode(self, y, cond):
        h = self.encoder(torch.cat([y, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, cond):
        mu, logvar = self.encode(y, cond)
        z = self.reparameterize(mu, logvar)
        recon_y = self.decoder(torch.cat([z, cond], dim=1))
        return recon_y, mu, logvar

# 2. MDN 
class MixtureDensityNetwork(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, num_mixtures=5, hidden_dim=128):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.super_vector_dim = super_vector_dim
        cond_dim = spec_dim + num_topologies
        
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.pi = nn.Linear(hidden_dim, num_mixtures)
        self.mu = nn.Linear(hidden_dim, num_mixtures * super_vector_dim)
        self.sigma = nn.Linear(hidden_dim, num_mixtures * super_vector_dim)

    def forward(self, cond):
        h = self.shared(cond)
        pi = torch.softmax(self.pi(h), dim=1)
        bs = cond.size(0)
        mu = self.mu(h).view(bs, self.num_mixtures, self.super_vector_dim)
        sigma = F.elu(self.sigma(h)) + 1 + 1e-8
        sigma = sigma.view(bs, self.num_mixtures, self.super_vector_dim)
        return pi, mu, sigma

# 3. MC-Dropout ResNet (Bayesian Approximation)
class MCDropoutResNet(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        cond_dim = spec_dim + num_topologies
        self.in_layer = nn.Linear(cond_dim, hidden_dim)
        
        self.res1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_layer = nn.Linear(hidden_dim, super_vector_dim)

    def forward(self, cond):
        x = F.relu(self.in_layer(cond))
        res = x
        x = F.relu(self.res1(x))
        x = self.dropout(x)
        x = F.relu(self.res2(x))
        x = self.dropout(x) + res  # Conexión residual
        return self.out_layer(x)

# 4. Tabular DDPM (Diffusion Model)
class TabularDDPM(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, hidden_dim=256):
        super().__init__()
        cond_dim = spec_dim + num_topologies
        # La red predice el ruido añadido dado y_ruidoso, condicion, y tiempo t
        self.net = nn.Sequential(
            nn.Linear(super_vector_dim + cond_dim + 1, hidden_dim), nn.Mish(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, super_vector_dim)
        )

    def forward(self, y_noisy, cond, t):
        # t debe estar normalizado entre 0 y 1
        x = torch.cat([y_noisy, cond, t], dim=1)
        return self.net(x)

# 5. Conditional GAN (cGAN)
class cGANGenerator(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, latent_dim=32, hidden_dim=128):
        super().__init__()
        cond_dim = spec_dim + num_topologies
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, super_vector_dim)
        )
    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=1))

class cGANDiscriminator(nn.Module):
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, hidden_dim=128):
        super().__init__()
        cond_dim = spec_dim + num_topologies
        self.net = nn.Sequential(
            nn.Linear(super_vector_dim + cond_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, y, cond):
        return self.net(torch.cat([y, cond], dim=1))