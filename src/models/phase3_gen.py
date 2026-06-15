# src/models/phase3_gen.py

import torch
import torch.nn as nn

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder para variables de diseño."""
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, latent_dim=16):
        super(ConditionalVAE, self).__init__()
        
        # Condición: Especificaciones + One-Hot(Topology)
        cond_dim = spec_dim + num_topologies
        self.latent_dim = latent_dim
        
        # ENCODER: [y_design, condition] -> latent space z
        self.encoder = nn.Sequential(
            nn.Linear(super_vector_dim + cond_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # DECODER: [latent z, condition] -> y_design
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, super_vector_dim)
        )

    def encode(self, y, cond):
        inputs = torch.cat([y, cond], dim=1)
        hidden = self.encoder(inputs)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        inputs = torch.cat([z, cond], dim=1)
        return self.decoder(inputs)

    def forward(self, y, cond):
        mu, logvar = self.encode(y, cond)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decode(z, cond)
        return y_recon, mu, logvar
        
    def sample(self, cond, num_samples=1000):
        """Inferencia: Genera una nube de puntos densa."""
        batch_size = cond.size(0)
        z = torch.randn(batch_size * num_samples, self.latent_dim).to(cond.device)
        cond_repeated = cond.repeat_interleave(num_samples, dim=0)
        return self.decode(z, cond_repeated)

class MixtureDensityNetwork(nn.Module):
    """MDN para predecir distribuciones estadísticas de las variables de diseño."""
    def __init__(self, spec_dim=3, num_topologies=12, super_vector_dim=50, num_mixtures=5):
        super(MixtureDensityNetwork, self).__init__()
        
        cond_dim = spec_dim + num_topologies
        self.super_vector_dim = super_vector_dim
        self.num_mixtures = num_mixtures
        
        self.shared_network = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Salidas de la MDN: pi (pesos), mu (medias), sigma (desviaciones) 
        self.pi = nn.Linear(128, num_mixtures)
        self.mu = nn.Linear(128, num_mixtures * super_vector_dim)
        self.sigma = nn.Linear(128, num_mixtures * super_vector_dim)

    def forward(self, cond):
        hidden = self.shared_network(cond)
        
        # Softmax para asegurar que las mezclas sumen 1
        pi = torch.softmax(self.pi(hidden), dim=1) 
        
        # Reshape para separar num_mixtures y D
        batch_size = cond.size(0)
        mu = self.mu(hidden).view(batch_size, self.num_mixtures, self.super_vector_dim)
        
        # ELU + 1 + epsilon asegura que la desviación estándar sea siempre positiva
        sigma = torch.nn.functional.elu(self.sigma(hidden)) + 1 + 1e-8
        sigma = sigma.view(batch_size, self.num_mixtures, self.super_vector_dim)
        
        return pi, mu, sigma