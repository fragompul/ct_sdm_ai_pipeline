# src/models/phase1_ood.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

class DenseAutoencoder(nn.Module):
    """Autoencoder Denso propuesto para proyección de OOD."""
    def __init__(self, input_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4) # Espacio latente
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class OODDetectorBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "IsolationForest": IsolationForest(random_state=random_state, contamination=0.1),
            "OneClassSVM": OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
            "GMM": GaussianMixture(n_components=3, covariance_type='full', random_state=random_state)
        }
        self.fitted_models = {}

    def train_baselines(self, X_train_valid):
        """Entrena los modelos usando SOLO especificaciones válidas."""
        for name, model in self.models.items():
            model.fit(X_train_valid)
            self.fitted_models[name] = model
            print(f"[{name}] Entrenado correctamente.")

    def evaluate(self, X_test, y_test):
        """Evalúa calculando el ROC-AUC con el dataset sintético."""
        results = {}
        for name, model in self.fitted_models.items():
            if name == "IsolationForest":
                # decision_function: mayor es más normal
                scores = model.decision_function(X_test) 
            elif name == "OneClassSVM":
                scores = model.decision_function(X_test)
            elif name == "GMM":
                # score_samples da el log-likelihood
                scores = model.score_samples(X_test) 
                
            # y_test tiene 1 (válido) y -1 (anomalía). Ajustamos para ROC-AUC
            y_true_binary = (np.array(y_test) == 1).astype(int)
            roc_auc = roc_auc_score(y_true_binary, scores)
            results[name] = roc_auc
            print(f"[{name}] ROC-AUC: {roc_auc:.4f}")
            
        return results