# src/models/phase1_ood.py

import numpy as np
import time
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class OODDetectorBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = {}
        self.fitted_models = {}
        self.timings = {}

    def _optimize_sklearn_models(self, X_train_valid, X_val, y_val, n_trials=10):
        """Usa Optuna para buscar hiperparámetros óptimos evaluando en validación."""
        def objective(trial, model_name):
            if model_name == "IsolationForest":
                cont = trial.suggest_float("contamination", 0.01, 0.2)
                n_est = trial.suggest_int("n_estimators", 50, 300)
                model = IsolationForest(contamination=cont, n_estimators=n_est, random_state=self.random_state)
            elif model_name == "OneClassSVM":
                nu = trial.suggest_float("nu", 0.01, 0.2)
                gamma = trial.suggest_categorical("gamma", ["scale", "auto", 0.1, 0.01])
                model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            elif model_name == "GMM":
                n_comp = trial.suggest_int("n_components", 1, 5)
                model = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=self.random_state)

            model.fit(X_train_valid)
            
            if model_name in ["IsolationForest", "OneClassSVM"]:
                scores = model.decision_function(X_val)
            else:
                scores = model.score_samples(X_val)
                
            y_true_binary = (np.array(y_val) == 1).astype(int)
            return roc_auc_score(y_true_binary, scores)

        for model_name in ["IsolationForest", "OneClassSVM", "GMM"]:
            start_hpo = time.time()
            study = optuna.create_study(direction="maximize")
            # Ocultamos logs de optuna para no ensuciar la terminal
            optuna.logging.set_verbosity(optuna.logging.WARNING) 
            study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
            self.timings[f"{model_name}_hpo_time"] = time.time() - start_hpo
            self.best_params[model_name] = study.best_params

    def train_all(self, X_train_valid, X_val, y_val, n_trials=10):
        # 1. Optimizar modelos de Scikit-Learn
        self._optimize_sklearn_models(X_train_valid, X_val, y_val, n_trials)
        
        # Entrenar finales
        self.fitted_models["IsolationForest"] = IsolationForest(**self.best_params["IsolationForest"], random_state=self.random_state)
        self.fitted_models["OneClassSVM"] = OneClassSVM(kernel='rbf', **self.best_params["OneClassSVM"])
        self.fitted_models["GMM"] = GaussianMixture(**self.best_params["GMM"], covariance_type='full', random_state=self.random_state)

        for name, model in self.fitted_models.items():
            start_train = time.time()
            model.fit(X_train_valid)
            self.timings[f"{name}_train_time"] = time.time() - start_train

        # 2. Entrenar Dense Autoencoder (PyTorch)
        start_ae = time.time()
        self.ae_model = DenseAutoencoder(input_dim=X_train_valid.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.ae_model.parameters(), lr=0.001)
        
        tensor_x = torch.tensor(X_train_valid, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.ae_model.train()
        for epoch in range(50): # Hardcodeado a 50 epochs para rapidez, se puede optimizar
            for batch_x, _ in loader:
                optimizer.zero_grad()
                loss = criterion(self.ae_model(batch_x), batch_x)
                loss.backward()
                optimizer.step()
                
        self.ae_model.eval()
        self.timings["DenseAutoencoder_train_time"] = time.time() - start_ae

    def evaluate_and_get_scores(self, X_test, y_test):
        """Devuelve scores continuos y predicciones binarias para métricas y gráficas."""
        results = {}
        y_true_binary = (np.array(y_test) == 1).astype(int) # 1 = Normal, 0 (-1) = Anomalía

        for name, model in self.fitted_models.items():
            start_inf = time.time()
            if name in ["IsolationForest", "OneClassSVM"]:
                scores = model.decision_function(X_test)
                y_pred = model.predict(X_test)
                # Sklearn predict da 1 (normal) y -1 (anomalía). Mapeamos a 1 y 0
                y_pred_bin = (y_pred == 1).astype(int) 
            elif name == "GMM":
                scores = model.score_samples(X_test)
                # Umbral heurístico para GMM (percentil 5 de train sería ideal, aquí usamos un threshold simple)
                threshold = np.percentile(scores, 5) 
                y_pred_bin = (scores >= threshold).astype(int)
            
            self.timings[f"{name}_inference_time"] = time.time() - start_inf
            
            # Calcular métricas OOD
            auc_val = roc_auc_score(y_true_binary, scores)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_bin, average='binary', zero_division=0)
            
            results[name] = {
                "scores": scores, "y_pred_bin": y_pred_bin,
                "metrics": {"ROC-AUC": auc_val, "Precision": prec, "Recall": rec, "F1": f1}
            }

        # Evaluar Autoencoder
        start_inf_ae = time.time()
        tensor_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            reconstructions = self.ae_model(tensor_test)
            # El score es el -MSE (Mayor = Más normal, Menor/Negativo = Anomalía)
            mse_scores = -torch.mean((tensor_test - reconstructions)**2, dim=1).numpy()
            
        threshold_ae = np.percentile(mse_scores, 5)
        y_pred_bin_ae = (mse_scores >= threshold_ae).astype(int)
        self.timings["DenseAutoencoder_inference_time"] = time.time() - start_inf_ae
        
        auc_ae = roc_auc_score(y_true_binary, mse_scores)
        prec_ae, rec_ae, f1_ae, _ = precision_recall_fscore_support(y_true_binary, y_pred_bin_ae, average='binary', zero_division=0)
        
        results["DenseAutoencoder"] = {
            "scores": mse_scores, "y_pred_bin": y_pred_bin_ae,
            "metrics": {"ROC-AUC": auc_ae, "Precision": prec_ae, "Recall": rec_ae, "F1": f1_ae}
        }

        return results, y_true_binary