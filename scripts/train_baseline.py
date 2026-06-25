# scripts/train_baseline.py

import sys
import yaml
import os
import time
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append('src')
from utils.logger import set_global_seeds, setup_logger, Timer, save_metrics_to_json

class BaselineRNN(nn.Module):
    """RNN específica para una única arquitectura [cite: 1094, 1324-1327]."""
    def __init__(self, input_dim, output_dim, hidden_units=64):
        super(BaselineRNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    set_global_seeds(config['project']['seed'])
    models_dir = os.path.join(config['paths']['models_dir'], "baseline")
    os.makedirs(models_dir, exist_ok=True)
    
    logger = setup_logger(name="baseline_logger", log_file="logs/baseline_training.log")
    metrics = {"training_times_seconds": {}, "classifier_metrics": {}, "rnns_metrics": {}}
    spec_cols = config["data"]["input_specs"]
    target_metrics = config["data"]["target_metrics"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando Entrenamiento del BASELINE. Dispositivo: {device}")

    try:
        raw_files = glob.glob(os.path.join(config['paths']['raw_data'], "*.csv"))
        all_data = []
        mapping_info = {}
        
        # 1. Recopilar datos globales para el clasificador [cite: 1290]
        for idx, file in enumerate(raw_files):
            top_id = idx + 1
            df = pd.read_csv(file)
            df['topology_id'] = top_id
            name = os.path.basename(file).replace(".csv", "")
            mapping_info[top_id] = name
            
            # Extraer solo las specs y el target para el dataset global
            all_data.append(df[spec_cols + ['topology_id']])
            
        global_df = pd.concat(all_data, ignore_index=True)
        
        # 2. Entrenar el Clasificador Determinista (Gradient Boosting) [cite: 1398, 1452]
        with Timer("Entrenamiento Clasificador Baseline", logger, metrics["training_times_seconds"], "classifier"):
            X_clf = global_df[spec_cols].values
            y_clf = global_df['topology_id'].values
            
            # Downsample opcional para balanceo (como menciona el paper) se asume balanceado si cada csv tiene 50k
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
            
            scaler_clf = StandardScaler()
            X_train_c_sc = scaler_clf.fit_transform(X_train_c)
            X_test_c_sc = scaler_clf.transform(X_test_c)
            joblib.dump(scaler_clf, os.path.join(models_dir, 'baseline_clf_scaler.pkl'))
            
            clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_c_sc, y_train_c)
            joblib.dump(clf, os.path.join(models_dir, 'baseline_classifier.pkl'))
            
            preds_c = clf.predict(X_test_c_sc)
            acc = accuracy_score(y_test_c, preds_c)
            metrics["classifier_metrics"]["Accuracy"] = acc
            logger.info(f"Clasificador Baseline entrenado. Accuracy: {acc:.4f}")

        # 3. Entrenar 12 Redes de Regresión Independientes (RNNs) 
        with Timer("Entrenamiento de las 12 RNNs independientes", logger, metrics["training_times_seconds"], "rnns_total"):
            for idx, file in enumerate(raw_files):
                top_id = idx + 1
                name = mapping_info[top_id]
                logger.info(f"--- Entrenando RNN específica para ID {top_id} ({name}) ---")
                
                df = pd.read_csv(file)
                # Las variables de diseño son todo lo que no es spec ni métrica de salida
                dv_cols = [c for c in df.columns if c not in spec_cols and c not in target_metrics]
                
                X_rnn = df[spec_cols].values
                y_rnn = df[dv_cols].values
                
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rnn, y_rnn, test_size=0.2, random_state=42)
                
                scaler_X_rnn = StandardScaler()
                scaler_y_rnn = StandardScaler()
                
                X_train_r_sc = scaler_X_rnn.fit_transform(X_train_r)
                y_train_r_sc = scaler_y_rnn.fit_transform(y_train_r)
                X_test_r_sc = scaler_X_rnn.transform(X_test_r)
                y_test_r_sc = scaler_y_rnn.transform(y_test_r)
                
                joblib.dump(scaler_X_rnn, os.path.join(models_dir, f'baseline_scaler_X_{top_id}.pkl'))
                joblib.dump(scaler_y_rnn, os.path.join(models_dir, f'baseline_scaler_y_{top_id}.pkl'))
                joblib.dump(dv_cols, os.path.join(models_dir, f'baseline_features_{top_id}.pkl'))
                
                # PyTorch Dataset
                train_loader = DataLoader(TensorDataset(torch.tensor(X_train_r_sc, dtype=torch.float32), 
                                                        torch.tensor(y_train_r_sc, dtype=torch.float32)), 
                                          batch_size=256, shuffle=True)
                
                rnn_model = BaselineRNN(input_dim=len(spec_cols), output_dim=len(dv_cols)).to(device)
                optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                epochs = 100
                start_rnn = time.time()
                for epoch in range(epochs):
                    rnn_model.train()
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        loss = criterion(rnn_model(batch_x), batch_y)
                        loss.backward()
                        optimizer.step()
                
                rnn_time = time.time() - start_rnn
                
                # Evaluación
                rnn_model.eval()
                with torch.no_grad():
                    preds_r_sc = rnn_model(torch.tensor(X_test_r_sc, dtype=torch.float32).to(device)).cpu().numpy()
                    mse = mean_squared_error(y_test_r_sc, preds_r_sc)
                    
                torch.save(rnn_model.state_dict(), os.path.join(models_dir, f'baseline_rnn_{top_id}.pth'))
                
                logger.info(f"[RNN {top_id}] Entrenada en {rnn_time:.2f}s | Test MSE: {mse:.4f}")
                metrics["rnns_metrics"][f"RNN_{top_id}"] = {"MSE": mse, "Time": rnn_time, "Num_Vars": len(dv_cols)}
                
        logger.info("¡Entrenamiento del Baseline finalizado con éxito!")

    except Exception as e:
        logger.error("Error crítico en el entrenamiento del baseline.", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/baseline_metrics_report.json")

if __name__ == "__main__":
    main()