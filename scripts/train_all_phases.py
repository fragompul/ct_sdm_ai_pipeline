# scripts/train_all_phases.py

import sys
import yaml
import os
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append('src')

from utils.logger import set_global_seeds, setup_logger
from models.phase1_ood import OODDetectorBenchmark
from models.phase2_router import TopologicalRouter
from models.phase3_gen import MixtureDensityNetwork
from models.phase4_surrogate import SurrogateMLP, PINNSurrogateLoss
from data.dataloaders import Phase3GenerativeDataset, Phase4SurrogateDataset
from training.custom_losses import MaskedNLLLoss
from training.trainer import GenericTrainer

def main():
    # --- CONFIGURACIÓN INICIAL ---
    logger = setup_logger()
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_global_seeds(config['project']['seed'])
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando Pipeline de Entrenamiento. Dispositivo: {device}")
    
    spec_cols = config["data"]["input_specs"]

    # --- PASO 1: FASE 1 (OOD GUARDRAILS) ---
    logger.info("--- Entrenando Fase 1 (Detección OOD) ---")
    df_ood = pd.read_csv(os.path.join(config["paths"]["test_ood_data"], "phase1_ood_data.csv"))
    
    # Entrenar solo con datos físicamente válidos
    X_valid = df_ood[df_ood['label'] == 1][spec_cols].values
    
    # Escalar datos (Vital para SVM y GMM)
    scaler_specs = StandardScaler()
    X_valid_scaled = scaler_specs.fit_transform(X_valid)
    joblib.dump(scaler_specs, os.path.join(models_dir, 'scaler_specs.pkl'))
    
    ood_benchmark = OODDetectorBenchmark(random_state=config['project']['seed'])
    ood_benchmark.train_baselines(X_valid_scaled)
    joblib.dump(ood_benchmark.fitted_models, os.path.join(models_dir, 'phase1_ood_models.pkl'))
    
    # Evaluar con anomalías
    X_test_scaled = scaler_specs.transform(df_ood[spec_cols].values)
    ood_benchmark.evaluate(X_test_scaled, df_ood['label'].values)

    # --- PASO 2: FASE 2 (ROUTER PROBABILÍSTICO) ---
    logger.info("--- Entrenando Fase 2 (Stacking Meta-Ensemble Router) ---")
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    
    X_router = scaler_specs.transform(df_unified[spec_cols].values)
    y_router = df_unified['topology_id'].values
    
    router = TopologicalRouter(random_state=config['project']['seed'], 
                               lambda_penalty=config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda'])
    router.train(X_router, y_router)
    joblib.dump(router.model, os.path.join(models_dir, 'phase2_router_model.pkl'))

    # --- PREPARATIVOS PARA REDES NEURONALES (Fase 3 y 4) ---
    # Escalar las variables de diseño (Y) para que las redes converjan
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    logger.info(f"Dimensión del Súper-Vector detectada: {super_vector_dim}")
    
    scaler_dv = StandardScaler()
    # Entrenamos el scaler ignorando los ceros del padding para no sesgar la media
    # Como atajo seguro para el pipeline base, escalamos todo y la red aprenderá a mapear a 0
    df_unified_scaled = df_unified.copy()
    df_unified_scaled[dv_cols] = scaler_dv.fit_transform(df_unified[dv_cols])
    df_unified_scaled[spec_cols] = scaler_specs.transform(df_unified[spec_cols])
    joblib.dump(scaler_dv, os.path.join(models_dir, 'scaler_design_vars.pkl'))

    # --- PASO 3: FASE 3 (UNIFIED GENERATIVE SPACE - MDN) ---
    logger.info("--- Entrenando Fase 3 (Mixture Density Network con Masked Loss) ---")
    dataset_phase3 = Phase3GenerativeDataset(df_unified_scaled, spec_cols)
    dataloader_p3 = DataLoader(dataset_phase3, batch_size=config['phases_benchmarks']['phase3_generative']['batch_size'], shuffle=True)
    
    mdn_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim)
    criterion_p3 = MaskedNLLLoss()
    trainer_p3 = GenericTrainer(mdn_model, criterion_p3, lr=1e-3, device=device)
    
    epochs_p3 = config['phases_benchmarks']['phase3_generative']['epochs'] # Ej. 200
    for epoch in range(1, epochs_p3 + 1):
        loss = trainer_p3.train_epoch(dataloader_p3, is_masked=True)
        if epoch % 50 == 0:
            logger.info(f"[Fase 3] Epoch {epoch}/{epochs_p3} - Masked NLL Loss: {loss:.4f}")
            
    trainer_p3.save_model(os.path.join(models_dir, 'phase3_mdn.pth'))

    # --- PASO 4: FASE 4 (SURROGATE OPTIMIZATION) ---
    logger.info("--- Entrenando Fase 4 (Surrogate MLP PINN) ---")
    dataset_phase4 = Phase4SurrogateDataset(df_unified_scaled, spec_cols)
    dataloader_p4 = DataLoader(dataset_phase4, batch_size=config['phases_benchmarks']['phase4_surrogate']['batch_size'], shuffle=True)
    
    mlp_surrogate = SurrogateMLP(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols))
    criterion_p4 = PINNSurrogateLoss(lambda_physics=0.1)
    trainer_p4 = GenericTrainer(mlp_surrogate, criterion_p4, lr=1e-3, device=device)
    
    epochs_p4 = config['phases_benchmarks']['phase4_surrogate']['epochs'] # Ej. 500
    for epoch in range(1, epochs_p4 + 1):
        loss = trainer_p4.train_epoch(dataloader_p4, is_masked=False)
        if epoch % 100 == 0:
            logger.info(f"[Fase 4] Epoch {epoch}/{epochs_p4} - Surrogate Total Loss: {loss:.4f}")
            
    trainer_p4.save_model(os.path.join(models_dir, 'phase4_surrogate.pth'))

    logger.info("¡Entrenamiento finalizado! Modelos y Scalers guardados exitosamente en 'saved_models/'.")

if __name__ == "__main__":
    main()
