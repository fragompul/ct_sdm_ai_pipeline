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
from sklearn.metrics import log_loss

sys.path.append('src')

from utils.logger import set_global_seeds, setup_logger, Timer, save_metrics_to_json
from models.phase1_ood import OODDetectorBenchmark
from models.phase2_router import TopologicalRouter
from models.phase3_gen import MixtureDensityNetwork
from models.phase4_surrogate import SurrogateMLP, PINNSurrogateLoss
from data.dataloaders import Phase3GenerativeDataset, Phase4SurrogateDataset
from training.custom_losses import MaskedNLLLoss
from training.trainer import GenericTrainer

def main():
    logger = setup_logger()
    
    # Mega-diccionario para el paper
    paper_metrics = {
        "hardware": {},
        "training_times_seconds": {},
        "phase1_ood_roc_auc": {},
        "phase2_router_metrics": {},
        "phase3_generative_loss": {},
        "phase4_surrogate_loss": {}
    }

    try:
        with open("configs/default_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        set_global_seeds(config['project']['seed'])
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paper_metrics["hardware"]["device"] = str(device)
        logger.info(f"Iniciando Pipeline. Dispositivo detectado: {device}")
        
        spec_cols = config["data"]["input_specs"]

        # --- PASO 1: FASE 1 (OOD GUARDRAILS) ---
        with Timer("Entrenamiento Fase 1 (Detección OOD)", logger, paper_metrics["training_times_seconds"], "phase1_ood"):
            df_ood = pd.read_csv(os.path.join(config["paths"]["test_ood_data"], "phase1_ood_data.csv"))
            X_valid = df_ood[df_ood['label'] == 1][spec_cols].values
            
            scaler_specs = StandardScaler()
            X_valid_scaled = scaler_specs.fit_transform(X_valid)
            joblib.dump(scaler_specs, os.path.join(models_dir, 'scaler_specs.pkl'))
            
            ood_benchmark = OODDetectorBenchmark(random_state=config['project']['seed'])
            ood_benchmark.train_baselines(X_valid_scaled)
            joblib.dump(ood_benchmark.fitted_models, os.path.join(models_dir, 'phase1_ood_models.pkl'))
            
            # Evaluar y registrar métricas
            X_test_scaled = scaler_specs.transform(df_ood[spec_cols].values)
            roc_results = ood_benchmark.evaluate(X_test_scaled, df_ood['label'].values)
            paper_metrics["phase1_ood_roc_auc"] = roc_results
            
            for model_name, roc in roc_results.items():
                logger.info(f"  -> {model_name} ROC-AUC: {roc:.4f}")

        # --- PASO 2: FASE 2 (ROUTER PROBABILÍSTICO) ---
        with Timer("Entrenamiento Fase 2 (Stacking Router)", logger, paper_metrics["training_times_seconds"], "phase2_router"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            X_router = scaler_specs.transform(df_unified[spec_cols].values)
            y_router = df_unified['topology_id'].values
            
            router = TopologicalRouter(random_state=config['project']['seed'], 
                                       lambda_penalty=config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda'])
            router.train(X_router, y_router)
            joblib.dump(router.model, os.path.join(models_dir, 'phase2_router_model.pkl'))
            
            # Calcular Log-Loss en training para el log (simulado sobre el mismo train por simplificación)
            y_pred_proba = router.model.predict_proba(X_router)
            train_log_loss = log_loss(y_router, y_pred_proba)
            paper_metrics["phase2_router_metrics"]["train_log_loss"] = train_log_loss
            logger.info(f"  -> Stacking Ensemble Train Log-Loss: {train_log_loss:.4f}")

        # --- PREPARATIVOS ESCALADO ---
        with Timer("Escalado de Variables de Diseño", logger):
            dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
            super_vector_dim = len(dv_cols)
            logger.info(f"Dimensión del Súper-Vector: {super_vector_dim}")
            
            scaler_dv = StandardScaler()
            df_unified_scaled = df_unified.copy()
            df_unified_scaled[dv_cols] = scaler_dv.fit_transform(df_unified[dv_cols])
            df_unified_scaled[spec_cols] = scaler_specs.transform(df_unified[spec_cols])
            joblib.dump(scaler_dv, os.path.join(models_dir, 'scaler_design_vars.pkl'))

        # --- PASO 3: FASE 3 (UNIFIED GENERATIVE SPACE - MDN) ---
        with Timer("Entrenamiento Fase 3 (MDN)", logger, paper_metrics["training_times_seconds"], "phase3_mdn"):
            dataset_phase3 = Phase3GenerativeDataset(df_unified_scaled, spec_cols)
            dataloader_p3 = DataLoader(dataset_phase3, batch_size=config['phases_benchmarks']['phase3_generative']['batch_size'], shuffle=True)
            
            mdn_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim)
            criterion_p3 = MaskedNLLLoss()
            trainer_p3 = GenericTrainer(mdn_model, criterion_p3, lr=1e-3, device=device)
            
            epochs_p3 = config['phases_benchmarks']['phase3_generative']['epochs']
            final_loss_p3, _ = trainer_p3.train_full(dataloader_p3, epochs_p3, is_masked=True, logger=logger, phase_name="Fase 3")
            
            paper_metrics["phase3_generative_loss"]["final_masked_nll"] = final_loss_p3
            trainer_p3.save_model(os.path.join(models_dir, 'phase3_mdn.pth'))

        # --- PASO 4: FASE 4 (SURROGATE OPTIMIZATION) ---
        with Timer("Entrenamiento Fase 4 (MLP Surrogate)", logger, paper_metrics["training_times_seconds"], "phase4_surrogate"):
            dataset_phase4 = Phase4SurrogateDataset(df_unified_scaled, spec_cols)
            dataloader_p4 = DataLoader(dataset_phase4, batch_size=config['phases_benchmarks']['phase4_surrogate']['batch_size'], shuffle=True)
            
            mlp_surrogate = SurrogateMLP(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols))
            criterion_p4 = PINNSurrogateLoss(lambda_physics=0.1)
            trainer_p4 = GenericTrainer(mlp_surrogate, criterion_p4, lr=1e-3, device=device)
            
            epochs_p4 = config['phases_benchmarks']['phase4_surrogate']['epochs']
            final_loss_p4, _ = trainer_p4.train_full(dataloader_p4, epochs_p4, is_masked=False, logger=logger, phase_name="Fase 4")
            
            paper_metrics["phase4_surrogate_loss"]["final_surrogate_loss"] = final_loss_p4
            trainer_p4.save_model(os.path.join(models_dir, 'phase4_surrogate.pth'))

        logger.info("¡Entrenamiento finalizado con éxito!")

    except Exception as e:
        logger.error("El pipeline se detuvo debido a un error crítico.")
    
    finally:
        # Esto se ejecuta SIEMPRE, haya error o no, para salvar lo que se haya medido hasta el momento del fallo
        save_metrics_to_json(paper_metrics)
        logger.info("Métricas guardadas en logs/metrics_report.json")

if __name__ == "__main__":
    main()