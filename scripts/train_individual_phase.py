# scripts/train_individual_phase.py

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

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_phase1_ood(config):
    logger = setup_logger(name="phase1_logger", log_file="logs/phase1_ood.log")
    metrics = {"training_times_seconds": {}, "phase1_ood_roc_auc": {}}
    models_dir = config['paths']['models_dir']
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Entrenamiento Fase 1 (Detección OOD)", logger, metrics["training_times_seconds"], "phase1_ood"):
            df_ood = pd.read_csv(os.path.join(config["paths"]["test_ood_data"], "phase1_ood_data.csv"))
            X_valid = df_ood[df_ood['label'] == 1][spec_cols].values
            
            scaler_specs = StandardScaler()
            X_valid_scaled = scaler_specs.fit_transform(X_valid)
            joblib.dump(scaler_specs, os.path.join(models_dir, 'scaler_specs.pkl'))
            
            ood_benchmark = OODDetectorBenchmark(random_state=config['project']['seed'])
            ood_benchmark.train_baselines(X_valid_scaled)
            joblib.dump(ood_benchmark.fitted_models, os.path.join(models_dir, 'phase1_ood_models.pkl'))
            
            X_test_scaled = scaler_specs.transform(df_ood[spec_cols].values)
            roc_results = ood_benchmark.evaluate(X_test_scaled, df_ood['label'].values)
            metrics["phase1_ood_roc_auc"] = roc_results
            
            for model_name, roc in roc_results.items():
                logger.info(f"  -> {model_name} ROC-AUC: {roc:.4f}")
    except Exception as e:
        logger.error("Error en Fase 1", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/phase1_metrics.json")

def run_phase2_router(config):
    logger = setup_logger(name="phase2_logger", log_file="logs/phase2_router.log")
    metrics = {"training_times_seconds": {}, "phase2_router_metrics": {}}
    models_dir = config['paths']['models_dir']
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Entrenamiento Fase 2 (Stacking Router)", logger, metrics["training_times_seconds"], "phase2_router"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
            
            X_router = scaler_specs.transform(df_unified[spec_cols].values)
            y_router = df_unified['topology_id'].values
            
            router = TopologicalRouter(random_state=config['project']['seed'], 
                                       lambda_penalty=config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda'])
            router.train(X_router, y_router)
            joblib.dump(router.model, os.path.join(models_dir, 'phase2_router_model.pkl'))
            
            y_pred_proba = router.model.predict_proba(X_router)
            train_log_loss = log_loss(y_router, y_pred_proba)
            metrics["phase2_router_metrics"]["train_log_loss"] = train_log_loss
            logger.info(f"  -> Stacking Ensemble Train Log-Loss: {train_log_loss:.4f}")
    except Exception as e:
        logger.error("Error en Fase 2", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/phase2_metrics.json")

def prepare_data_phase3_4(config, df_unified, models_dir, spec_cols, logger):
    """Función auxiliar para preparar y escalar el super-vector de diseño."""
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    scaler_dv = StandardScaler()
    df_unified_scaled = df_unified.copy()
    df_unified_scaled[dv_cols] = scaler_dv.fit_transform(df_unified[dv_cols])
    df_unified_scaled[spec_cols] = scaler_specs.transform(df_unified[spec_cols])
    joblib.dump(scaler_dv, os.path.join(models_dir, 'scaler_design_vars.pkl'))
    
    return df_unified_scaled, super_vector_dim

def run_phase3_mdn(config, device):
    logger = setup_logger(name="phase3_logger", log_file="logs/phase3_mdn.log")
    metrics = {"training_times_seconds": {}, "phase3_generative_loss": {}}
    models_dir = config['paths']['models_dir']
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Entrenamiento Fase 3 (MDN)", logger, metrics["training_times_seconds"], "phase3_mdn"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            df_unified_scaled, super_vector_dim = prepare_data_phase3_4(config, df_unified, models_dir, spec_cols, logger)
            
            dataset_phase3 = Phase3GenerativeDataset(df_unified_scaled, spec_cols)
            dataloader_p3 = DataLoader(dataset_phase3, batch_size=config['phases_benchmarks']['phase3_generative']['batch_size'], shuffle=True)
            
            mdn_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim)
            criterion_p3 = MaskedNLLLoss()
            trainer_p3 = GenericTrainer(mdn_model, criterion_p3, lr=1e-3, device=device)
            
            epochs_p3 = config['phases_benchmarks']['phase3_generative']['epochs']
            final_loss_p3, _ = trainer_p3.train_full(dataloader_p3, epochs_p3, is_masked=True, logger=logger, phase_name="Fase 3")
            
            metrics["phase3_generative_loss"]["final_masked_nll"] = final_loss_p3
            trainer_p3.save_model(os.path.join(models_dir, 'phase3_mdn.pth'))
    except Exception as e:
        logger.error("Error en Fase 3", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/phase3_metrics.json")

def run_phase4_surrogate(config, device):
    logger = setup_logger(name="phase4_logger", log_file="logs/phase4_surrogate.log")
    metrics = {"training_times_seconds": {}, "phase4_surrogate_loss": {}}
    models_dir = config['paths']['models_dir']
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Entrenamiento Fase 4 (MLP Surrogate)", logger, metrics["training_times_seconds"], "phase4_surrogate"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            df_unified_scaled, super_vector_dim = prepare_data_phase3_4(config, df_unified, models_dir, spec_cols, logger)
            
            dataset_phase4 = Phase4SurrogateDataset(df_unified_scaled, spec_cols)
            dataloader_p4 = DataLoader(dataset_phase4, batch_size=config['phases_benchmarks']['phase4_surrogate']['batch_size'], shuffle=True)
            
            mlp_surrogate = SurrogateMLP(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols))
            criterion_p4 = PINNSurrogateLoss(lambda_physics=0.1)
            trainer_p4 = GenericTrainer(mlp_surrogate, criterion_p4, lr=1e-3, device=device)
            
            epochs_p4 = config['phases_benchmarks']['phase4_surrogate']['epochs']
            final_loss_p4, _ = trainer_p4.train_full(dataloader_p4, epochs_p4, is_masked=False, logger=logger, phase_name="Fase 4")
            
            metrics["phase4_surrogate_loss"]["final_surrogate_loss"] = final_loss_p4
            trainer_p4.save_model(os.path.join(models_dir, 'phase4_surrogate.pth'))
    except Exception as e:
        logger.error("Error en Fase 4", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/phase4_metrics.json")


def main():
    config = load_config()
    set_global_seeds(config['project']['seed'])
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando ejecutor de fases individuales. Dispositivo: {device}")
    
    # ==========================================
    # DESCOMENTA LA FASE QUE QUIERAS EJECUTAR
    # ==========================================
    
    # run_phase1_ood(config)
    # run_phase2_router(config)
    # run_phase3_mdn(config, device)
    # run_phase4_surrogate(config, device)

if __name__ == "__main__":
    main()