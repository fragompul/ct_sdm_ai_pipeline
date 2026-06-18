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
from sklearn.model_selection import train_test_split

sys.path.append('src')

from utils.logger import set_global_seeds, setup_logger, Timer, save_metrics_to_json
from utils.metrics_plotter import plot_roc_curve, plot_confusion_matrix, plot_training_curves
from models.phase1_ood import OODDetectorBenchmark
from models.phase2_router import TopologicalRouterBenchmark
from models.phase3_gen import MixtureDensityNetwork, ConditionalVAE, MCDropoutResNet, cGANGenerator, cGANDiscriminator, TabularDDPM
from models.phase4_surrogate import SurrogateMLP, PINNSurrogateLoss
from data.dataloaders import Phase3GenerativeDataset, Phase4SurrogateDataset
from training.custom_losses import MaskedNLLLoss, MaskedMSELoss, MaskedVAELoss
from training.trainer import GenerativeTrainer

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_phase1_ood(config):
    logger = setup_logger(name="phase1_logger", log_file="logs/phase1_ood.log")
    metrics = {"training_times_seconds": {}, "phase1_ood_metrics": {}}
    models_dir = config['paths']['models_dir']
    plots_dir = "logs/plots/phase1/"
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Optimizacion y Entrenamiento Fase 1 (Detección OOD)", logger, metrics["training_times_seconds"], "phase1_ood"):
            df_ood = pd.read_csv(os.path.join(config["paths"]["test_ood_data"], "phase1_ood_data.csv"))
            
            # 1. Separar datos válidos para entrenamiento y validación OOD
            valid_data = df_ood[df_ood['label'] == 1]
            anomalies = df_ood[df_ood['label'] == -1]
            
            X_valid_train, X_valid_val = train_test_split(valid_data[spec_cols].values, test_size=0.2, random_state=42)
            
            # Para la validación necesitamos mezclar anomalías para que Optuna pueda medir ROC-AUC
            X_val_combined = np.vstack((X_valid_val, anomalies[spec_cols].values))
            y_val_combined = np.array([1]*len(X_valid_val) + [-1]*len(anomalies))
            
            scaler = StandardScaler()
            X_valid_train_scaled = scaler.fit_transform(X_valid_train)
            X_val_combined_scaled = scaler.transform(X_val_combined)
            joblib.dump(scaler, os.path.join(models_dir, 'scaler_specs.pkl'))
            
            benchmark = OODDetectorBenchmark(random_state=config['project']['seed'])
            # HPO de 10 trials
            benchmark.train_all(X_valid_train_scaled, X_val_combined_scaled, y_val_combined, n_trials=10)
            
            # 2. Evaluar y Guardar Plots
            # Simulamos un Test Set para gráficas usando el Validation Set aquí por simplicidad
            results, y_true_binary = benchmark.evaluate_and_get_scores(X_val_combined_scaled, y_val_combined)
            
            best_model_name = None
            best_auc = 0
            
            for name, data in results.items():
                logger.info(f"[{name}] AUC: {data['metrics']['ROC-AUC']:.4f} | F1: {data['metrics']['F1']:.4f}")
                plot_roc_curve(y_true_binary, data["scores"], name, "Phase1", plots_dir)
                plot_confusion_matrix(y_true_binary, data["y_pred_bin"], name, "Phase1", plots_dir, class_names=["Anomaly", "Valid"])
                
                metrics["phase1_ood_metrics"][name] = data['metrics']
                if data['metrics']['ROC-AUC'] > best_auc:
                    best_auc = data['metrics']['ROC-AUC']
                    best_model_name = name
                    
            logger.info(f"Mejor Modelo Fase 1: {best_model_name} (AUC: {best_auc:.4f})")
            metrics["best_model"] = best_model_name
            metrics["timings_detailed"] = benchmark.timings
            
            # Guardamos el mejor modelo
            if best_model_name == "DenseAutoencoder":
                torch.save(benchmark.ae_model.state_dict(), os.path.join(models_dir, 'phase1_ood_ae.pth'))
            else:
                joblib.dump(benchmark.fitted_models[best_model_name], os.path.join(models_dir, 'phase1_ood_model.pkl'))
                
    except Exception as e:
        logger.error("Error en Fase 1", exc_info=True)
    finally:
        save_metrics_to_json(metrics, "logs/phase1_metrics.json")

def run_phase2_router(config):
    logger = setup_logger(name="phase2_logger", log_file="logs/phase2_router.log")
    metrics = {"training_times_seconds": {}, "phase2_router_metrics": {}}
    models_dir = config['paths']['models_dir']
    plots_dir = "logs/plots/phase2/"
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Optimizacion y Entrenamiento Fase 2 (Router)", logger, metrics["training_times_seconds"], "phase2_router"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
            
            X_all = scaler_specs.transform(df_unified[spec_cols].values)
            y_all = df_unified['topology_id'].values - 1 # Clases de 0 a 11
            
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
            
            benchmark = TopologicalRouterBenchmark(random_state=config['project']['seed'])
            
            # n_trials=10 es un buen balance para la Fase 2 masiva. 
            benchmark.optimize_and_train_all(X_train, y_train, X_val, y_val, n_trials=10)
            
            results = benchmark.evaluate_all(X_test, y_test)
            
            best_model_name = None
            best_logloss = float('inf')
            
            # Nombres extraídos para la Matriz de Confusión
            class_names = []
            for i in range(12):
                if (i+1) in benchmark.mapping_info:
                    class_names.append(benchmark.mapping_info[i+1]['name'])
                else:
                    class_names.append(f"Top_{i+1}")
            
            for name, data in results.items():
                logger.info(f"[{name}] Acc: {data['metrics']['Accuracy']:.4f} | Top3: {data['metrics']['Top3_Accuracy']:.4f} | LogLoss: {data['metrics']['LogLoss']:.4f}")
                
                plot_roc_curve(y_test, data["y_proba"], name, "Phase2", plots_dir, n_classes=12)
                plot_confusion_matrix(y_test, data["y_pred"], name, "Phase2", plots_dir, class_names=class_names)
                
                metrics["phase2_router_metrics"][name] = data['metrics']
                if data['metrics']['LogLoss'] < best_logloss:
                    best_logloss = data['metrics']['LogLoss']
                    best_model_name = name
                    
            logger.info(f"Mejor Modelo Fase 2: {best_model_name} (LogLoss: {best_logloss:.4f})")
            metrics["best_model"] = best_model_name
            metrics["timings_detailed"] = benchmark.timings
            
            joblib.dump(benchmark.trained_models[best_model_name], os.path.join(models_dir, 'phase2_router_model.pkl'))
            
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

def run_phase3_generative(config, device):
    logger = setup_logger(name="phase3_logger", log_file="logs/phase3_generative.log")
    metrics = {"training_times_seconds": {}, "phase3_generative_metrics": {}}
    models_dir = config['paths']['models_dir']
    plots_dir = "logs/plots/phase3/"
    spec_cols = config["data"]["input_specs"]
    
    try:
        with Timer("Optimizacion y Entrenamiento Fase 3 (Generative)", logger, metrics["training_times_seconds"], "phase3_total"):
            df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
            df_unified_scaled, super_vector_dim = prepare_data_phase3_4(config, df_unified, models_dir, spec_cols, logger)
            
            # Split Train/Val para HPO
            df_train, df_val = train_test_split(df_unified_scaled, test_size=0.15, random_state=42)
            train_loader = DataLoader(Phase3GenerativeDataset(df_train, spec_cols), batch_size=256, shuffle=True)
            val_loader = DataLoader(Phase3GenerativeDataset(df_val, spec_cols), batch_size=256, shuffle=False)
            
            model_names = ["MDN", "cVAE", "MCDropout", "TabularDDPM", "cGAN"]
            best_val_losses = {}
            best_params = {}
            
            for name in model_names:
                logger.info(f"--- Optimizando {name} ---")
                start_hpo = time.time()
                
                def objective(trial, current_name=name):
                    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
                    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
                    
                    if current_name == "MDN":
                        model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim)
                        criterion = MaskedNLLLoss()
                    elif current_name == "cVAE":
                        model = ConditionalVAE(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim)
                        criterion = MaskedVAELoss(beta=trial.suggest_float("beta", 0.1, 2.0))
                    elif current_name == "MCDropout":
                        model = MCDropoutResNet(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim, dropout_rate=trial.suggest_float("dropout", 0.1, 0.5))
                        criterion = MaskedMSELoss()
                    elif current_name == "TabularDDPM":
                        model = TabularDDPM(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim)
                        criterion = MaskedMSELoss()
                    elif current_name == "cGAN":
                        model = {
                            'G': cGANGenerator(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim),
                            'D': cGANDiscriminator(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=hidden_dim)
                        }
                        criterion = None # Manejado en Trainer
                    
                    trainer = GenerativeTrainer(current_name, model, criterion, lr=lr, device=device)
                    
                    # Entrenar pocas epochs para HPO rápido
                    for _ in range(20): trainer.train_epoch(train_loader)
                    
                    val_loss = trainer.eval_epoch(val_loader)
                    return val_loss if current_name != "cGAN" else trainer.train_epoch(train_loader) # Proxy para GAN

                study = optuna.create_study(direction="minimize")
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study.optimize(objective, n_trials=5) # 5 trials por modelo
                
                best_params[name] = study.best_params
                metrics["training_times_seconds"][f"{name}_hpo_time"] = time.time() - start_hpo
                
                # --- ENTRENAMIENTO FINAL COMPLETO ---
                start_train = time.time()
                bp = study.best_params
                
                if name == "MDN":
                    final_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim'])
                    final_crit = MaskedNLLLoss()
                elif name == "cVAE":
                    final_model = ConditionalVAE(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim'])
                    final_crit = MaskedVAELoss(beta=bp['beta'])
                elif name == "MCDropout":
                    final_model = MCDropoutResNet(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim'], dropout_rate=bp['dropout'])
                    final_crit = MaskedMSELoss()
                elif name == "TabularDDPM":
                    final_model = TabularDDPM(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim'])
                    final_crit = MaskedMSELoss()
                elif name == "cGAN":
                    final_model = {
                        'G': cGANGenerator(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim']),
                        'D': cGANDiscriminator(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=bp['hidden_dim'])
                    }
                    final_crit = None
                
                trainer = GenerativeTrainer(name, final_model, final_crit, lr=bp['lr'], device=device)
                
                train_losses, val_losses = [], []
                epochs = config['phases_benchmarks']['phase3_generative']['epochs'] # Ej. 200
                
                for epoch in range(1, epochs + 1):
                    t_loss = trainer.train_epoch(train_loader)
                    v_loss = trainer.eval_epoch(val_loader)
                    train_losses.append(t_loss)
                    if name != "cGAN": val_losses.append(v_loss)
                    
                    if epoch % 50 == 0:
                        logger.info(f"[{name}] Epoch {epoch}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
                        
                metrics["training_times_seconds"][f"{name}_train_time"] = time.time() - start_train
                
                # Guardar Plots y Modelos
                plot_training_curves(train_losses, val_losses, name, "Phase3", plots_dir)
                trainer.save_model(os.path.join(models_dir, f'phase3_{name.lower()}.pth'))
                
                best_val_losses[name] = val_losses[-1] if val_losses else t_loss
                metrics["phase3_generative_metrics"][name] = {"best_val_loss": best_val_losses[name], "params": bp}

            # Seleccionar el ganador global (para inferencia downstream)
            winner = min(best_val_losses, key=best_val_losses.get)
            logger.info(f"Mejor Modelo Fase 3: {winner} (Val Loss: {best_val_losses[winner]:.4f})")
            metrics["best_model"] = winner
            
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
    run_phase3_generative(config, device)
    # run_phase4_surrogate(config, device)

if __name__ == "__main__":
    main()