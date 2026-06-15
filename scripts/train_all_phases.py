# scripts/train_all_phases.py

import sys
import yaml
import os
sys.path.append('src')

from utils.logger import set_global_seeds, setup_logger
from data.dataset_builder import CTSDMDatasetBuilder

def main():
    # 1. Configuración inicial
    logger = setup_logger()
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_global_seeds(config['project']['seed'])
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    
    logger.info("Iniciando Pipeline de Entrenamiento Completo...")

    # 2. Construcción de Datos
    logger.info("--- PASO 1: Procesamiento de Datos ---")
    builder = CTSDMDatasetBuilder()
    builder.load_and_scan_variables()
    builder.build_unified_datasets()
    
    logger.info("--- PASO 2: Entrenamiento Fase 1 (OOD) ---")
    # Aquí cargarías pandas DataFrame de data/test_ood/
    # Instanciar OODDetectorBenchmark, hacer fit() y guardar con joblib/pickle
    logger.info("Fase 1 simulada: Modelos OOD entrenados.")

    logger.info("--- PASO 3: Entrenamiento Fase 2 (Router) ---")
    # Instanciar TopologicalRouter, hacer fit(X_specs, y_topology) y guardar con joblib
    logger.info("Fase 2 simulada: Stacking Ensemble entrenado.")

    logger.info("--- PASO 4: Entrenamiento Fase 3 (Generative) ---")
    # Crear PyTorch DataLoaders para el dataset unificado
    # Instanciar MaskedNLLLoss y MixtureDensityNetwork
    # Usar GenericTrainer para entrenar durante N epochs y guardar .pth
    logger.info("Fase 3 simulada: MDN unificada entrenada.")

    logger.info("--- PASO 5: Entrenamiento Fase 4 (Surrogate) ---")
    # Instanciar SurrogateMLP y PINNSurrogateLoss
    # Usar GenericTrainer y guardar .pth
    logger.info("Fase 4 simulada: Proxy Surrogate entrenado.")

    logger.info("¡Entrenamiento finalizado! Todos los modelos listos para inferencia.")

if __name__ == "__main__":
    main()