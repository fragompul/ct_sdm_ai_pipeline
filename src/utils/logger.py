# src/utils/logger.py

import os
import random
import numpy as np
import torch
import logging
import time
import json
import traceback

def set_global_seeds(seed=42):
    """Fija todas las semillas para garantizar reproducibilidad exacta."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Semillas globales fijadas en: {seed}")

def setup_logger(name="ct_sdm_pipeline", log_file="logs/execution.log"):
    """Configura un logger estándar con formato detallado."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevenir que se añadan múltiples handlers si se llama varias veces
    if not logger.handlers:
        # Formato profesional: Fecha | Nivel | Mensaje
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

class Timer:
    """Context Manager para registrar automáticamente tiempos de ejecución y errores."""
    def __init__(self, process_name, logger, metrics_dict=None, metric_key=None):
        self.process_name = process_name
        self.logger = logger
        self.metrics_dict = metrics_dict
        self.metric_key = metric_key

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"[INICIO] {self.process_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"[FIN] {self.process_name} completado en {elapsed_time:.4f} segundos.")
            # Guardar el tiempo en el diccionario de métricas si se proporcionó
            if self.metrics_dict is not None and self.metric_key is not None:
                self.metrics_dict[self.metric_key] = elapsed_time
        else:
            # Capturar toda la traza del error en el log
            error_trace = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.logger.error(f"[ERROR CATASTRÓFICO] Fallo en {self.process_name} después de {elapsed_time:.4f} segundos.")
            self.logger.error(f"Traza del error:\n{error_trace}")
        
        # Devolver False permite que la excepción se siga propagando y detenga el programa
        return False

def save_metrics_to_json(metrics_dict, filepath="logs/metrics_report.json"):
    """Guarda todas las métricas en un JSON estructurado para el paper."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)