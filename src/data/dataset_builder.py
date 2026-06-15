# src/data/dataset_builder.py

import os
import glob
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

class CTSDMDatasetBuilder:
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.raw_path = self.config["paths"]["raw_data"]
        self.processed_path = self.config["paths"]["processed_data"]
        self.test_ood_path = self.config["paths"]["test_ood_data"]
        self.specs = self.config["data"]["input_specs"]
        
        # Diccionarios internos para el procesamiento
        self.topology_map = {} # Mapeo de nombre de archivo a ID (1-12)
        self.all_design_vars = set() # El conjunto global de variables D
        self.raw_dataframes = {}
        
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.test_ood_path, exist_ok=True)

    def _parse_filename(self, filename: str) -> dict:
        """Extrae la información de la topología del nombre del archivo (ej. FB_2_Active_RC.csv)"""
        basename = os.path.basename(filename).replace(".csv", "")
        parts = basename.split("_")
        return {
            "form": parts[0],             # FB o FF
            "order": int(parts[1]),       # 2, 3 o 4
            "implementation": f"{parts[2]}_{parts[3]}" # Active_RC o Gm_C
        }

    def load_and_scan_variables(self):
        """Paso 1: Leer todos los CSVs y descubrir TODAS las variables de diseño posibles."""
        files = glob.glob(os.path.join(self.raw_path, "*.csv"))
        if len(files) != 12:
            print(f"Advertencia: Se esperaban 12 datasets, pero se encontraron {len(files)}.")

        for idx, file in enumerate(files):
            top_id = idx + 1
            df = pd.read_csv(file)
            name = os.path.basename(file).replace(".csv", "")
            
            self.topology_map[top_id] = name
            self.raw_dataframes[top_id] = df
            
            # Asumimos que las columnas que NO son specs y NO son métricas finales, son variables de diseño
            non_design_cols = self.specs + self.config["data"]["target_metrics"]
            design_vars = [col for col in df.columns if col not in non_design_cols]
            self.all_design_vars.update(design_vars)
            
        self.all_design_vars = sorted(list(self.all_design_vars))
        print(f"Súper-Vector D definido con {len(self.all_design_vars)} variables únicas.")

    def generate_phase1_dataset(self, full_valid_df: pd.DataFrame):
        """Genera datos válidos y anomalías sintéticas para la detección OOD (Fase 1)."""
        valid_specs = full_valid_df[self.specs].copy()
        valid_specs['label'] = 1 # 1 = Válido
        
        # Crear anomalías sintéticas (ej. SNDR altísimo con Power bajísimo)
        anomalies = pd.DataFrame({
            "SNDR": np.random.uniform(120, 150, size=5000),
            "Bw": np.random.uniform(10e6, 50e6, size=5000),
            "Power": np.random.uniform(0.1, 0.5, size=5000), # mW (imposible para ese SNDR)
            "label": -1 # -1 = OOD (Convención de sklearn para anomalías)
        })
        
        phase1_data = pd.concat([valid_specs, anomalies]).sample(frac=1).reset_index(drop=True)
        phase1_data.to_csv(os.path.join(self.test_ood_path, "phase1_ood_data.csv"), index=False)
        print("Dataset Fase 1 (OOD Guardrails) generado.")

    def build_unified_datasets(self):
        """Paso 2: Construye los datasets para las Fases 2, 3 y 4 aplicando Zero-Padding y Máscaras."""
        unified_rows = []
        
        for top_id, df in self.raw_dataframes.items():
            for _, row in df.iterrows():
                data_point = {
                    "topology_id": top_id
                }
                
                # 1. Añadir Especificaciones y Métricas
                for col in self.specs + self.config["data"]["target_metrics"]:
                    if col in row:
                        data_point[col] = row[col]
                
                # 2. Construir Súper-Vector (Zero-Padding) y Máscara
                for var in self.all_design_vars:
                    if var in row:
                        data_point[f"dv_{var}"] = row[var]
                        data_point[f"mask_{var}"] = 1.0  # Variable pertinente a la topología
                    else:
                        data_point[f"dv_{var}"] = 0.0    # Zero-padding
                        data_point[f"mask_{var}"] = 0.0  # Máscara para evitar gradiente
                        
                unified_rows.append(data_point)
                
        full_df = pd.DataFrame(unified_rows)
        
        # Guardar dataset global
        full_df.to_csv(os.path.join(self.processed_path, "unified_ct_sdm_dataset.csv"), index=False)
        print("Dataset Unificado (Fases 2, 3 y 4) generado con Zero-Padding y Boolean Masks.")
        
        # Generar dataset de Fase 1
        self.generate_phase1_dataset(full_df)

if __name__ == "__main__":
    builder = CTSDMDatasetBuilder()
    builder.load_and_scan_variables()
    builder.build_unified_datasets()