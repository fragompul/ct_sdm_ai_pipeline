# src/data/dataloaders.py

import torch
from torch.utils.data import Dataset
import numpy as np

class Phase3GenerativeDataset(Dataset):
    """Dataset para la Fase 3 (MDN / cVAE)."""
    def __init__(self, df, spec_cols, num_topologies=12):
        # 1. Especificaciones (escaladas previamente)
        self.specs = torch.tensor(df[spec_cols].values, dtype=torch.float32)
        
        # 2. Topología a One-Hot
        topology_ids = torch.tensor(df['topology_id'].values - 1, dtype=torch.long) # Asume IDs 1-12
        self.topology_onehot = torch.nn.functional.one_hot(topology_ids, num_classes=num_topologies).float()
        
        # Condición = Specs + OneHot
        self.cond = torch.cat([self.specs, self.topology_onehot], dim=1)
        
        # 3. Super-Vector de Diseño (Columnas dv_)
        dv_cols = sorted([c for c in df.columns if c.startswith('dv_')])
        self.y_design = torch.tensor(df[dv_cols].values, dtype=torch.float32)
        
        # 4. Máscaras Booleanas (Columnas mask_)
        mask_cols = sorted([c for c in df.columns if c.startswith('mask_')])
        self.masks = torch.tensor(df[mask_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, idx):
        return {
            'cond': self.cond[idx],
            'y': self.y_design[idx],
            'mask': self.masks[idx]
        }

class Phase4SurrogateDataset(Dataset):
    """Dataset para la Fase 4 (Proxy Surrogate)."""
    def __init__(self, df, spec_cols, num_topologies=12):
        # 1. Topología One-Hot
        topology_ids = torch.tensor(df['topology_id'].values - 1, dtype=torch.long)
        topology_onehot = torch.nn.functional.one_hot(topology_ids, num_classes=num_topologies).float()
        
        # 2. Super-Vector de Diseño
        dv_cols = sorted([c for c in df.columns if c.startswith('dv_')])
        y_design = torch.tensor(df[dv_cols].values, dtype=torch.float32)
        
        # Entrada Surrogate = OneHot + y_design
        self.x_surrogate = torch.cat([topology_onehot, y_design], dim=1)
        
        # 3. Métricas Objetivo (Usamos las specs como proxy del simulador)
        self.y_metrics = torch.tensor(df[spec_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x_surrogate)

    def __getitem__(self, idx):
        return {
            'x': self.x_surrogate[idx],
            'y': self.y_metrics[idx]
        }
