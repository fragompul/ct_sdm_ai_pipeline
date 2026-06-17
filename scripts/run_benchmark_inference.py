# scripts/run_benchmark_inference.py

import sys
import yaml
import os
import joblib
import torch
import numpy as np
import pandas as pd
import time

sys.path.append('src')
from utils.logger import setup_logger
from models.phase3_gen import MixtureDensityNetwork
from models.phase4_surrogate import SurrogateMLP
from optimization.gradient_ascent import DifferentiableSearch

def load_mask_for_topology(topology_id, df_unified):
    row = df_unified[df_unified['topology_id'] == topology_id].iloc[0]
    mask_cols = sorted([c for c in df_unified.columns if c.startswith('mask_')])
    return row[mask_cols].values.astype(np.float32)

def main():
    logger = setup_logger(name="benchmark_logger", log_file="logs/benchmark_inference.log")
    logger.info("INICIANDO BENCHMARK DE INFERENCIA AUTOMÁTICO")
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    models_dir = config['paths']['models_dir']
    device = torch.device("cpu")
    
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    spec_cols = config["data"]["input_specs"]
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    scaler_dv = joblib.load(os.path.join(models_dir, 'scaler_design_vars.pkl'))
    ood_models = joblib.load(os.path.join(models_dir, 'phase1_ood_models.pkl'))
    router = joblib.load(os.path.join(models_dir, 'phase2_router_model.pkl'))
    
    mdn_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim)
    mdn_model.load_state_dict(torch.load(os.path.join(models_dir, 'phase3_mdn.pth'), map_location=device))
    mdn_model.eval()
    
    surrogate_model = SurrogateMLP(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols))
    surrogate_model.load_state_dict(torch.load(os.path.join(models_dir, 'phase4_surrogate.pth'), map_location=device))
    search_algo = DifferentiableSearch(surrogate_model, scaler_specs, device=device)
    
    lambda_penalty = config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda']
    cost_matrix = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.5, 1.1, 1.3, 2.1, 2.3, 3.2, 3.6])
    
    N_TESTS = 100
    successful_routes = 0
    start_total = time.time()
    
    for i in range(N_TESTS):
        # Generar specs aleatorias lógicas
        sndr_req = np.random.uniform(60.0, 120.0)
        bw_req = np.random.uniform(1e6, 50e6)
        power_req = np.random.uniform(0.5e-3, 10e-3)
        
        x_input = np.array([[sndr_req, bw_req, power_req]])
        x_input_scaled = scaler_specs.transform(x_input)
        
        logger.info(f"Test {i+1}/{N_TESTS} | Specs: SNDR={sndr_req:.1f}, Bw={bw_req/1e6:.1f}M, Pwr={power_req*1000:.1f}m")
        
        # Fase 1
        is_valid = ood_models['OneClassSVM'].predict(x_input_scaled)[0]
        if is_valid != 1:
            logger.warning(f"  -> Test {i+1} descartado por OOD Guardrails.")
            continue
            
        successful_routes += 1
            
        # Fase 2
        raw_probs = router.predict_proba(x_input_scaled)[0]
        adj_probs = (raw_probs * np.exp(-lambda_penalty * cost_matrix))
        adj_probs /= np.sum(adj_probs)
        top3_indices = np.argsort(adj_probs)[-3:][::-1]
        
        logger.info(f"  -> Top-3 IDs Seleccionados: {[idx+1 for idx in top3_indices]}")
        
        # Fases 3 & 4
        for top_idx in top3_indices:
            topology_id = top_idx + 1
            one_hot = np.zeros((1, 12))
            one_hot[0, top_idx] = 1
            one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
            cond_t = torch.cat([torch.tensor(x_input_scaled, dtype=torch.float32), one_hot_t], dim=1)
            
            with torch.no_grad():
                pi, mu, sigma = mdn_model(cond_t)
                best_mixture_idx = torch.argmax(pi, dim=1)
                initial_y_scaled = mu[0, best_mixture_idx, :].view(1, -1)
                
            mask = load_mask_for_topology(topology_id, df_unified)
            _, best_fom = search_algo.optimize(initial_y_scaled, one_hot_t, mask, steps=50) # Menos steps por rapidez
            
            logger.info(f"     * Top {topology_id} optimizada. FoMs Predictivo: {best_fom:.2f} dB")
            
    total_time = time.time() - start_total
    logger.info("="*50)
    logger.info(f"BENCHMARK COMPLETADO.")
    logger.info(f"Total pruebas: {N_TESTS} | Validadas (In-Distribution): {successful_routes}")
    logger.info(f"Tiempo Total: {total_time:.2f} segundos ({(total_time/N_TESTS):.3f}s por inferencia completa).")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()