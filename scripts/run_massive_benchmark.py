# scripts/run_massive_benchmark.py

import sys
import yaml
import os
import joblib
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cbadc as cb

sys.path.append('src')
from utils.logger import setup_logger
from models.phase3_gen import TabularDDPM
from models.phase4_surrogate import SurrogateResNet
from optimization.gradient_ascent import DifferentiableSearch
from utils.simulator import CBADCSimulator

# Importar la RNN del Baseline para poder cargar sus pesos
import torch.nn as nn
class BaselineRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=64):
        super(BaselineRNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, output_dim)
        )
    def forward(self, x): return self.net(x)

def load_mask_for_topology(topology_id, df_unified):
    row = df_unified[df_unified['topology_id'] == topology_id].iloc[0]
    mask_cols = sorted([c for c in df_unified.columns if c.startswith('mask_')])
    return row[mask_cols].values.astype(np.float32)

def calculate_fom(sndr, bw, power):
    return sndr + 10 * np.log10((bw / (power + 1e-12)) + 1e-12)

def plot_benchmark_results(results_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    valid_df = results_df[results_df["FoM_Real"] > 0]

    # 1. Comparativa de Tiempos de Cómputo
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df, x="Pipeline", y="Time_ms", palette="Set2")
    plt.yscale("log")
    plt.title("Computational Time Comparison (AI Inference Only)")
    plt.ylabel("Time (ms) [Log Scale]")
    plt.savefig(os.path.join(out_dir, "Benchmark_Time_Comparison.png"), dpi=300)
    plt.close()

    # 2. Distribución del FoM Real Simulado
    if not valid_df.empty:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=valid_df, x="FoM_Real", hue="Pipeline", fill=True, common_norm=False, palette="Set1", alpha=0.5)
        plt.title("Distribution of Achieved Figure of Merit (FoMs) in cbadc")
        plt.xlabel("Schreier FoM (dB)")
        plt.ylabel("Density")
        plt.savefig(os.path.join(out_dir, "Benchmark_FoM_Distribution.png"), dpi=300)
        plt.close()
        
        # 3. Gráfico de Dispersión
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=valid_df, x="Time_ms", y="FoM_Real", hue="Pipeline", style="Pipeline", s=100, palette="Set1", alpha=0.7)
        plt.xscale("log")
        plt.title("Efficiency Pareto: Real FoM vs Computational Cost")
        plt.xlabel("AI Computation Time (ms) [Log Scale]")
        plt.ylabel("Schreier FoM (dB)")
        plt.savefig(os.path.join(out_dir, "Benchmark_Pareto_Front.png"), dpi=300)
        plt.close()

def main():
    logger = setup_logger(name="benchmark_logger", log_file="logs/massive_benchmark.log")
    logger.info("INICIANDO BENCHMARK MASIVO (1000 SIMULACIONES)")
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    models_dir = config['paths']['models_dir']
    baseline_dir = os.path.join(models_dir, "baseline")
    device = torch.device("cpu")
    
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    spec_cols = config["data"]["input_specs"]
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    scaler_dv = joblib.load(os.path.join(models_dir, 'scaler_design_vars.pkl'))
    ood_model = joblib.load(os.path.join(models_dir, 'phase1_ood_models.pkl'))['OneClassSVM']
    
    clf_base = joblib.load(os.path.join(baseline_dir, 'baseline_classifier.pkl'))
    scaler_clf_base = joblib.load(os.path.join(baseline_dir, 'baseline_clf_scaler.pkl'))
    
    router = joblib.load(os.path.join(models_dir, 'phase2_router_model.pkl'))
    
    ddpm_state = torch.load(os.path.join(models_dir, 'phase3_tabularddpm.pth'), map_location=device)
    ddpm_hidden_dim = ddpm_state['net.0.weight'].shape[0]
    gen_model = TabularDDPM(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=ddpm_hidden_dim).to(device)
    gen_model.load_state_dict(ddpm_state)
    gen_model.eval()
    
    resnet_state = torch.load(os.path.join(models_dir, 'phase4_resnet.pth'), map_location=device)
    resnet_hidden_dim = resnet_state['in_proj.weight'].shape[0]
    surrogate_model = SurrogateResNet(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols), hidden_dim=resnet_hidden_dim).to(device)
    surrogate_model.load_state_dict(resnet_state)
    surrogate_model.eval()
    
    search_algo = DifferentiableSearch(surrogate_model, scaler_specs, device=device)
    lambda_penalty = config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda']
    cost_matrix = getattr(router, "cost_matrix", np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 0.1, 0.6, 1.1, 1.6, 2.1, 2.6]))
    simulator = CBADCSimulator()

    N_TESTS = 1000
    results_list = []
    
    logger.info("Generando especificaciones y simulando...")
    for i in tqdm(range(N_TESTS), desc="Benchmark Progress"):
        is_valid = False
        while not is_valid:
            sndr_req = np.random.uniform(60.0, 115.0)
            bw_req = np.random.uniform(1e6, 50e6)
            power_req = np.random.uniform(0.5e-3, 15e-3)
            
            x_input = np.array([[sndr_req, bw_req, power_req]])
            x_input_scaled = scaler_specs.transform(x_input)
            
            if ood_model.predict(x_input_scaled)[0] == 1:
                is_valid = True
                
        target_dict = {"SNDR": sndr_req, "Bw": bw_req, "Power": power_req}

        # ==========================================
        # EJECUCIÓN BASELINE
        # ==========================================
        start_base = time.time()
        x_base_sc = scaler_clf_base.transform(x_input)
        top_id_base = int(clf_base.predict(x_base_sc)[0])
        
        feature_names_base = joblib.load(os.path.join(baseline_dir, f'baseline_features_{top_id_base}.pkl'))
        rnn_base = BaselineRNN(len(spec_cols), len(feature_names_base)).to(device)
        rnn_base.load_state_dict(torch.load(os.path.join(baseline_dir, f'baseline_rnn_{top_id_base}.pth'), map_location=device))
        rnn_base.eval()
        
        scaler_X_base = joblib.load(os.path.join(baseline_dir, f'baseline_scaler_X_{top_id_base}.pkl'))
        scaler_y_base = joblib.load(os.path.join(baseline_dir, f'baseline_scaler_y_{top_id_base}.pkl'))
        
        with torch.no_grad():
            preds_base_sc = rnn_base(torch.tensor(scaler_X_base.transform(x_input), dtype=torch.float32).to(device))
            design_vars_base = scaler_y_base.inverse_transform(preds_base_sc.cpu().numpy())[0]
            
        time_base_ms = (time.time() - start_base) * 1000
        
        top_name_base = router.mapping_info.get(top_id_base, {}).get("name", "") if hasattr(router, "mapping_info") else ""
        sim_base = simulator.simulate(top_id_base, design_vars_base, target_dict, feature_names_base, top_name=top_name_base)
        fom_base = calculate_fom(sim_base["SNDR"], sim_base["Bw"], sim_base["Power"]) if sim_base["Success"] else 0
        
        results_list.append({
            "Test_ID": i, "Pipeline": "Baseline (2024)", "Topology": top_id_base,
            "Time_ms": time_base_ms, "FoM_Real": fom_base
        })

        # ==========================================
        # EJECUCIÓN NUEVO PIPELINE
        # ==========================================
        start_new = time.time()
        raw_probs = router.predict_proba(x_input_scaled)[0]
        adj_probs = (raw_probs * np.exp(-lambda_penalty * cost_matrix))
        top3_indices = np.argsort(adj_probs / np.sum(adj_probs))[-3:][::-1]
        
        best_fom_pred = -float('inf')
        best_design_new = None
        best_top_new = None
        
        for top_idx in top3_indices:
            topology_id = top_idx + 1
            one_hot = np.zeros((1, 12)); one_hot[0, top_idx] = 1
            one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
            cond_t = torch.cat([torch.tensor(x_input_scaled, dtype=torch.float32).to(device), one_hot_t], dim=1)
            
            with torch.no_grad():
                noise = torch.randn(1, super_vector_dim).to(device)
                initial_y_scaled = gen_model(noise, cond_t, torch.zeros(1,1).to(device))
                
            mask = load_mask_for_topology(topology_id, df_unified)
            best_y_scaled, best_fom = search_algo.optimize(initial_y_scaled, one_hot_t, mask, steps=30) 
            
            # --- PROTECCIÓN ANTI-DIVERGENCIA (FALLBACK) ---
            if best_y_scaled is None or np.isnan(best_fom):
                best_y_scaled = initial_y_scaled
                best_fom = -999.0
            # ----------------------------------------------
            
            if best_fom > best_fom_pred or best_design_new is None:
                best_fom_pred = best_fom
                best_design_new = best_y_scaled
                best_top_new = topology_id
                
        vars_real_new = scaler_dv.inverse_transform(best_design_new.cpu().numpy())[0]
        time_new_ms = (time.time() - start_new) * 1000
        
        top_name_new = router.mapping_info.get(best_top_new, {}).get("name", "") if hasattr(router, "mapping_info") else ""
        sim_new = simulator.simulate(best_top_new, vars_real_new, target_dict, dv_cols, top_name=top_name_new)
        fom_new = calculate_fom(sim_new["SNDR"], sim_new["Bw"], sim_new["Power"]) if sim_new["Success"] else 0
        
        results_list.append({
            "Test_ID": i, "Pipeline": "Proposed AI-Driven", "Topology": best_top_new,
            "Time_ms": time_new_ms, "FoM_Real": fom_new
        })

    logger.info("Benchmark finalizado. Generando gráficas para el paper...")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("logs/massive_benchmark_results.csv", index=False)
    plot_benchmark_results(results_df, "logs/plots/benchmark/")
    logger.info("¡Todo listo! Revisa logs/plots/benchmark/")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()