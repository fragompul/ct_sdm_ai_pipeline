# scripts/run_single_inference.py

import sys
import yaml
import os
import joblib
import torch
import numpy as np
import pandas as pd
import argparse
import time

sys.path.append('src')
from models.phase3_gen import MixtureDensityNetwork, TabularDDPM
from models.phase4_surrogate import SurrogateMLP, SurrogateResNet
from optimization.gradient_ascent import DifferentiableSearch
from utils.xai_shap import ShapleyExplainer
from utils.simulator import CBADCSimulator

def load_mask_for_topology(topology_id, df_unified):
    row = df_unified[df_unified['topology_id'] == topology_id].iloc[0]
    mask_cols = sorted([c for c in df_unified.columns if c.startswith('mask_')])
    return row[mask_cols].values.astype(np.float32)

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_baseline_inference(target_specs, config, device):
    print("\n" + "="*60)
    print("🏛️  EJECUTANDO FLUJO BASELINE CLÁSICO (PAPER 2024)")
    print("="*60)
    
    start_total = time.time()
    models_dir = os.path.join(config['paths']['models_dir'], "baseline")
    spec_cols = config["data"]["input_specs"]
    
    x_input = np.array([[target_specs[k] for k in spec_cols]])
    
    # 1. Clasificador Determinista 
    start_clf = time.time()
    scaler_clf = joblib.load(os.path.join(models_dir, 'baseline_clf_scaler.pkl'))
    clf = joblib.load(os.path.join(models_dir, 'baseline_classifier.pkl'))
    
    x_input_scaled = scaler_clf.transform(x_input)
    top_id = int(clf.predict(x_input_scaled)[0])
    clf_time = (time.time() - start_clf) * 1000
    
    print(f"\n🛤️  [PASO 1] Clasificador Determinista:")
    print(f"   -> Topología seleccionada (Hard Label): ID {top_id}")
    print(f"   -> Tiempo de inferencia: {clf_time:.2f} ms")
    
    # 2. Inferencia en RNN Específica
    start_rnn = time.time()
    scaler_X = joblib.load(os.path.join(models_dir, f'baseline_scaler_X_{top_id}.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, f'baseline_scaler_y_{top_id}.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, f'baseline_features_{top_id}.pkl'))
    
    import torch.nn as nn
    class BaselineRNN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_units=64):
            super(BaselineRNN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
                nn.Linear(hidden_units, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(),
                nn.Linear(hidden_units, hidden_units), nn.ReLU(),
                nn.Linear(hidden_units, output_dim)
            )
        def forward(self, x): return self.net(x)
        
    rnn = BaselineRNN(len(spec_cols), len(feature_names)).to(device)
    rnn.load_state_dict(torch.load(os.path.join(models_dir, f'baseline_rnn_{top_id}.pth'), map_location=device))
    rnn.eval()
    
    with torch.no_grad():
        x_rnn_scaled = scaler_X.transform(x_input)
        preds_scaled = rnn(torch.tensor(x_rnn_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        final_design_vars = scaler_y.inverse_transform(preds_scaled)[0]
        
    rnn_time = (time.time() - start_rnn) * 1000
    total_time = (time.time() - start_total) * 1000
    
    print(f"\n🧬 [PASO 2] Inferencia en RNN Aislada:")
    print(f"   -> Variables predichas para ID {top_id}:")
    for name, val in zip(feature_names, final_design_vars):
        print(f"      * {name}: {val:.4e}")
    print(f"   -> Tiempo de red: {rnn_time:.2f} ms")
    
    print("\n" + "="*60)
    print(f"⏱️  TIEMPO TOTAL BASELINE: {total_time:.2f} ms")
    print("="*60 + "\n")

def run_new_pipeline_inference(target_specs, config, device):
    print("\n" + "="*60)
    print("🚀 NUEVO PIPELINE GENERATIVO (AI-DRIVEN UNIFIED)")
    print("="*60)
    
    start_total = time.time()
    models_dir = config['paths']['models_dir']
    
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    spec_cols = config["data"]["input_specs"]
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    scaler_dv = joblib.load(os.path.join(models_dir, 'scaler_design_vars.pkl'))
    ood_models = joblib.load(os.path.join(models_dir, 'phase1_ood_models.pkl'))
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
    
    x_input = np.array([[target_specs[k] for k in spec_cols]])
    x_input_scaled = scaler_specs.transform(x_input)
    
    # --- FASE 1: OOD GUARDRAILS ---
    start_p1 = time.time()
    is_valid = ood_models['OneClassSVM'].predict(x_input_scaled)[0]
    p1_time = (time.time() - start_p1) * 1000
    print(f"\n🛡️  [FASE 1] Verificación OOD (OneClassSVM) [{p1_time:.2f} ms]:")
    print("   ✅ Validado." if is_valid == 1 else "   ⚠️ Anómalo.")

    # --- FASE 2: PROBABILISTIC ROUTER ---
    start_p2 = time.time()
    lambda_penalty = config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda']
    cost_matrix = getattr(router, "cost_matrix", np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 0.1, 0.6, 1.1, 1.6, 2.1, 2.6]))
    
    raw_probs = router.predict_proba(x_input_scaled)[0]
    adj_probs = (raw_probs * np.exp(-lambda_penalty * cost_matrix))
    adj_probs /= np.sum(adj_probs) 
    top3_indices = np.argsort(adj_probs)[-3:][::-1]
    p2_time = (time.time() - start_p2) * 1000
    
    print(f"\n🛤️  [FASE 2] Enrutamiento Topológico Probabilístico [{p2_time:.2f} ms]:")
    for i, idx in enumerate(top3_indices):
        print(f"      {i+1}. Topología ID {idx+1} | Probabilidad: {adj_probs[idx]*100:.1f}%")

    # --- FASE 5: EXPLICABILIDAD SHAP EN TEXTO NATURAL ---
    print("\n🔍 [FASE 5] Explicabilidad (XAI) para la Topología ganadora...")
    top1_class_idx = top3_indices[0]
    X_background = scaler_specs.transform(df_unified[spec_cols].sample(100, random_state=42).values)
    explainer = ShapleyExplainer(trained_model=router, X_train_sample=X_background, feature_names=spec_cols)
    
    impacts, texto_explicativo = explainer.generate_explanation(x_input_scaled, top1_class_idx, final_adj_prob=adj_probs[top1_class_idx])
    print(f"\n{texto_explicativo}")

    # --- FASE 3 Y 4: MODELADO GENERATIVO Y OPTIMIZACIÓN ---
    print("\n🧬 [FASE 3 & 4] Búsqueda Activa Subrogada en Espacio Unificado...")
    search_algo = DifferentiableSearch(surrogate_model, scaler_specs, device=device)
    simulator = CBADCSimulator()
    
    for top_idx in top3_indices:
        start_p34 = time.time()
        topology_id = top_idx + 1
        
        one_hot = np.zeros((1, 12))
        one_hot[0, top_idx] = 1
        one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
        cond_t = torch.cat([torch.tensor(x_input_scaled, dtype=torch.float32).to(device), one_hot_t], dim=1)
        
        with torch.no_grad():
            t = torch.zeros(1, 1).to(device)
            noise = torch.randn(1, super_vector_dim).to(device)
            initial_y_scaled = gen_model(noise, cond_t, t)
            
        mask = load_mask_for_topology(topology_id, df_unified)
        best_y_scaled, best_fom_pred = search_algo.optimize(initial_y_scaled, one_hot_t, mask, steps=50)
        
        # --- PROTECCIÓN ANTI-DIVERGENCIA ---
        if best_y_scaled is None or np.isnan(best_fom_pred):
            best_y_scaled = initial_y_scaled
            best_fom_pred = -999.0
            print(f"      [!] Optimizador divergió, aplicando semilla original del TabularDDPM.")
        # -----------------------------------
        
        best_y_real = scaler_dv.inverse_transform(best_y_scaled.cpu().numpy())[0]
        p34_time = (time.time() - start_p34) * 1000
        print(f"\n   ⚙️  ID {topology_id} optimizado por IA en {p34_time:.2f} ms -> FoMs PREDICTIVO: {best_fom_pred:.2f} dB")
        
        # =========================================================
        # SIGN-OFF: COMPROBACIÓN FÍSICA REAL CON CBADC
        # =========================================================
        top_name = router.mapping_info.get(topology_id, {}).get("name", "") if hasattr(router, "mapping_info") else ""
        sim_results = simulator.simulate(topology_id, best_y_real, target_specs, dv_cols, top_name=top_name)
        
        if sim_results["Success"]:
            fom_sim = sim_results["SNDR"] + 10 * np.log10((sim_results["Bw"] / (sim_results["Power"] + 1e-12)) + 1e-12)
            print(f"      ✅ [Simulador CBADC] Física comprobada en {sim_results['Simulation_Time']:.2f} s")
            print(f"         └─ SNDR Real: {sim_results['SNDR']:.2f} dB | Power Real: {sim_results['Power']*1000:.2f} mW")
            print(f"         └─ FoMs REAL CONFIRMADO: {fom_sim:.2f} dB")
        else:
            print(f"      ❌ [Simulador CBADC] Fallo en la convergencia.")

    total_time = (time.time() - start_total) * 1000
    print("\n" + "="*60)
    print(f"⏱️  TIEMPO TOTAL NUEVO PIPELINE: {total_time:.2f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description="Ejecutar Inferencia")
    parser.add_argument("--mode", type=str, choices=["baseline", "new", "both"], default="both", help="Modo de inferencia a ejecutar")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cpu")
    
    target_specs = {"SNDR": 105.0, "Bw": 20e6, "Power": 1.5e-3}
    
    print(f"📡 Especificaciones solicitadas:")
    for k, v in target_specs.items(): print(f"   - {k}: {v}")

    if args.mode in ["baseline", "both"]:
        run_baseline_inference(target_specs, config, device)
        
    if args.mode in ["new", "both"]:
        run_new_pipeline_inference(target_specs, config, device)