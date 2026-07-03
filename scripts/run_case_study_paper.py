# scripts/run_case_study_paper.py

import sys
import os
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

sys.path.append('src')
from utils.logger import setup_logger
from models.phase3_gen import TabularDDPM
from models.phase4_surrogate import SurrogateResNet
from optimization.gradient_ascent import DifferentiableSearch
from utils.xai_shap import ShapleyExplainer
from utils.simulator import CBADCSimulator

# Importar Baseline
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

def format_vector(vector_dict):
    """Auxiliar para formatear diccionarios de predicciones en consola."""
    out = ""
    for k, v in vector_dict.items():
        if abs(v) > 0 and abs(v) < 1e-4:
            out += f"{k}: {v:.2e} | "
        else:
            out += f"{k}: {v:.4f} | "
    return out[:-3]

def plot_routing_comparison(top_id_base, adj_probs_new, mapping_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_classes = len(adj_probs_new)
    labels = [mapping_info.get(i+1, {}).get("name", f"Top {i+1}") for i in range(n_classes)]
    
    baseline_probs = np.zeros(n_classes)
    baseline_probs[top_id_base - 1] = 1.0 
    
    x = np.arange(n_classes)
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_probs * 100, width, label='Baseline 2024 (Deterministic)', color='lightcoral')
    plt.bar(x + width/2, adj_probs_new * 100, width, label='Proposed AI-Driven (Probabilistic)', color='teal')
    
    plt.ylabel('Selection Confidence (%)', fontsize=12)
    plt.title('Topological Routing Comparison: Rigid vs. Probabilistic', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "CaseStudy_Routing_Comparison.png"), dpi=300)
    plt.close()

def run_case_study(target_specs):
    logger = setup_logger(name="case_study_logger", log_file="logs/case_study.log")
    
    logger.info("="*80)
    logger.info("🔬 CASE STUDY FOR THE PAPER (TCAS-I / TCAD)")
    logger.info("="*80)
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    models_dir = config['paths']['models_dir']
    baseline_dir = os.path.join(models_dir, "baseline")
    plots_dir = "logs/plots/case_study/"
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device("cpu")
    
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    spec_cols = config["data"]["input_specs"]
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    scaler_dv = joblib.load(os.path.join(models_dir, 'scaler_design_vars.pkl'))
    
    x_input = np.array([[target_specs[k] for k in spec_cols]])
    x_input_scaled = scaler_specs.transform(x_input)
    
    logger.info("📡 Target Specifications Input:")
    for k, v in target_specs.items(): logger.info(f"   ➤ {k}: {v}")

    # =========================================================================
    # 1. FLUJO BASELINE
    # =========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("🏛️  1. BASELINE EXECUTION (NO OOD, NO XAI, RIGID)")
    logger.info("-" * 80)
    
    clf_base = joblib.load(os.path.join(baseline_dir, 'baseline_classifier.pkl'))
    scaler_clf_base = joblib.load(os.path.join(baseline_dir, 'baseline_clf_scaler.pkl'))
    router = joblib.load(os.path.join(models_dir, 'phase2_router_model.pkl'))
    
    logger.info("   [Phase 1] OOD Check: BYPASSED (Baseline assumes feasibility)")
    
    x_base_sc = scaler_clf_base.transform(x_input)
    top_id_base = int(clf_base.predict(x_base_sc)[0])
    top_name_base = router.mapping_info.get(top_id_base, {}).get("name", "Unknown") if hasattr(router, "mapping_info") else f"Topology_{top_id_base}"
    
    logger.info(f"   [Phase 2] Routing: Rigid prediction of ID {top_id_base} [{top_name_base}] (100% confidence).")
    
    feature_names_base = joblib.load(os.path.join(baseline_dir, f'baseline_features_{top_id_base}.pkl'))
    rnn_base = BaselineRNN(len(spec_cols), len(feature_names_base)).to(device)
    rnn_base.load_state_dict(torch.load(os.path.join(baseline_dir, f'baseline_rnn_{top_id_base}.pth'), map_location=device))
    rnn_base.eval()
    scaler_X_base = joblib.load(os.path.join(baseline_dir, f'baseline_scaler_X_{top_id_base}.pkl'))
    scaler_y_base = joblib.load(os.path.join(baseline_dir, f'baseline_scaler_y_{top_id_base}.pkl'))
    
    with torch.no_grad():
        preds_base_sc = rnn_base(torch.tensor(scaler_X_base.transform(x_input), dtype=torch.float32).to(device))
        design_vars_base = scaler_y_base.inverse_transform(preds_base_sc.cpu().numpy())[0]
        
    dict_base_out = dict(zip(feature_names_base, design_vars_base))
    logger.info(f"   [Phase 3] RNN Raw Physical Output:")
    logger.info(f"             -> {format_vector(dict_base_out)}")

    # =========================================================================
    # 2. NUEVO PIPELINE PROPUESTO
    # =========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("🚀 2. PROPOSED PIPELINE EXECUTION (OOD + PROB + XAI + GEN)")
    logger.info("-" * 80)
    
    ood_model = joblib.load(os.path.join(models_dir, 'phase1_ood_models.pkl'))['OneClassSVM']
    
    ddpm_state = torch.load(os.path.join(models_dir, 'phase3_tabularddpm.pth'), map_location=device)
    gen_model = TabularDDPM(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim, hidden_dim=ddpm_state['net.0.weight'].shape[0]).to(device)
    gen_model.load_state_dict(ddpm_state)
    gen_model.eval()
    
    resnet_state = torch.load(os.path.join(models_dir, 'phase4_resnet.pth'), map_location=device)
    surrogate_model = SurrogateResNet(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols), hidden_dim=resnet_state['in_proj.weight'].shape[0]).to(device)
    surrogate_model.load_state_dict(resnet_state)
    surrogate_model.eval()
    search_algo = DifferentiableSearch(surrogate_model, scaler_specs, device=device)

    # --- Fase 1: OOD ---
    is_valid = ood_model.predict(x_input_scaled)[0]
    raw_ood_score = ood_model.decision_function(x_input_scaled)[0]
    logger.info(f"   [Phase 1] OOD Check: {'Valid' if is_valid == 1 else 'Anomalous'} (Margin score: {raw_ood_score:.4f})")

    # --- Fase 2: Probabilistic Router ---
    lambda_penalty = config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda']
    cost_matrix = getattr(router, "cost_matrix", np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 0.1, 0.6, 1.1, 1.6, 2.1, 2.6]))
    
    raw_probs = router.predict_proba(x_input_scaled)[0]
    adj_probs = (raw_probs * np.exp(-lambda_penalty * cost_matrix))
    adj_probs /= np.sum(adj_probs) 
    top3_indices = np.argsort(adj_probs)[-3:][::-1]
    top1_class_idx = top3_indices[0]
    
    logger.info("   [Phase 2] Routing: Probabilistic prediction computed.")
    for i, idx in enumerate(top3_indices):
        top_name = router.mapping_info.get(idx+1, {}).get("name", "Unknown") if hasattr(router, "mapping_info") else f"Top_{idx+1}"
        logger.info(f"             -> Rank {i+1}: ID {idx+1} [{top_name}] (Prob: {adj_probs[idx]*100:.2f}%)")

    mapping_info = getattr(router, "mapping_info", {})
    plot_routing_comparison(top_id_base, adj_probs, mapping_info, plots_dir)

    # --- Fase 5: Explicabilidad (XAI) ---
    logger.info("\n   [Phase 5] Explainability (XAI): Evaluating Top-1 Architecture...")
    X_background = scaler_specs.transform(df_unified[spec_cols].sample(150, random_state=42).values)
    explainer = ShapleyExplainer(trained_model=router, X_train_sample=X_background, feature_names=spec_cols)
    impacts, _ = explainer.generate_explanation(x_input_scaled, top1_class_idx, final_adj_prob=adj_probs[top1_class_idx], out_dir=plots_dir)
    
    ev = explainer.explainer.expected_value
    base_val = ev[top1_class_idx] if isinstance(ev, (list, np.ndarray)) else ev
    base_prob_pct = float(np.mean(base_val)) * 100
    
    logger.info("             📝 PROPOSED PAPER TEXT (XAI SECTION):")
    frase_xai = f"             \"The routing model assigned Topology ID {top1_class_idx+1} with a final probability of {adj_probs[top1_class_idx]*100:.1f}%. "
    frase_xai += f"The baseline probability for this class in the dataset is only {base_prob_pct:.1f}%. "
    frase_xai += "The prediction is explained as follows: "
    
    explicaciones = []
    for j, feature in enumerate(spec_cols):
        impact_pct = float(np.mean(impacts[j])) * 100
        accion = "positively pushed the prediction by" if impact_pct > 0 else "penalized the confidence by"
        explicaciones.append(f"the {feature} requirement ({target_specs[feature]:.2e}) {accion} {abs(impact_pct):.1f}%")
        
    frase_xai += ", and ".join(explicaciones) + ".\""
    logger.info(frase_xai)

    # --- Fase 3 y 4: Generación y Optimización ---
    logger.info(f"\n   [Phase 3] Generative Space (TabularDDPM): Generating Seed Vector for ID {top1_class_idx+1}...")
    top1_id = top1_class_idx + 1
    one_hot = np.zeros((1, 12)); one_hot[0, top1_class_idx] = 1
    one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
    cond_t = torch.cat([torch.tensor(x_input_scaled, dtype=torch.float32).to(device), one_hot_t], dim=1)
    
    with torch.no_grad():
        noise = torch.randn(1, super_vector_dim).to(device)
        initial_y_scaled = gen_model(noise, cond_t, torch.zeros(1,1).to(device))
        initial_y_real = scaler_dv.inverse_transform(initial_y_scaled.cpu().numpy())[0]
        
    mask = load_mask_for_topology(top1_id, df_unified)
    # Extraer valores reales (solo no-nulos) para el log
    seed_dict = {k: v for k, v in zip(dv_cols, initial_y_real) if mask[list(dv_cols).index(k)] > 0}
    logger.info(f"             -> Generated Raw Seed: {format_vector(seed_dict)}")
    
    logger.info("   [Phase 4] Optimization (ResNet Surrogate): Running Gradient Ascent...")
    best_y_scaled, best_fom = search_algo.optimize(initial_y_scaled, one_hot_t, mask, steps=50) 
    
    vars_real_new = scaler_dv.inverse_transform(best_y_scaled.cpu().numpy())[0]
    opt_dict = {k: v for k, v in zip(dv_cols, vars_real_new) if mask[list(dv_cols).index(k)] > 0}
    
    logger.info(f"             -> Optimized Output: {format_vector(opt_dict)}")
    logger.info(f"             -> Predicted FoM (Schreier): {best_fom:.2f} dB")
    
    # 3. Comprobación Final en Simulador (Sign-off)
    logger.info("\n   🏁 [Sign-Off] Physical Simulator Verification (cbadc)...")
    simulator = CBADCSimulator()
    top_name_new = router.mapping_info.get(top1_id, {}).get("name", "") if hasattr(router, "mapping_info") else ""
    
    sim_new = simulator.simulate(top1_id, vars_real_new, target_specs, dv_cols, top_name=top_name_new)
    if sim_new["Success"]:
        logger.info(f"      ✅ Validated! (Time: {sim_new['Simulation_Time']:.2f}s)")
        logger.info(f"      ✅ Real Simulated SNDR: {sim_new['SNDR']:.2f} dB")
    else:
        logger.info("      ⚠️ Simulator convergence error.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    mi_caso_de_estudio = {
        "SNDR": 100.0,
        "Bw": 10e6,
        "Power": 5e-3
    }
    
    run_case_study(mi_caso_de_estudio)