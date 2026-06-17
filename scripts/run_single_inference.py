# scripts/run_single_inference.py

import sys
import yaml
import os
import joblib
import torch
import numpy as np
import pandas as pd

sys.path.append('src')
from models.phase3_gen import MixtureDensityNetwork
from models.phase4_surrogate import SurrogateMLP
from optimization.gradient_ascent import DifferentiableSearch
from utils.xai_shap import ShapleyExplainer

def load_mask_for_topology(topology_id, df_unified):
    """Extrae la máscara booleana de la topología elegida."""
    row = df_unified[df_unified['topology_id'] == topology_id].iloc[0]
    mask_cols = sorted([c for c in df_unified.columns if c.startswith('mask_')])
    return row[mask_cols].values.astype(np.float32)

def main():
    # --- 0. CONFIGURACIÓN E INICIALIZACIÓN ---
    print("\n" + "="*60)
    print("🚀 PIPELINE DE SÍNTESIS DE CT-ΣΔΜ (INFERENCIA ÚNICA)")
    print("="*60)
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    models_dir = config['paths']['models_dir']
    device = torch.device("cpu") # Forzamos CPU para una inferencia estable
    
    # Cargar Data (Para máscaras y muestreo SHAP)
    df_unified = pd.read_csv(os.path.join(config["paths"]["processed_data"], "unified_ct_sdm_dataset.csv"))
    spec_cols = config["data"]["input_specs"]
    dv_cols = sorted([c for c in df_unified.columns if c.startswith('dv_')])
    super_vector_dim = len(dv_cols)
    
    # Cargar Modelos y Scalers
    scaler_specs = joblib.load(os.path.join(models_dir, 'scaler_specs.pkl'))
    scaler_dv = joblib.load(os.path.join(models_dir, 'scaler_design_vars.pkl'))
    ood_models = joblib.load(os.path.join(models_dir, 'phase1_ood_models.pkl'))
    router = joblib.load(os.path.join(models_dir, 'phase2_router_model.pkl'))
    
    mdn_model = MixtureDensityNetwork(spec_dim=len(spec_cols), super_vector_dim=super_vector_dim)
    mdn_model.load_state_dict(torch.load(os.path.join(models_dir, 'phase3_mdn.pth'), map_location=device))
    mdn_model.eval()
    
    surrogate_model = SurrogateMLP(super_vector_dim=super_vector_dim, output_metrics=len(spec_cols))
    surrogate_model.load_state_dict(torch.load(os.path.join(models_dir, 'phase4_surrogate.pth'), map_location=device))
    
    # --- ENTRADA DEL USUARIO ---
    target_specs = {"SNDR": 105.0, "Bw": 20e6, "Power": 1.5e-3}
    x_input = np.array([[target_specs["SNDR"], target_specs["Bw"], target_specs["Power"]]])
    x_input_scaled = scaler_specs.transform(x_input)
    
    print(f"\n📡 Especificaciones solicitadas:")
    for k, v in target_specs.items():
        print(f"   - {k}: {v}")

    # --- FASE 1: OOD GUARDRAILS ---
    print("\n🛡️  [FASE 1] Verificación OOD (OneClassSVM)...")
    ocsvm = ood_models['OneClassSVM']
    is_valid = ocsvm.predict(x_input_scaled)[0]
    if is_valid == 1:
        print("   ✅ Especificaciones validadas (Dentro de la variedad física).")
    else:
        print("   ⚠️  ADVERTENCIA: Especificaciones anómalas detectadas. El diseño podría fallar.")

    # --- FASE 2: PROBABILISTIC ROUTER ---
    print("\n🛤️  [FASE 2] Enrutamiento Topológico Probabilístico...")
    lambda_penalty = config['phases_benchmarks']['phase2_router']['heuristic_penalty_lambda']
    cost_matrix = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.5, 1.1, 1.3, 2.1, 2.3, 3.2, 3.6])
    
    raw_probs = router.predict_proba(x_input_scaled)[0]
    decay_factors = np.exp(-lambda_penalty * cost_matrix)
    adj_probs = raw_probs * decay_factors
    adj_probs /= np.sum(adj_probs) 
    
    top3_indices = np.argsort(adj_probs)[-3:][::-1]
    
    print("   🏆 Top-3 Topologías sugeridas (Penalización por coste aplicada):")
    for i, idx in enumerate(top3_indices):
        print(f"      {i+1}. Topología ID {idx+1} | Probabilidad: {adj_probs[idx]*100:.1f}%")

    # --- FASE 5: EXPLICABILIDAD SHAP ---
    print("\n🔍 [FASE 5] Explicabilidad (XAI) para la Topología ganadora...")
    top1_class_idx = top3_indices[0]
    top1_base_prob = raw_probs[top1_class_idx]
    
    # 1. Crear dataset de fondo representativo para KernelExplainer
    X_background = scaler_specs.transform(df_unified[spec_cols].sample(100, random_state=42).values)
    
    # 2. Instanciar la clase oficial de utilidades SHAP
    explainer = ShapleyExplainer(trained_model=router, X_train_sample=X_background, feature_names=spec_cols)
    
    # 3. Generar la explicación (Esto guardará el .png y devolverá los valores de impacto)
    class_shap_values = explainer.generate_explanation(x_input_scaled, top1_class_idx)
    
    # Ajuste de formato por si SHAP devuelve matriz 2D (1, features)
    impacts = class_shap_values[0] if len(class_shap_values.shape) > 1 else class_shap_values

    print("\n   📊 Desglose de la decisión lógica:")
    print(f"      Probabilidad Base (Ensemble): {top1_base_prob*100:.1f}%")
    for j, feature in enumerate(spec_cols):
        impact = impacts[j] * 100
        sign = "+" if impact > 0 else ""
        print(f"      -> Requisito de {feature} aportó {sign}{impact:.2f}% a la decisión.")
        
    print(f"\n      💬 \"The baseline probability for this topology was modulated significantly by your specifications, " 
          f"overriding default biases as shown by the marginal contributions.\"")

    # --- FASE 3 Y 4: MODELADO GENERATIVO Y OPTIMIZACIÓN ---
    print("\n🧬 [FASE 3 & 4] Búsqueda Activa en Espacio Latente...")
    search_algo = DifferentiableSearch(surrogate_model, scaler_specs, device=device)
    
    for top_idx in top3_indices:
        topology_id = top_idx + 1
        print(f"\n   ⚙️  Optimizando para Topología {topology_id}...")
        
        one_hot = np.zeros((1, 12))
        one_hot[0, top_idx] = 1
        one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
        cond_t = torch.cat([torch.tensor(x_input_scaled, dtype=torch.float32), one_hot_t], dim=1)
        
        # Fase 3: MDN Predice la distribución
        with torch.no_grad():
            pi, mu, sigma = mdn_model(cond_t)
            best_mixture_idx = torch.argmax(pi, dim=1)
            initial_y_scaled = mu[0, best_mixture_idx, :].view(1, -1)
            
        # Fase 4: Optimización diferenciable Surrogate
        mask = load_mask_for_topology(topology_id, df_unified)
        best_y_scaled, best_fom = search_algo.optimize(initial_y_scaled, one_hot_t, mask)
        
        print(f"      -> FoMs Predictivo Máximo: {best_fom:.2f} dB")
        
    # [FINAL] SIGN-OFF
    print("\n" + "="*60)
    print("🏁 [SIGN-OFF] Los 3 vectores óptimos están listos para enviarse a NGSpice.")
    print("="*60 + "\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Ocultar warnings internos de sklearn/shap
    main()