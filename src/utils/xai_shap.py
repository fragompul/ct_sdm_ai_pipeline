# src/utils/xai_shap.py

import shap
import numpy as np
import matplotlib.pyplot as plt
import os

class ShapleyExplainer:
    def __init__(self, trained_model, X_train_sample, feature_names):
        self.model = trained_model
        self.feature_names = feature_names
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("XAI: Usando TreeExplainer.")
        except Exception:
            predict_fn = lambda x: self.model.predict_proba(x)
            self.explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train_sample, 100))
            print("XAI: Usando KernelExplainer.")

    def generate_explanation(self, input_specs, predicted_topology_index, final_adj_prob=None, out_dir="logs/plots/xai/"):
        os.makedirs(out_dir, exist_ok=True)
        shap_values = self.explainer.shap_values(input_specs)
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]
        else:
            class_shap_values = shap_values[0, :, predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]

        # 1. Generar Force Plot
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            base_value=expected_value, 
            shap_values=class_shap_values, 
            features=input_specs, 
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"Lógica de Decisión para Topología {predicted_topology_index + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_explanation_top_{predicted_topology_index + 1}.png"))
        plt.close()

        # 2. Generar Texto Natural para el Diseñador Analógico 
        base_prob = expected_value * 100
        impacts = class_shap_values[0] if len(class_shap_values.shape) > 1 else class_shap_values
        
        text_report = f"--- INFORME DE INTELIGENCIA EXPLICABLE (XAI) ---\n"
        text_report += f"La probabilidad base (sesgo del dataset) para esta arquitectura es del {base_prob:.1f}%.\n"
        if final_prob: text_report += f"Sin embargo, la confianza final se ajustó al {final_prob*100:.1f}%. Esto se debe a:\n"
        
        for j, feature in enumerate(self.feature_names):
            impact_percent = impacts[j] * 100
            sign = "aumentó" if impact_percent > 0 else "penalizó"
            text_report += f"  • El requisito de {feature} ({input_specs[0][j]:.2e}) {sign} la probabilidad en un {abs(impact_percent):.2f}%.\n"
            
        return impacts, text_report